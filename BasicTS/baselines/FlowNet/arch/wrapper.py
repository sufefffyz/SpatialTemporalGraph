from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model.backbone import FlowNet


def _load_normalized_distance(path: str, num_nodes: int, normalize: str = "max") -> torch.Tensor:
    matrix_path = Path(path).expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(f"FlowNet distance matrix not found: {matrix_path}")
    dist = np.load(matrix_path).astype("float32")
    if dist.shape[0] < num_nodes or dist.shape[1] < num_nodes:
        raise ValueError(f"Distance matrix shape {dist.shape} is smaller than num_nodes={num_nodes}.")
    dist = dist[:num_nodes, :num_nodes]
    finite = np.isfinite(dist)
    if not finite.all():
        max_finite = float(np.nanmax(dist[finite])) if finite.any() else 1.0
        dist = np.where(finite, dist, max_finite)
    dist = np.maximum(dist, 0.0)
    np.fill_diagonal(dist, 0.0)

    positive = dist[dist > 0]
    if positive.size:
        if normalize == "p95":
            scale = float(np.quantile(positive, 0.95))
        elif normalize == "median":
            scale = float(np.median(positive))
        else:
            scale = float(np.max(positive))
        if scale > 0:
            dist = dist / scale
    return torch.from_numpy(dist.astype("float32"))


class BasicTSFlowNet(nn.Module):
    """BasicTS adapter for the official FlowNet backbone."""

    def __init__(self, **model_args):
        super().__init__()
        config = Namespace(
            seq_len=model_args["seq_len"],
            pred_len=model_args["pred_len"],
            patch_len=model_args.get("patch_len", 4),
            stride=model_args.get("stride", 2),
            moving_avg=model_args.get("moving_avg", 3),
            nhead=model_args.get("nhead", 4),
            ffn_dim=model_args.get("ffn_dim", 128),
            d_model=model_args.get("d_model", 64),
            n_expert=model_args.get("n_expert", 16),
            n_layer=model_args.get("n_layer", 2),
            dropout=model_args.get("dropout", 0.1),
            rate=model_args.get("rate", 4),
            enc_in=model_args["num_nodes"],
            dec_in=model_args["num_nodes"],
            c_out=model_args["num_nodes"],
            freq=model_args.get("freq", "5min"),
        )
        dist_mtx = _load_normalized_distance(
            model_args["dist_mtx_path"],
            model_args["num_nodes"],
            model_args.get("dist_norm", "max"),
        )
        self.backbone = FlowNet(config, dist_mtx)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        x_enc = history_data[..., 0]
        prediction = self.backbone(x_enc, None, None, None)
        return prediction.unsqueeze(-1)

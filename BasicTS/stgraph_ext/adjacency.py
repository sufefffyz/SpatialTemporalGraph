import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _extract_agcrn_adj(model: Any) -> torch.Tensor:
    node_embeddings = _unwrap_model(model).node_embeddings.detach()
    return F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)


def _extract_gwnet_adj(model: Any) -> torch.Tensor:
    core = _unwrap_model(model)
    return F.softmax(F.relu(torch.mm(core.nodevec1.detach(), core.nodevec2.detach())), dim=1)


def _extract_mtgnn_adj(model: Any) -> torch.Tensor:
    core = _unwrap_model(model)
    idx = core.idx.to(core.gc.emb1.weight.device)
    return core.gc.fullA(idx).detach()


def _extract_d2stgnn_adj(model: Any) -> torch.Tensor:
    core = _unwrap_model(model)
    return F.softmax(F.relu(torch.mm(core.node_emb_d.detach(), core.node_emb_u.detach().transpose(0, 1))), dim=1)


def _extract_gts_adj(runner: Any, sample_batch: dict) -> dict[str, torch.Tensor]:
    was_training = runner.model.training
    runner.model.eval()
    try:
        with torch.no_grad():
            forward_return = runner.forward(sample_batch, epoch=None, iter_num=None, train=False)
        result = {"adj": forward_return["pred_adj"].detach()}
        if "prior_adj" in forward_return:
            result["prior_adj"] = forward_return["prior_adj"].detach()
        return result
    finally:
        runner.model.train(was_training)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_learned_graph(runner: Any, sample_batch: dict | None = None) -> dict[str, np.ndarray]:
    model_name = runner.model_name.lower()
    if model_name == "agcrn":
        return {"adj": _to_numpy(_extract_agcrn_adj(runner.model))}
    if model_name == "graphwavenet":
        return {"adj": _to_numpy(_extract_gwnet_adj(runner.model))}
    if model_name == "mtgnn":
        return {"adj": _to_numpy(_extract_mtgnn_adj(runner.model))}
    if model_name == "d2stgnn":
        return {
            "adj": _to_numpy(_extract_d2stgnn_adj(runner.model)),
            "graph_semantics": np.asarray("static_directed"),
        }
    if model_name == "gts":
        if sample_batch is None:
            raise ValueError("GTS adjacency extraction requires a sample batch.")
        return {key: _to_numpy(value) for key, value in _extract_gts_adj(runner, sample_batch).items()}
    raise ValueError(f"Unsupported model for adjacency extraction: {runner.model_name}.")


def save_adjacency_snapshot(
    runner: Any,
    tag: str,
    epoch: int | str,
    sample_batch: dict | None = None,
) -> str:
    snapshot_dir = os.path.join(runner.ckpt_save_dir, "learned_graphs")
    os.makedirs(snapshot_dir, exist_ok=True)
    payload = extract_learned_graph(runner, sample_batch=sample_batch)
    payload["epoch"] = np.asarray(epoch)
    payload["model_name"] = np.asarray(runner.model_name)
    payload["dataset_name"] = np.asarray(runner.dataset_name)
    output_path = os.path.join(snapshot_dir, f"{tag}.npz")
    np.savez_compressed(output_path, **payload)
    return output_path

#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import pickle
from pathlib import Path

import numpy as np
import torch

from baselines.AdaptiveGraph import DynamicThresholdDCRNN, DynamicThresholdGraphWaveNet, DynamicThresholdSTGCN


def make_distance(path: Path, n: int) -> None:
    rng = np.random.default_rng(2026)
    xy = rng.normal(size=(n, 2)).astype("float32")
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).astype("float32")
    np.fill_diagonal(dist, 0.0)
    np.save(path, dist)


def make_candidate(path: Path, n: int, k: int = 3) -> None:
    adj = np.zeros((n, n), dtype="float32")
    for i in range(n):
        for step in range(1, k + 1):
            adj[i, (i + step) % n] = 1.0
    with path.open("wb") as f:
        pickle.dump(adj, f)


def dynamic_args(path: Path, mode: str, candidate_path: Path | None = None) -> dict:
    args = {
        "dist_mtx_path": str(path),
        "mode": mode,
        "d_model": 8,
        "target_avg_degree": 3.0,
        "radius_scale": 0.5,
        "temperature": 0.1,
        "dist_norm": "max",
    }
    if candidate_path is not None:
        args["candidate_adj_path"] = str(candidate_path)
    return args


def main() -> None:
    torch.manual_seed(2026)
    batch, seq_len, horizon, nodes = 2, 12, 12, 8
    history_1c = torch.randn(batch, seq_len, nodes, 1)
    history_2c = torch.randn(batch, seq_len, nodes, 2)
    future_1c = torch.randn(batch, horizon, nodes, 1)
    future_2c = torch.randn(batch, horizon, nodes, 2)

    with tempfile.TemporaryDirectory() as tmp:
        dist_path = Path(tmp) / "dist.npy"
        candidate_path = Path(tmp) / "adj_mx.pkl"
        make_distance(dist_path, nodes)
        make_candidate(candidate_path, nodes)
        for mode in ("soft", "hard"):
            for candidate in (None, candidate_path):
                candidate_label = "candidate" if candidate is not None else "dense"
                graph_args = dynamic_args(dist_path, mode, candidate)
                support = DynamicThresholdGraphWaveNet(
                    num_nodes=nodes,
                    seq_len=seq_len,
                    dynamic_graph=graph_args,
                    in_dim=2,
                    out_dim=horizon,
                    residual_channels=4,
                    dilation_channels=4,
                    skip_channels=8,
                    end_channels=16,
                    blocks=4,
                    layers=2,
                ).dynamic_support.transition_supports(history_2c)
                if candidate is None:
                    assert isinstance(support[0], torch.Tensor), (mode, candidate_label, "support_type")
                else:
                    assert not isinstance(support[0], torch.Tensor), (mode, candidate_label, "support_type")

            gwnet = DynamicThresholdGraphWaveNet(
                num_nodes=nodes,
                seq_len=seq_len,
                dynamic_graph=dynamic_args(dist_path, mode, candidate_path),
                in_dim=2,
                out_dim=horizon,
                residual_channels=4,
                dilation_channels=4,
                skip_channels=8,
                end_channels=16,
                blocks=4,
                layers=2,
            )
            y = gwnet(history_2c, future_2c, batch_seen=0, epoch=1, train=True)
            assert tuple(y.shape) == (batch, horizon, nodes, 1), (mode, "gwnet", tuple(y.shape))

            dcrnn = DynamicThresholdDCRNN(
                dynamic_graph=dynamic_args(dist_path, mode, candidate_path),
                cl_decay_steps=2000,
                horizon=horizon,
                input_dim=2,
                max_diffusion_step=2,
                num_nodes=nodes,
                num_rnn_layers=1,
                output_dim=1,
                rnn_units=8,
                seq_len=seq_len,
                use_curriculum_learning=True,
            )
            y = dcrnn(history_2c, future_2c, batch_seen=1)
            assert tuple(y.shape) == (batch, horizon, nodes, 1), (mode, "dcrnn", tuple(y.shape))

            stgcn = DynamicThresholdSTGCN(
                Ks=2,
                Kt=2,
                blocks=[[1], [8, 4, 8], [16, 16], [horizon]],
                T=seq_len,
                n_vertex=nodes,
                act_func="glu",
                graph_conv_type="cheb_graph_conv",
                dynamic_graph=dynamic_args(dist_path, mode),
                bias=True,
                droprate=0.1,
            )
            y = stgcn(history_1c, future_1c, batch_seen=0, epoch=1, train=True)
            assert tuple(y.shape) == (batch, horizon, nodes, 1), (mode, "stgcn", tuple(y.shape))
    print("dynamic threshold model smoke passed")


if __name__ == "__main__":
    main()

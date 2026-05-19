#!/usr/bin/env python3
from __future__ import annotations

import tempfile
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


def dynamic_args(path: Path, mode: str) -> dict:
    return {
        "dist_mtx_path": str(path),
        "mode": mode,
        "d_model": 8,
        "target_avg_degree": 3.0,
        "radius_scale": 0.5,
        "temperature": 0.1,
        "dist_norm": "max",
    }


def main() -> None:
    torch.manual_seed(2026)
    batch, seq_len, horizon, nodes = 2, 12, 12, 8
    history_1c = torch.randn(batch, seq_len, nodes, 1)
    history_2c = torch.randn(batch, seq_len, nodes, 2)
    future_1c = torch.randn(batch, horizon, nodes, 1)
    future_2c = torch.randn(batch, horizon, nodes, 2)

    with tempfile.TemporaryDirectory() as tmp:
        dist_path = Path(tmp) / "dist.npy"
        make_distance(dist_path, nodes)
        for mode in ("soft", "hard"):
            gwnet = DynamicThresholdGraphWaveNet(
                num_nodes=nodes,
                seq_len=seq_len,
                dynamic_graph=dynamic_args(dist_path, mode),
                in_dim=2,
                out_dim=horizon,
                residual_channels=4,
                dilation_channels=4,
                skip_channels=8,
                end_channels=16,
                blocks=1,
                layers=1,
            )
            y = gwnet(history_2c, future_2c, batch_seen=0, epoch=1, train=True)
            assert tuple(y.shape) == (batch, horizon, nodes, 1), (mode, "gwnet", tuple(y.shape))

            dcrnn = DynamicThresholdDCRNN(
                dynamic_graph=dynamic_args(dist_path, mode),
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

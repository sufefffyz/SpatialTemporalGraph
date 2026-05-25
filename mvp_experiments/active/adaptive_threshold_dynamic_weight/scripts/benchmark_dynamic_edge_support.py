#!/usr/bin/env python3
"""Benchmark dense vs candidate-edge dynamic threshold supports."""

from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
sys.path.insert(0, str(BASICTS_ROOT))

from baselines.AdaptiveGraph.dynamic_support import (  # noqa: E402
    DynamicThresholdSupport,
    dynamic_edge_support_matmul_3d,
    dynamic_edge_support_matmul_4d,
)


def make_inputs(work_dir: Path, num_nodes: int, k: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(20260525)
    xy = rng.normal(size=(num_nodes, 2)).astype("float32")
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).astype("float32")
    np.fill_diagonal(dist, 0.0)
    dist_path = work_dir / "dist.npy"
    np.save(dist_path, dist)

    order = np.argsort(dist, axis=1)
    adj = np.zeros((num_nodes, num_nodes), dtype="float32")
    for i in range(num_nodes):
        neighbors = order[i, 1 : k + 1]
        adj[i, neighbors] = 1.0
    adj_path = work_dir / "adj_mx.pkl"
    with adj_path.open("wb") as f:
        pickle.dump(adj, f)
    return dist_path, adj_path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / repeats


def spmm_loop_3d(support, x):
    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    num_nodes = support.num_nodes
    outs = []
    for batch_idx in range(x.shape[0]):
        sparse = torch.sparse_coo_tensor(edge_index, edge_weight[batch_idx], (num_nodes, num_nodes)).coalesce()
        outs.append(torch.sparse.mm(sparse, x[batch_idx]))
    return torch.stack(outs, dim=0)


def spmm_loop_4d(support, x):
    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    transposed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    batch_size, channels, num_nodes, steps = x.shape
    outs = []
    for batch_idx in range(batch_size):
        sparse_t = torch.sparse_coo_tensor(
            transposed_edge_index,
            edge_weight[batch_idx],
            (num_nodes, num_nodes),
        ).coalesce()
        x_flat = x[batch_idx].permute(1, 0, 2).reshape(num_nodes, channels * steps)
        y_flat = torch.sparse.mm(sparse_t, x_flat)
        outs.append(y_flat.reshape(num_nodes, channels, steps).permute(1, 0, 2))
    return torch.stack(outs, dim=0)


def block_spmm_3d(support, x):
    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    batch_size, num_nodes, features = x.shape
    edge_count = edge_index.shape[1]
    offsets = (torch.arange(batch_size, device=x.device) * num_nodes).view(-1, 1)
    rows = (edge_index[0].view(1, -1) + offsets).reshape(-1)
    cols = (edge_index[1].view(1, -1) + offsets).reshape(-1)
    block_index = torch.stack([rows, cols], dim=0)
    sparse = torch.sparse_coo_tensor(
        block_index,
        edge_weight.reshape(batch_size * edge_count),
        (batch_size * num_nodes, batch_size * num_nodes),
    ).coalesce()
    return torch.sparse.mm(sparse, x.reshape(batch_size * num_nodes, features)).reshape(batch_size, num_nodes, features)


def block_spmm_4d(support, x):
    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    batch_size, channels, num_nodes, steps = x.shape
    edge_count = edge_index.shape[1]
    offsets = (torch.arange(batch_size, device=x.device) * num_nodes).view(-1, 1)
    rows = (edge_index[1].view(1, -1) + offsets).reshape(-1)
    cols = (edge_index[0].view(1, -1) + offsets).reshape(-1)
    block_index = torch.stack([rows, cols], dim=0)
    sparse_t = torch.sparse_coo_tensor(
        block_index,
        edge_weight.reshape(batch_size * edge_count),
        (batch_size * num_nodes, batch_size * num_nodes),
    ).coalesce()
    x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, channels * steps)
    y_flat = torch.sparse.mm(sparse_t, x_flat)
    return y_flat.reshape(batch_size, num_nodes, channels, steps).permute(0, 2, 1, 3).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=716)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260525)
    with tempfile.TemporaryDirectory() as tmp:
        dist_path, adj_path = make_inputs(Path(tmp), args.num_nodes, args.k)
        graph_args = {
            "dist_mtx_path": str(dist_path),
            "candidate_adj_path": str(adj_path),
            "mode": "hard",
            "d_model": 8,
            "target_avg_degree": float(args.k),
            "radius_scale": 0.5,
            "temperature": 0.1,
            "dist_norm": "max",
            "edge_chunk_size": args.chunk_size,
        }
        dense_support = DynamicThresholdSupport(
            num_nodes=args.num_nodes,
            seq_len=args.seq_len,
            normalization="transition",
            **graph_args,
        ).to(device)
        dense_support.edge_output = False
        edge_support = DynamicThresholdSupport(
            num_nodes=args.num_nodes,
            seq_len=args.seq_len,
            normalization="transition",
            **graph_args,
        ).to(device)
        edge_support.load_state_dict(dense_support.state_dict())
        edge_support.edge_output = True

        history = torch.randn(args.batch_size, args.seq_len, args.num_nodes, 2, device=device)
        x3 = torch.randn(args.batch_size, args.num_nodes, args.features, device=device)
        x4 = torch.randn(args.batch_size, args.channels, args.num_nodes, args.steps, device=device)

        dense_supp = dense_support.transition_supports(history)
        edge_supp = edge_support.transition_supports(history)
        dense3 = torch.einsum("bij,bjf->bif", dense_supp[0], x3)
        edge3 = dynamic_edge_support_matmul_3d(edge_supp[0], x3)
        spmm3 = spmm_loop_3d(edge_supp[0], x3)
        block3 = block_spmm_3d(edge_supp[0], x3)
        dense4 = torch.einsum("bcvl,bvw->bcwl", x4, dense_supp[0])
        edge4 = dynamic_edge_support_matmul_4d(edge_supp[0], x4)
        spmm4 = spmm_loop_4d(edge_supp[0], x4)
        block4 = block_spmm_4d(edge_supp[0], x4)

        support_dense_s = time_call(lambda: dense_support.transition_supports(history), device, args.warmup, args.repeats)
        support_edge_s = time_call(lambda: edge_support.transition_supports(history), device, args.warmup, args.repeats)
        dcrnn_dense_s = time_call(lambda: torch.einsum("bij,bjf->bif", dense_supp[0], x3), device, args.warmup, args.repeats)
        dcrnn_edge_s = time_call(lambda: dynamic_edge_support_matmul_3d(edge_supp[0], x3), device, args.warmup, args.repeats)
        dcrnn_spmm_s = time_call(lambda: spmm_loop_3d(edge_supp[0], x3), device, args.warmup, args.repeats)
        dcrnn_block_s = time_call(lambda: block_spmm_3d(edge_supp[0], x3), device, args.warmup, args.repeats)
        gwnet_dense_s = time_call(lambda: torch.einsum("bcvl,bvw->bcwl", x4, dense_supp[0]), device, args.warmup, args.repeats)
        gwnet_edge_s = time_call(lambda: dynamic_edge_support_matmul_4d(edge_supp[0], x4), device, args.warmup, args.repeats)
        gwnet_spmm_s = time_call(lambda: spmm_loop_4d(edge_supp[0], x4), device, args.warmup, args.repeats)
        gwnet_block_s = time_call(lambda: block_spmm_4d(edge_supp[0], x4), device, args.warmup, args.repeats)

        print(f"device={device}")
        print(f"shape=B{args.batch_size} N{args.num_nodes} K{args.k} C{args.channels} L{args.steps} F{args.features}")
        print(f"edge_count={edge_supp[0].edge_index.shape[1]}")
        print(f"max_abs_diff_3d={float((dense3 - edge3).abs().max()):.8g}")
        print(f"max_abs_diff_3d_spmm={float((dense3 - spmm3).abs().max()):.8g}")
        print(f"max_abs_diff_3d_block={float((dense3 - block3).abs().max()):.8g}")
        print(f"max_abs_diff_4d={float((dense4 - edge4).abs().max()):.8g}")
        print(f"max_abs_diff_4d_spmm={float((dense4 - spmm4).abs().max()):.8g}")
        print(f"max_abs_diff_4d_block={float((dense4 - block4).abs().max()):.8g}")
        print(f"support_dense_ms={support_dense_s * 1000:.4f}")
        print(f"support_edge_ms={support_edge_s * 1000:.4f}")
        print(f"dcrnn_mul_dense_ms={dcrnn_dense_s * 1000:.4f}")
        print(f"dcrnn_mul_edge_ms={dcrnn_edge_s * 1000:.4f}")
        print(f"dcrnn_mul_spmm_loop_ms={dcrnn_spmm_s * 1000:.4f}")
        print(f"dcrnn_mul_block_spmm_ms={dcrnn_block_s * 1000:.4f}")
        print(f"gwnet_mul_dense_ms={gwnet_dense_s * 1000:.4f}")
        print(f"gwnet_mul_edge_ms={gwnet_edge_s * 1000:.4f}")
        print(f"gwnet_mul_spmm_loop_ms={gwnet_spmm_s * 1000:.4f}")
        print(f"gwnet_mul_block_spmm_ms={gwnet_block_s * 1000:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check dense-vs-sparse Graph WaveNet nconv equivalence and speed.

Run from the repository root, for example:

python mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/benchmark_sparse_gwnet_conv.py \
  --dataset SD_OSRMGG_B100 --device cuda --repeats 200
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
sys.path.insert(0, str(BASICTS_ROOT))

from baselines.GWNet.arch import PackedSparseSupport  # noqa: E402
from basicts.utils.adjacent_matrix_norm import calculate_transition_matrix  # noqa: E402


def load_raw_adj(dataset: str) -> np.ndarray:
    adj_path = BASICTS_ROOT / "datasets" / dataset / "adj_mx.pkl"
    with adj_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 3:
            obj = obj[2]
        elif len(obj) == 1:
            obj = obj[0]
    return np.asarray(obj, dtype=np.float32)


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
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SD_OSRMGG_B100")
    parser.add_argument("--support", choices=["forward", "backward", "raw"], default="forward")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    raw_adj = load_raw_adj(args.dataset)
    if args.support == "forward":
        support = calculate_transition_matrix(raw_adj).T
    elif args.support == "backward":
        support = calculate_transition_matrix(raw_adj.T).T
    else:
        support = raw_adj

    support = torch.tensor(support, dtype=torch.float32, device=device)
    sparse_support = PackedSparseSupport(support.cpu()).to(device)
    num_nodes = support.shape[0]
    x = torch.randn(
        args.batch_size,
        args.channels,
        num_nodes,
        args.steps,
        dtype=torch.float32,
        device=device,
    )

    def dense_call():
        return torch.einsum("ncvl,vw->ncwl", x, support)

    def sparse_call():
        return sparse_support(x)

    with torch.no_grad():
        dense_out = dense_call()
        sparse_out = sparse_call()
        max_abs_diff = (dense_out - sparse_out).abs().max().item()
        dense_s = time_call(dense_call, device, args.warmup, args.repeats)
        sparse_s = time_call(sparse_call, device, args.warmup, args.repeats)

    positive = support.abs() > 0
    nnz = int(positive.sum().item())
    max_in_degree = int(positive.sum(dim=0).max().item())
    avg_in_degree = float(positive.sum(dim=0).float().mean().item())
    speedup = dense_s / sparse_s if sparse_s > 0 else float("inf")

    print(f"dataset={args.dataset}")
    print(f"support={args.support}")
    print(f"device={device}")
    print(f"shape=B{args.batch_size} C{args.channels} N{num_nodes} L{args.steps}")
    print(f"nnz={nnz} avg_in_degree={avg_in_degree:.4f} max_in_degree={max_in_degree}")
    print(f"max_abs_diff={max_abs_diff:.8g}")
    print(f"dense_ms={dense_s * 1000:.4f}")
    print(f"sparse_ms={sparse_s * 1000:.4f}")
    print(f"speedup={speedup:.4f}x")


if __name__ == "__main__":
    main()

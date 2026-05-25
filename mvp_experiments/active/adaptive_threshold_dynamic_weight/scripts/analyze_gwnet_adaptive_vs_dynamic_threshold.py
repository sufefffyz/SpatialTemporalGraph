#!/usr/bin/env python3
"""Compare original adaptive GWNet and DynamicThreshold GWNet by node and graph degree.

The script is intended to run from the repository root or from BasicTS on the
server. It evaluates both checkpoints on the same BasicTS test loader. This
avoids relying on BasicTS saved raw memmaps, which can be misaligned when an
incomplete final batch is saved by batch index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _basic_ts_dir(repo_root: Path) -> Path:
    return repo_root / "BasicTS"


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def _decile_buckets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    edges = np.percentile(values, np.arange(10, 100, 10))
    return np.searchsorted(edges, values, side="right")


def _bucket_rows(
    bucket_values: np.ndarray,
    original_mae: np.ndarray,
    dynamic_mae: np.ndarray,
    original_degree: np.ndarray,
    dynamic_degree: np.ndarray,
) -> list[dict[str, float | int | str]]:
    buckets = _decile_buckets(bucket_values)
    rows: list[dict[str, float | int | str]] = []
    for b in range(10):
        idx = np.flatnonzero(buckets == b)
        if idx.size == 0:
            continue
        orig = original_mae[idx]
        dyn = dynamic_mae[idx]
        rows.append(
            {
                "bucket": b,
                "node_count": int(idx.size),
                "bucket_min": float(np.min(bucket_values[idx])),
                "bucket_max": float(np.max(bucket_values[idx])),
                "original_mae": float(np.mean(orig)),
                "dynamic_mae": float(np.mean(dyn)),
                "delta_dynamic_minus_original": float(np.mean(dyn - orig)),
                "mean_original_degree": float(np.mean(original_degree[idx])),
                "mean_dynamic_degree": float(np.mean(dynamic_degree[idx])),
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_sd_original_degree(basic_ts_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    adj_path = basic_ts_dir / "datasets" / "SD" / "adj_mx.pkl"
    with adj_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 3:
            raw_adj = obj[2]
        elif len(obj) == 1:
            raw_adj = obj[0]
        else:
            raw_adj = obj[-1]
    else:
        raw_adj = obj
    raw_adj = np.asarray(raw_adj, dtype=np.float32)
    active = raw_adj > 0
    eye = np.eye(active.shape[0], dtype=bool)
    out_degree = np.sum(active & ~eye, axis=1).astype(float)
    in_degree = np.sum(active.T & ~eye, axis=1).astype(float)
    return out_degree, in_degree


def _adaptive_topk_stats(ckpt_path: Path) -> dict[str, object]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state_dict"]
    if "nodevec1" not in state or "nodevec2" not in state:
        return {}
    adp = torch.softmax(torch.relu(state["nodevec1"] @ state["nodevec2"]), dim=1).cpu().numpy()
    topk_mass = {}
    for k in [1, 3, 5, 10, 20, 50]:
        topk = np.sort(adp, axis=1)[:, -k:]
        topk_mass[f"top{k}_mass_mean"] = float(np.mean(np.sum(topk, axis=1)))
    eff_degree = 1.0 / np.sum(np.square(adp), axis=1)
    entropy = -np.sum(adp * np.log(adp + 1e-12), axis=1)
    return {
        "adaptive_adj": {
            "effective_degree": _summary(eff_degree),
            "entropy": _summary(entropy),
            **topk_mass,
        }
    }


def _prepare_runner(
    cfg_path: str,
    ckpt_path: str,
    run_tag: str,
    seed: int,
    gpu: str,
    device: str,
    basic_ts_dir: Path,
):
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["BASICTS_RUN_TAG"] = run_tag
    os.environ["BASICTS_SEED"] = str(seed)

    sys.path.insert(0, str(basic_ts_dir))
    os.chdir(basic_ts_dir)

    from easytorch.config import init_cfg
    from easytorch.device import set_device_type
    from easytorch.utils import set_visible_devices

    set_device_type("gpu" if device.startswith("cuda") else "cpu")
    if device.startswith("cuda"):
        set_visible_devices(gpu)

    cfg = init_cfg(cfg_path, save=True)
    runner = cfg["RUNNER"](cfg)
    runner.init_test(cfg)
    runner.load_model(ckpt_path, strict=True)
    runner.model.eval()
    return cfg, runner


def _prepare_original_runner(args: argparse.Namespace, basic_ts_dir: Path):
    return _prepare_runner(
        args.original_cfg,
        args.original_adaptive_ckpt,
        args.original_run_tag,
        args.seed,
        args.gpu,
        args.device,
        basic_ts_dir,
    )


def _prepare_dynamic_runner(args: argparse.Namespace, basic_ts_dir: Path):
    os.environ["DYNAMIC_GRAPH_MODE"] = "hard"
    os.environ["DYNAMIC_GRAPH_WEIGHT_MODE"] = "binary"
    os.environ["DYNAMIC_GWNET_ADDAPTADJ"] = "1" if args.dynamic_addaptadj else "0"
    return _prepare_runner(
        args.dynamic_cfg,
        args.dynamic_ckpt,
        args.dynamic_run_tag,
        args.seed,
        args.gpu,
        args.device,
        basic_ts_dir,
    )


def _masked_abs_sum_count(pred: np.ndarray, target: np.ndarray, null_val: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    valid = np.abs(target - null_val) > 1e-5
    err = np.abs(pred - target) * valid
    return err.sum(axis=(0, 1, 3)), valid.sum(axis=(0, 1, 3))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1)


def _evaluate_runner_mae(runner, num_nodes: int, desc: str, null_val: float = 0.0) -> dict[str, np.ndarray]:
    sum_abs = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)
    targets = []

    with torch.no_grad():
        for batch in tqdm(runner.test_data_loader, desc=desc):
            forward_return = runner.forward(batch, epoch=None, iter_num=None, train=False)
            pred = forward_return["prediction"].detach().cpu().numpy().astype(np.float32)
            target = forward_return["target"].detach().cpu().numpy().astype(np.float32)
            targets.append(target)
            batch_sum, batch_count = _masked_abs_sum_count(pred, target, null_val)
            sum_abs += batch_sum
            count += batch_count

    return {
        "per_node_mae": _safe_divide(sum_abs, count),
        "targets": np.concatenate(targets, axis=0),
    }


def _evaluate_dynamic_and_degrees(args: argparse.Namespace, runner, num_nodes: int, null_val: float = 0.0):
    device = next(runner.model.parameters()).device
    sum_abs = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)
    dyn_targets = []
    dyn_preds = []
    degree_sum = np.zeros(num_nodes, dtype=np.float64)
    degree_sumsq = np.zeros(num_nodes, dtype=np.float64)
    indegree_sum = np.zeros(num_nodes, dtype=np.float64)
    sample_count = 0
    all_degree_values = []

    eye = torch.eye(num_nodes, dtype=torch.bool, device=device)

    with torch.no_grad():
        for batch in tqdm(runner.test_data_loader, desc="dynamic eval"):
            degree_batch = {
                "inputs": batch["inputs"].clone(),
                "target": batch["target"].clone(),
            }
            degree_batch = runner.preprocessing(degree_batch)
            history = runner.select_input_features(degree_batch["inputs"].to(device))
            weights = runner.model.dynamic_support._masked_weights(history)
            active = weights > 0.5
            active_no_self = active & ~eye.unsqueeze(0)
            out_degree = active_no_self.sum(dim=-1).float()
            in_degree = active_no_self.sum(dim=-2).float()
            degree_sum += out_degree.sum(dim=0).cpu().numpy()
            degree_sumsq += torch.square(out_degree).sum(dim=0).cpu().numpy()
            indegree_sum += in_degree.sum(dim=0).cpu().numpy()
            sample_count += out_degree.shape[0]
            all_degree_values.append(out_degree.cpu().numpy().reshape(-1))

            forward_return = runner.forward(batch, epoch=None, iter_num=None, train=False)
            pred = forward_return["prediction"].detach().cpu().numpy().astype(np.float32)
            target = forward_return["target"].detach().cpu().numpy().astype(np.float32)
            dyn_preds.append(pred)
            dyn_targets.append(target)
            batch_sum, batch_count = _masked_abs_sum_count(pred, target, null_val)
            sum_abs += batch_sum
            count += batch_count

    per_node_mae = _safe_divide(sum_abs, count)
    mean_degree = degree_sum / sample_count
    var_degree = np.maximum(degree_sumsq / sample_count - np.square(mean_degree), 0.0)
    std_degree = np.sqrt(var_degree)
    mean_indegree = indegree_sum / sample_count
    all_degrees = np.concatenate(all_degree_values)
    return {
        "per_node_mae": per_node_mae,
        "predictions": np.concatenate(dyn_preds, axis=0),
        "targets": np.concatenate(dyn_targets, axis=0),
        "dynamic_degree_mean": mean_degree,
        "dynamic_degree_std": std_degree,
        "dynamic_indegree_mean": mean_indegree,
        "all_dynamic_degrees": all_degrees,
    }


def _plot_outputs(output_dir: Path, node_rows: list[dict], bucket_rows: list[dict], dynamic_degrees: np.ndarray, original_degrees: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(original_degrees, bins=30, alpha=0.65, label="Original SD out-degree")
    ax.hist(dynamic_degrees, bins=30, alpha=0.65, label="Dynamic mean out-degree")
    ax.set_xlabel("Out-degree excluding self-loop")
    ax.set_ylabel("Node count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "degree_distribution_original_vs_dynamic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bucket_rows))
    width = 0.36
    ax.bar(x - width / 2, [r["original_mae"] for r in bucket_rows], width, label="Original adaptive")
    ax.bar(x + width / 2, [r["dynamic_mae"] for r in bucket_rows], width, label="DynamicThreshold")
    ax.set_xlabel("Original SD out-degree decile")
    ax.set_ylabel("Per-node MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mae_by_original_degree_decile.png", dpi=180)
    plt.close(fig)

    node_delta = np.array([r["delta_dynamic_minus_original"] for r in node_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(original_degrees, node_delta, s=12, alpha=0.65)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Original SD out-degree excluding self-loop")
    ax.set_ylabel("Dynamic MAE - original adaptive MAE")
    fig.tight_layout()
    fig.savefig(output_dir / "node_delta_vs_original_degree.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = _repo_root()
    basic_ts = _basic_ts_dir(repo_root)
    parser.add_argument("--original-cfg", default="baselines/GWNet/SD.py")
    parser.add_argument("--original-run-tag", default="sd_aligned_fresh_wandb_project_20260511_191840")
    parser.add_argument("--original-adaptive-ckpt", default=str(basic_ts / "checkpoints/GraphWaveNet/SD_original_100_12_12_sd_aligned_fresh_wandb_project_20260511_191840/ee9d39fd3520017a6931110f1a52414e/GraphWaveNet_best_val_MAE.pt"))
    parser.add_argument("--dynamic-cfg", default="baselines/GWNet/SD_dynamic_threshold.py")
    parser.add_argument("--dynamic-ckpt", default=str(basic_ts / "checkpoints/DynamicThresholdGraphWaveNet/SD_dynamic_threshold_hard_exp_tanh_gwnet_100_12_12_addaptadj_dynamic_threshold_hard_addaptadj_20260522_gpu1/94d620cfd224e6d2cf7cd2c344b1a3cd/DynamicThresholdGraphWaveNet_best_val_MAE.pt"))
    parser.add_argument("--dynamic-run-tag", default="dynamic_threshold_hard_addaptadj_20260522_gpu1")
    parser.add_argument("--dynamic-addaptadj", action="store_true", default=True)
    parser.add_argument("--output-dir", default=str(repo_root / "mvp_experiments/active/adaptive_threshold_dynamic_weight/results/gwnet_original_adaptive_vs_dynamic_threshold_addaptadj_live_20260525"))
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = 716

    _, original_runner = _prepare_original_runner(args, basic_ts)
    original = _evaluate_runner_mae(original_runner, num_nodes, "original adaptive eval")
    original_mae = original["per_node_mae"]
    num_samples = original["targets"].shape[0]

    _, runner = _prepare_dynamic_runner(args, basic_ts)
    dynamic = _evaluate_dynamic_and_degrees(args, runner, num_nodes)
    dynamic_mae = dynamic["per_node_mae"]

    target_match = bool(np.allclose(original["targets"], dynamic["targets"], atol=1e-4, rtol=1e-5))
    if not target_match:
        print("WARNING: original and dynamic targets are not exactly aligned.", file=sys.stderr)

    original_out_degree, original_in_degree = _load_sd_original_degree(basic_ts)
    dynamic_degree = dynamic["dynamic_degree_mean"]
    dynamic_degree_std = dynamic["dynamic_degree_std"]
    dynamic_indegree = dynamic["dynamic_indegree_mean"]
    delta_mae = dynamic_mae - original_mae
    degree_delta = dynamic_degree - original_out_degree

    node_rows = []
    for node in range(num_nodes):
        node_rows.append(
            {
                "node_index": node,
                "original_adaptive_mae": float(original_mae[node]),
                "dynamic_threshold_mae": float(dynamic_mae[node]),
                "delta_dynamic_minus_original": float(delta_mae[node]),
                "original_out_degree": float(original_out_degree[node]),
                "original_in_degree": float(original_in_degree[node]),
                "dynamic_out_degree_mean": float(dynamic_degree[node]),
                "dynamic_out_degree_std": float(dynamic_degree_std[node]),
                "dynamic_in_degree_mean": float(dynamic_indegree[node]),
                "degree_delta_dynamic_minus_original": float(degree_delta[node]),
            }
        )

    bucket_by_original_degree = _bucket_rows(original_out_degree, original_mae, dynamic_mae, original_out_degree, dynamic_degree)
    bucket_by_dynamic_degree = _bucket_rows(dynamic_degree, original_mae, dynamic_mae, original_out_degree, dynamic_degree)
    bucket_by_degree_delta = _bucket_rows(degree_delta, original_mae, dynamic_mae, original_out_degree, dynamic_degree)

    summary = {
        "target_match": target_match,
        "num_samples": int(num_samples),
        "num_nodes": num_nodes,
        "overall": {
            "original_adaptive_mae_mean_node": float(np.mean(original_mae)),
            "dynamic_threshold_mae_mean_node": float(np.mean(dynamic_mae)),
            "delta_dynamic_minus_original_mean_node": float(np.mean(delta_mae)),
            "dynamic_better_node_count": int(np.sum(delta_mae < 0)),
            "dynamic_worse_node_count": int(np.sum(delta_mae > 0)),
        },
        "degree": {
            "original_out_degree": _summary(original_out_degree),
            "dynamic_out_degree_mean_per_node": _summary(dynamic_degree),
            "dynamic_out_degree_all_samples": _summary(dynamic["all_dynamic_degrees"]),
            "degree_delta_dynamic_minus_original": _summary(degree_delta),
            "dynamic_degree_std_per_node": _summary(dynamic_degree_std),
        },
        "correlation": {
            "original_degree_vs_delta_mae_pearson": _pearson(original_out_degree, delta_mae),
            "original_degree_vs_delta_mae_spearman": _spearman(original_out_degree, delta_mae),
            "dynamic_degree_vs_delta_mae_pearson": _pearson(dynamic_degree, delta_mae),
            "dynamic_degree_vs_delta_mae_spearman": _spearman(dynamic_degree, delta_mae),
            "degree_delta_vs_delta_mae_pearson": _pearson(degree_delta, delta_mae),
            "degree_delta_vs_delta_mae_spearman": _spearman(degree_delta, delta_mae),
        },
        **_adaptive_topk_stats(Path(args.original_adaptive_ckpt)),
    }

    _write_csv(output_dir / "per_node_mae_degree.csv", node_rows)
    _write_csv(output_dir / "bucket_by_original_degree_decile.csv", bucket_by_original_degree)
    _write_csv(output_dir / "bucket_by_dynamic_degree_decile.csv", bucket_by_dynamic_degree)
    _write_csv(output_dir / "bucket_by_degree_delta_decile.csv", bucket_by_degree_delta)
    _write_csv(output_dir / "top_dynamic_worse_nodes.csv", sorted(node_rows, key=lambda r: r["delta_dynamic_minus_original"], reverse=True)[:30])
    _write_csv(output_dir / "top_dynamic_better_nodes.csv", sorted(node_rows, key=lambda r: r["delta_dynamic_minus_original"])[:30])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _plot_outputs(output_dir, node_rows, bucket_by_original_degree, dynamic_degree, original_out_degree)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()

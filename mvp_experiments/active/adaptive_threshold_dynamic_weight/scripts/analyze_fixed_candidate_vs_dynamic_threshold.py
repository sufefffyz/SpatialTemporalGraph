#!/usr/bin/env python3
"""Compare a fixed candidate graph GWNet against DynamicThreshold GWNet.

This is a post-hoc diagnostic script. It evaluates two checkpoints on aligned
BasicTS loaders and reports whether DynamicThreshold improves beyond the fixed
candidate graph itself.
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


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
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
    return _pearson(_rankdata(x), _rankdata(y))


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_adj_degree(basic_ts_dir: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = basic_ts_dir / "datasets" / dataset_name / "adj_mx.pkl"
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (list, tuple)):
        if len(obj) == 3:
            obj = obj[2]
        elif len(obj) == 1:
            obj = obj[0]
        else:
            obj = obj[-1]
    adj = np.asarray(obj, dtype=np.float32)
    active = adj > 0
    eye = np.eye(active.shape[0], dtype=bool)
    out_degree = np.sum(active & ~eye, axis=1).astype(float)
    in_degree = np.sum(active.T & ~eye, axis=1).astype(float)
    return out_degree, in_degree


def _masked_abs_sum_count(pred: np.ndarray, target: np.ndarray, null_val: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    valid = np.abs(target - null_val) > 1e-5
    err = np.abs(pred - target) * valid
    return err.sum(axis=(0, 1, 3)), valid.sum(axis=(0, 1, 3))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1)


def _prepare_runner(
    cfg_path: str,
    ckpt_path: str,
    env_updates: dict[str, str | None],
    seed: int,
    gpu: str,
    device: str,
    split: str,
    basic_ts_dir: Path,
    strict_load: bool,
):
    env_updates = {
        **env_updates,
        "WANDB_MODE": "disabled",
        "BASICTS_SEED": str(seed),
    }
    _set_env(env_updates)

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
    if split == "train":
        cfg.TRAIN.DATA.SHUFFLE = False
        runner.test_data_loader = runner.build_train_data_loader(cfg)
    elif split == "valid":
        runner.test_data_loader = runner.build_val_data_loader(cfg)
    elif split != "test":
        raise ValueError(f"Unsupported split: {split}")
    runner.load_model(ckpt_path, strict=strict_load)
    runner.model.eval()
    return cfg, runner


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
    return {"per_node_mae": _safe_divide(sum_abs, count), "targets": np.concatenate(targets, axis=0)}


def _evaluate_dynamic_and_degrees(runner, num_nodes: int, null_val: float = 0.0) -> dict[str, np.ndarray]:
    device = next(runner.model.parameters()).device
    sum_abs = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)
    targets = []
    degree_sum = np.zeros(num_nodes, dtype=np.float64)
    degree_sumsq = np.zeros(num_nodes, dtype=np.float64)
    sample_count = 0
    all_degree_values = []
    eye = torch.eye(num_nodes, dtype=torch.bool, device=device)

    with torch.no_grad():
        for batch in tqdm(runner.test_data_loader, desc="dynamic eval"):
            degree_batch = {"inputs": batch["inputs"].clone(), "target": batch["target"].clone()}
            degree_batch = runner.preprocessing(degree_batch)
            history = runner.select_input_features(degree_batch["inputs"].to(device))
            weights = runner.model.dynamic_support._masked_weights(history)
            active = (weights > 0.5) & ~eye.unsqueeze(0)
            out_degree = active.sum(dim=-1).float()
            degree_sum += out_degree.sum(dim=0).cpu().numpy()
            degree_sumsq += torch.square(out_degree).sum(dim=0).cpu().numpy()
            sample_count += out_degree.shape[0]
            all_degree_values.append(out_degree.cpu().numpy().reshape(-1))

            forward_return = runner.forward(batch, epoch=None, iter_num=None, train=False)
            pred = forward_return["prediction"].detach().cpu().numpy().astype(np.float32)
            target = forward_return["target"].detach().cpu().numpy().astype(np.float32)
            targets.append(target)
            batch_sum, batch_count = _masked_abs_sum_count(pred, target, null_val)
            sum_abs += batch_sum
            count += batch_count

    mean_degree = degree_sum / sample_count
    var_degree = np.maximum(degree_sumsq / sample_count - np.square(mean_degree), 0.0)
    return {
        "per_node_mae": _safe_divide(sum_abs, count),
        "targets": np.concatenate(targets, axis=0),
        "dynamic_degree_mean": mean_degree,
        "dynamic_degree_std": np.sqrt(var_degree),
        "all_dynamic_degrees": np.concatenate(all_degree_values),
    }


def _group_summary(name: str, idx: np.ndarray, fixed_degree: np.ndarray, dynamic_degree: np.ndarray, delta: np.ndarray) -> dict:
    return {
        "group": name,
        "node_count": int(idx.size),
        "delta_dynamic_minus_fixed": _summary(delta[idx]),
        "fixed_out_degree": _summary(fixed_degree[idx]),
        "dynamic_out_degree_mean": _summary(dynamic_degree[idx]),
        "degree_delta_dynamic_minus_fixed": _summary((dynamic_degree - fixed_degree)[idx]),
    }


def _plot_outputs(output_dir: Path, node_rows: list[dict], fixed_degree: np.ndarray, dynamic_degree: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    delta = np.array([row["delta_dynamic_minus_fixed"] for row in node_rows], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fixed_degree, bins=30, alpha=0.65, label="Fixed candidate out-degree")
    ax.hist(dynamic_degree, bins=30, alpha=0.65, label="Dynamic mean out-degree")
    ax.set_xlabel("Out-degree excluding self-loop")
    ax.set_ylabel("Node count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "degree_distribution_fixed_vs_dynamic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dynamic_degree, delta, s=12, alpha=0.65)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Dynamic mean out-degree")
    ax.set_ylabel("Dynamic MAE - fixed candidate MAE")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_vs_dynamic_degree.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    better = dynamic_degree[delta < 0]
    worse = dynamic_degree[delta > 0]
    ax.hist(better, bins=30, alpha=0.65, label=f"Better nodes ({better.size})")
    ax.hist(worse, bins=30, alpha=0.65, label=f"Worse nodes ({worse.size})")
    ax.set_xlabel("Dynamic mean out-degree")
    ax.set_ylabel("Node count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "better_worse_dynamic_degree_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = _repo_root()
    basic_ts = _basic_ts_dir(repo_root)
    parser.add_argument("--fixed-cfg", default="baselines/GWNet/SD_topk_prior.py")
    parser.add_argument("--fixed-ckpt", required=True)
    parser.add_argument("--fixed-dataset", default="SD_GSPTOPK_K064")
    parser.add_argument("--fixed-graph-tag", default="gspK64")
    parser.add_argument("--fixed-run-tag", default="simple_gsp_gwnet_fixed_20260525")
    parser.add_argument("--dynamic-cfg", default="baselines/GWNet/SD_dynamic_threshold.py")
    parser.add_argument("--dynamic-ckpt", required=True)
    parser.add_argument("--dynamic-dataset", default="SD")
    parser.add_argument("--dynamic-graph-tag", default="gspK64")
    parser.add_argument("--dynamic-run-tag", default="simple_gsp_dyn_noaddapt_20260525")
    parser.add_argument("--candidate-adj", default="datasets/SD_GSPTOPK_K064/adj_mx.pkl")
    parser.add_argument("--target-avg-degree", default="64")
    parser.add_argument("--dynamic-addaptadj", action="store_true", default=False)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = 716

    fixed_env = {
        "BASICTS_DATA_NAME": args.fixed_dataset,
        "BASICTS_GRAPH_TAG": args.fixed_graph_tag,
        "BASICTS_RUN_TAG": args.fixed_run_tag,
        "GWNET_ADDAPTADJ": "0",
    }
    _, fixed_runner = _prepare_runner(
        args.fixed_cfg,
        args.fixed_ckpt,
        fixed_env,
        args.seed,
        args.gpu,
        args.device,
        args.split,
        basic_ts,
        args.strict_load,
    )
    fixed = _evaluate_runner_mae(fixed_runner, num_nodes, "fixed candidate eval")

    dynamic_env = {
        "BASICTS_DATA_NAME": args.dynamic_dataset,
        "BASICTS_GRAPH_TAG": args.dynamic_graph_tag,
        "BASICTS_RUN_TAG": args.dynamic_run_tag,
        "DYNAMIC_GRAPH_MODE": "hard",
        "DYNAMIC_GRAPH_WEIGHT_MODE": "binary",
        "DYNAMIC_GRAPH_CANDIDATE_ADJ": args.candidate_adj,
        "DYNAMIC_GRAPH_TARGET_AVG_DEGREE": args.target_avg_degree,
        "DYNAMIC_GWNET_ADDAPTADJ": "1" if args.dynamic_addaptadj else "0",
    }
    _, dynamic_runner = _prepare_runner(
        args.dynamic_cfg,
        args.dynamic_ckpt,
        dynamic_env,
        args.seed,
        args.gpu,
        args.device,
        args.split,
        basic_ts,
        args.strict_load,
    )
    dynamic = _evaluate_dynamic_and_degrees(dynamic_runner, num_nodes)

    target_match = bool(np.allclose(fixed["targets"], dynamic["targets"], atol=1e-4, rtol=1e-5))
    fixed_degree, fixed_indegree = _load_adj_degree(basic_ts, args.fixed_dataset)
    dynamic_degree = dynamic["dynamic_degree_mean"]
    delta = dynamic["per_node_mae"] - fixed["per_node_mae"]
    degree_delta = dynamic_degree - fixed_degree

    node_rows = []
    for node in range(num_nodes):
        node_rows.append(
            {
                "node_index": node,
                "fixed_candidate_mae": float(fixed["per_node_mae"][node]),
                "dynamic_threshold_mae": float(dynamic["per_node_mae"][node]),
                "delta_dynamic_minus_fixed": float(delta[node]),
                "fixed_out_degree": float(fixed_degree[node]),
                "fixed_in_degree": float(fixed_indegree[node]),
                "dynamic_out_degree_mean": float(dynamic_degree[node]),
                "dynamic_out_degree_std": float(dynamic["dynamic_degree_std"][node]),
                "degree_delta_dynamic_minus_fixed": float(degree_delta[node]),
            }
        )

    better_idx = np.flatnonzero(delta < 0)
    worse_idx = np.flatnonzero(delta > 0)
    same_idx = np.flatnonzero(delta == 0)
    group_rows = [
        _group_summary("better", better_idx, fixed_degree, dynamic_degree, delta),
        _group_summary("worse", worse_idx, fixed_degree, dynamic_degree, delta),
        _group_summary("same", same_idx, fixed_degree, dynamic_degree, delta) if same_idx.size else None,
    ]
    group_rows = [row for row in group_rows if row is not None]

    summary = {
        "target_match": target_match,
        "split": args.split,
        "num_nodes": num_nodes,
        "overall": {
            "fixed_candidate_mae_mean_node": float(np.mean(fixed["per_node_mae"])),
            "dynamic_threshold_mae_mean_node": float(np.mean(dynamic["per_node_mae"])),
            "delta_dynamic_minus_fixed_mean_node": float(np.mean(delta)),
            "dynamic_better_node_count": int(better_idx.size),
            "dynamic_worse_node_count": int(worse_idx.size),
        },
        "degree": {
            "fixed_out_degree": _summary(fixed_degree),
            "dynamic_out_degree_mean_per_node": _summary(dynamic_degree),
            "dynamic_out_degree_all_samples": _summary(dynamic["all_dynamic_degrees"]),
            "degree_delta_dynamic_minus_fixed": _summary(degree_delta),
            "dynamic_degree_std_per_node": _summary(dynamic["dynamic_degree_std"]),
        },
        "group_summary": group_rows,
        "correlation": {
            "fixed_degree_vs_delta_mae_pearson": _pearson(fixed_degree, delta),
            "fixed_degree_vs_delta_mae_spearman": _spearman(fixed_degree, delta),
            "dynamic_degree_vs_delta_mae_pearson": _pearson(dynamic_degree, delta),
            "dynamic_degree_vs_delta_mae_spearman": _spearman(dynamic_degree, delta),
            "degree_delta_vs_delta_mae_pearson": _pearson(degree_delta, delta),
            "degree_delta_vs_delta_mae_spearman": _spearman(degree_delta, delta),
        },
    }

    _write_csv(output_dir / "per_node_fixed_vs_dynamic_mae_degree.csv", node_rows)
    _write_csv(output_dir / "top_dynamic_better_nodes.csv", sorted(node_rows, key=lambda r: r["delta_dynamic_minus_fixed"])[:30])
    _write_csv(output_dir / "top_dynamic_worse_nodes.csv", sorted(node_rows, key=lambda r: r["delta_dynamic_minus_fixed"], reverse=True)[:30])
    flat_group_rows = []
    for row in group_rows:
        flat = {"group": row["group"], "node_count": row["node_count"]}
        for section, values in row.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat[f"{section}_{key}"] = value
        flat_group_rows.append(flat)
    _write_csv(output_dir / "better_worse_degree_summary.csv", flat_group_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _plot_outputs(output_dir, node_rows, fixed_degree, dynamic_degree)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""General node-level analysis for DynamicThreshold vs original adaptive GWNet.

This script combines broad statistics with automatically selected case studies.
It starts from the corrected live-evaluation per-node error table and enriches it
with flow, metadata, original graph, dynamic graph, distance, and overlap
features. Dynamic graph features are recomputed on the server from the trained
checkpoint so case studies are not based on stale saved memmaps.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _basic_ts_dir(repo_root: Path) -> Path:
    return repo_root / "BasicTS"


def _default_result_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "mvp_experiments/active/adaptive_threshold_dynamic_weight/results/"
        / "gwnet_original_adaptive_vs_dynamic_threshold_addaptadj_live_20260525"
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return _pearson(_rankdata(x[mask]), _rankdata(y[mask]))


def _load_original_adj(basic_ts_dir: Path) -> np.ndarray:
    adj_path = basic_ts_dir / "datasets/SD/adj_mx.pkl"
    with adj_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)):
        raw_adj = obj[2] if len(obj) == 3 else obj[-1]
    else:
        raw_adj = obj
    raw_adj = np.asarray(raw_adj, dtype=np.float32)
    active = raw_adj > 0
    np.fill_diagonal(active, False)
    return active


def _node_flow_stats(data: np.ndarray, prefix: str) -> pd.DataFrame:
    mask = data > 0
    valid_count = mask.sum(axis=0).astype(float)
    mean = (data * mask).sum(axis=0) / np.maximum(valid_count, 1)
    std = np.sqrt(((data - mean) ** 2 * mask).sum(axis=0) / np.maximum(valid_count, 1))
    p50 = np.zeros(data.shape[1], dtype=float)
    p95 = np.zeros(data.shape[1], dtype=float)
    for node in range(data.shape[1]):
        vals = data[:, node][mask[:, node]]
        p50[node] = np.percentile(vals, 50) if vals.size else np.nan
        p95[node] = np.percentile(vals, 95) if vals.size else np.nan
    adjacent_valid = (data[:-1] > 0) & (data[1:] > 0)
    lag1_absdiff = (
        np.abs(data[1:] - data[:-1]) * adjacent_valid
    ).sum(axis=0) / np.maximum(adjacent_valid.sum(axis=0), 1)

    steps_per_day = 96
    usable = (data.shape[0] // steps_per_day) * steps_per_day
    daily_amp = np.full(data.shape[1], np.nan, dtype=float)
    if usable:
        daily = data[:usable].reshape(-1, steps_per_day, data.shape[1])
        daily_mask = daily > 0
        sums = (daily * daily_mask).sum(axis=0)
        counts = np.maximum(daily_mask.sum(axis=0), 1)
        profile = sums / counts
        daily_amp = np.nanpercentile(profile, 95, axis=0) - np.nanpercentile(profile, 5, axis=0)

    return pd.DataFrame(
        {
            f"{prefix}_valid_mean": mean,
            f"{prefix}_valid_std": std,
            f"{prefix}_cv": std / np.maximum(mean, 1e-6),
            f"{prefix}_zero_rate": 1.0 - valid_count / data.shape[0],
            f"{prefix}_p50": p50,
            f"{prefix}_p95": p95,
            f"{prefix}_lag1_absdiff": lag1_absdiff,
            f"{prefix}_daily_amp": daily_amp,
        }
    )


def _static_neighbor_features(
    original_adj: np.ndarray,
    osrm_m: np.ndarray,
    straight_m: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    n = original_adj.shape[0]
    deg = original_adj.sum(axis=1).astype(float)
    same_fwy = (meta["FwyBase"].to_numpy()[:, None] == meta["FwyBase"].to_numpy()[None, :])
    same_dir = (meta["Direction"].to_numpy()[:, None] == meta["Direction"].to_numpy()[None, :])
    np.fill_diagonal(same_fwy, False)
    np.fill_diagonal(same_dir, False)
    denom = np.maximum(deg, 1)
    return pd.DataFrame(
        {
            "original_osrm_mean_m": (original_adj * osrm_m).sum(axis=1) / denom,
            "original_straight_mean_m": (original_adj * straight_m).sum(axis=1) / denom,
            "original_same_fwy_share": (original_adj & same_fwy).sum(axis=1) / denom,
            "original_same_dir_share": (original_adj & same_dir).sum(axis=1) / denom,
            "original_is_isolated": (deg == 0).astype(float),
        }
    )


def _prepare_dynamic_runner(args: argparse.Namespace, basic_ts_dir: Path):
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["BASICTS_RUN_TAG"] = args.dynamic_run_tag
    os.environ["BASICTS_SEED"] = str(args.seed)
    os.environ["DYNAMIC_GRAPH_MODE"] = "hard"
    os.environ["DYNAMIC_GRAPH_WEIGHT_MODE"] = "binary"
    os.environ["DYNAMIC_GWNET_ADDAPTADJ"] = "1" if args.dynamic_addaptadj else "0"

    sys.path.insert(0, str(basic_ts_dir))
    os.chdir(basic_ts_dir)

    from easytorch.config import init_cfg
    from easytorch.device import set_device_type
    from easytorch.utils import set_visible_devices

    set_device_type("gpu" if args.device.startswith("cuda") else "cpu")
    if args.device.startswith("cuda"):
        set_visible_devices(args.gpu)

    cfg = init_cfg(args.dynamic_cfg, save=True)
    runner = cfg["RUNNER"](cfg)
    runner.init_test(cfg)
    runner.load_model(args.dynamic_ckpt, strict=True)
    runner.model.eval()
    return runner


def _dynamic_graph_features(
    args: argparse.Namespace,
    basic_ts_dir: Path,
    original_adj: np.ndarray,
    osrm_m: np.ndarray,
    straight_m: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    runner = _prepare_dynamic_runner(args, basic_ts_dir)
    device = next(runner.model.parameters()).device
    n = original_adj.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=device)
    original = torch.from_numpy(original_adj).to(device=device, dtype=torch.bool)
    osrm = torch.from_numpy(osrm_m.astype("float32")).to(device)
    straight = torch.from_numpy(straight_m.astype("float32")).to(device)
    same_fwy = torch.from_numpy(
        (meta["FwyBase"].to_numpy()[:, None] == meta["FwyBase"].to_numpy()[None, :])
    ).to(device)
    same_dir = torch.from_numpy(
        (meta["Direction"].to_numpy()[:, None] == meta["Direction"].to_numpy()[None, :])
    ).to(device)
    same_fwy.fill_diagonal_(False)
    same_dir.fill_diagonal_(False)

    sample_count = 0
    edge_count = np.zeros(n, dtype=np.float64)
    edge_count_sq_per_sample = np.zeros(n, dtype=np.float64)
    indegree_count = np.zeros(n, dtype=np.float64)
    osrm_sum = np.zeros(n, dtype=np.float64)
    straight_sum = np.zeros(n, dtype=np.float64)
    max_osrm_sum = np.zeros(n, dtype=np.float64)
    added_osrm_sum = np.zeros(n, dtype=np.float64)
    added_count = np.zeros(n, dtype=np.float64)
    overlap_count = np.zeros(n, dtype=np.float64)
    same_fwy_count = np.zeros(n, dtype=np.float64)
    same_dir_count = np.zeros(n, dtype=np.float64)

    with torch.no_grad():
        for batch in tqdm(runner.test_data_loader, desc="dynamic graph feature scan"):
            graph_batch = {"inputs": batch["inputs"].clone(), "target": batch["target"].clone()}
            graph_batch = runner.preprocessing(graph_batch)
            history = runner.select_input_features(graph_batch["inputs"].to(device))
            weights = runner.model.dynamic_support._masked_weights(history)
            active = (weights > 0.5) & ~eye.unsqueeze(0)

            out_degree = active.sum(dim=-1).float()
            indegree = active.sum(dim=-2).float()
            added = active & ~original.unsqueeze(0)
            overlap = active & original.unsqueeze(0)

            sample_count += int(active.shape[0])
            edge_count += out_degree.sum(dim=0).cpu().numpy()
            edge_count_sq_per_sample += torch.square(out_degree).sum(dim=0).cpu().numpy()
            indegree_count += indegree.sum(dim=0).cpu().numpy()
            osrm_sum += (active.float() * osrm.unsqueeze(0)).sum(dim=-1).sum(dim=0).cpu().numpy()
            straight_sum += (active.float() * straight.unsqueeze(0)).sum(dim=-1).sum(dim=0).cpu().numpy()
            max_osrm_sum += osrm.unsqueeze(0).masked_fill(~active, 0).amax(dim=-1).sum(dim=0).cpu().numpy()
            added_osrm_sum += (added.float() * osrm.unsqueeze(0)).sum(dim=-1).sum(dim=0).cpu().numpy()
            added_count += added.sum(dim=-1).sum(dim=0).cpu().numpy()
            overlap_count += overlap.sum(dim=-1).sum(dim=0).cpu().numpy()
            same_fwy_count += (active & same_fwy.unsqueeze(0)).sum(dim=-1).sum(dim=0).cpu().numpy()
            same_dir_count += (active & same_dir.unsqueeze(0)).sum(dim=-1).sum(dim=0).cpu().numpy()

    mean_degree = edge_count / sample_count
    degree_std = np.sqrt(
        np.maximum(edge_count_sq_per_sample / sample_count - np.square(mean_degree), 0.0)
    )
    original_degree = original_adj.sum(axis=1).astype(float)
    return pd.DataFrame(
        {
            "dynamic_out_degree_live": mean_degree,
            "dynamic_out_degree_live_std": degree_std,
            "dynamic_in_degree_live": indegree_count / sample_count,
            "dynamic_osrm_mean_m": osrm_sum / np.maximum(edge_count, 1),
            "dynamic_straight_mean_m": straight_sum / np.maximum(edge_count, 1),
            "dynamic_osrm_max_mean_m": max_osrm_sum / sample_count,
            "dynamic_added_osrm_mean_m": added_osrm_sum / np.maximum(added_count, 1),
            "dynamic_added_degree": added_count / sample_count,
            "dynamic_overlap_precision": overlap_count / np.maximum(edge_count, 1),
            "dynamic_overlap_recall": overlap_count / np.maximum(original_degree * sample_count, 1),
            "dynamic_same_fwy_share": same_fwy_count / np.maximum(edge_count, 1),
            "dynamic_same_dir_share": same_dir_count / np.maximum(edge_count, 1),
        }
    )


def _cohen_d(worse: np.ndarray, better: np.ndarray) -> float:
    worse = worse[np.isfinite(worse)]
    better = better[np.isfinite(better)]
    if worse.size < 2 or better.size < 2:
        return float("nan")
    pooled = np.sqrt(((worse.size - 1) * np.var(worse, ddof=1) + (better.size - 1) * np.var(better, ddof=1)) / (worse.size + better.size - 2))
    return float((np.mean(worse) - np.mean(better)) / pooled) if pooled > 0 else float("nan")


def _bootstrap_mean_diff(worse: np.ndarray, better: np.ndarray, rng: np.random.Generator, rounds: int = 2000) -> tuple[float, float]:
    worse = worse[np.isfinite(worse)]
    better = better[np.isfinite(better)]
    if worse.size == 0 or better.size == 0:
        return float("nan"), float("nan")
    vals = np.empty(rounds, dtype=float)
    for i in range(rounds):
        vals[i] = np.mean(rng.choice(worse, worse.size, replace=True)) - np.mean(
            rng.choice(better, better.size, replace=True)
        )
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _group_summary(feature_table: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260525)
    better = feature_table[feature_table["group"] == "dynamic_better"]
    worse = feature_table[feature_table["group"] == "dynamic_worse"]
    strong_better = feature_table[feature_table["strong_group"] == "strong_better"]
    strong_worse = feature_table[feature_table["strong_group"] == "strong_worse"]
    for feature in feature_cols:
        b = better[feature].to_numpy(dtype=float)
        w = worse[feature].to_numpy(dtype=float)
        sb = strong_better[feature].to_numpy(dtype=float)
        sw = strong_worse[feature].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_diff(w, b, rng)
        rows.append(
            {
                "feature": feature,
                "better_mean": np.nanmean(b),
                "worse_mean": np.nanmean(w),
                "worse_minus_better": np.nanmean(w) - np.nanmean(b),
                "worse_minus_better_ci95_low": ci_low,
                "worse_minus_better_ci95_high": ci_high,
                "cohen_d_worse_vs_better": _cohen_d(w, b),
                "better_median": np.nanmedian(b),
                "worse_median": np.nanmedian(w),
                "strong_better_mean": np.nanmean(sb),
                "strong_worse_mean": np.nanmean(sw),
                "strong_worse_minus_strong_better": np.nanmean(sw) - np.nanmean(sb),
            }
        )
    return pd.DataFrame(rows).sort_values("cohen_d_worse_vs_better", key=lambda s: np.abs(s), ascending=False)


def _correlation_table(feature_table: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    delta = feature_table["delta_dynamic_minus_original"].to_numpy(dtype=float)
    rows = []
    for feature in feature_cols:
        values = feature_table[feature].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "pearson_with_delta": _pearson(values, delta),
                "spearman_with_delta": _spearman(values, delta),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_with_delta", key=lambda s: np.abs(s), ascending=False)


def _bin_prevalence(feature_table: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        values = feature_table[feature]
        if values.nunique(dropna=True) < 3:
            continue
        try:
            bins = pd.qcut(values, q=5, duplicates="drop")
        except ValueError:
            continue
        temp = feature_table.assign(_bin=bins)
        for bucket, sub in temp.groupby("_bin", observed=True):
            rows.append(
                {
                    "feature": feature,
                    "bin": str(bucket),
                    "node_count": int(len(sub)),
                    "mean_value": float(sub[feature].mean()),
                    "mean_delta": float(sub["delta_dynamic_minus_original"].mean()),
                    "dynamic_better_rate": float((sub["group"] == "dynamic_better").mean()),
                    "dynamic_worse_rate": float((sub["group"] == "dynamic_worse").mean()),
                    "strong_better_rate": float((sub["strong_group"] == "strong_better").mean()),
                    "strong_worse_rate": float((sub["strong_group"] == "strong_worse").mean()),
                }
            )
    return pd.DataFrame(rows)


def _representative_cases(feature_table: pd.DataFrame, group_name: str, count: int, feature_cols: list[str]) -> pd.DataFrame:
    sub = feature_table[feature_table["group"] == group_name].copy()
    if sub.empty:
        return sub
    values = sub[feature_cols].astype(float)
    center = values.median()
    scale = values.std().replace(0, 1).fillna(1)
    sub["_representative_distance"] = (((values - center) / scale) ** 2).sum(axis=1)
    return sub.nsmallest(count, "_representative_distance")


def _case_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "node_index",
        "case_type",
        "delta_dynamic_minus_original",
        "original_adaptive_mae",
        "dynamic_threshold_mae",
        "ID",
        "Fwy",
        "Direction",
        "Lanes",
        "Lat",
        "Lng",
        "original_out_degree",
        "dynamic_out_degree_mean",
        "degree_delta_dynamic_minus_original",
        "dynamic_osrm_mean_m",
        "dynamic_added_osrm_mean_m",
        "dynamic_overlap_precision",
        "dynamic_same_fwy_share",
        "test_valid_mean",
        "test_p95",
        "test_zero_rate",
        "test_lag1_absdiff",
    ]
    rep_features = [
        "dynamic_out_degree_mean",
        "degree_delta_dynamic_minus_original",
        "dynamic_osrm_mean_m",
        "dynamic_overlap_precision",
        "test_valid_mean",
        "test_p95",
    ]
    parts = []
    for case_type, sub in [
        ("top_worse", feature_table.nlargest(15, "delta_dynamic_minus_original")),
        ("top_better", feature_table.nsmallest(15, "delta_dynamic_minus_original")),
        ("representative_worse", _representative_cases(feature_table, "dynamic_worse", 15, rep_features)),
        ("representative_better", _representative_cases(feature_table, "dynamic_better", 15, rep_features)),
    ]:
        tmp = sub.copy()
        tmp["case_type"] = case_type
        parts.append(tmp)
    cases = pd.concat(parts, ignore_index=True)
    return cases[[col for col in display_cols if col in cases.columns]]


def _plot_outputs(out_dir: Path, feature_table: pd.DataFrame, group_summary: pd.DataFrame, bin_table: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    top_effects = group_summary.reindex(group_summary["cohen_d_worse_vs_better"].abs().sort_values(ascending=False).index).head(16)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_effects["feature"], top_effects["cohen_d_worse_vs_better"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Cohen d: worse group - better group")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "feature_effect_sizes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        feature_table["dynamic_out_degree_mean"],
        feature_table["delta_dynamic_minus_original"],
        c=feature_table["test_valid_mean"],
        s=18,
        alpha=0.7,
        cmap="viridis",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Dynamic mean out-degree")
    ax.set_ylabel("Dynamic MAE - original adaptive MAE")
    fig.colorbar(sc, ax=ax, label="Test mean flow")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_vs_dynamic_degree.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        feature_table["dynamic_overlap_precision"],
        feature_table["delta_dynamic_minus_original"],
        c=feature_table["dynamic_out_degree_mean"],
        s=18,
        alpha=0.7,
        cmap="magma",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Dynamic edge overlap precision vs original graph")
    ax.set_ylabel("Dynamic MAE - original adaptive MAE")
    fig.colorbar(sc, ax=ax, label="Dynamic degree")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_vs_overlap_precision.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    clip_low, clip_high = np.percentile(feature_table["delta_dynamic_minus_original"], [2, 98])
    sc = ax.scatter(
        feature_table["Lng"],
        feature_table["Lat"],
        c=np.clip(feature_table["delta_dynamic_minus_original"], clip_low, clip_high),
        s=25,
        cmap="coolwarm",
        alpha=0.8,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax, label="Delta MAE clipped p2-p98")
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_delta_map.png", dpi=180)
    plt.close(fig)

    selected = [
        "dynamic_out_degree_mean",
        "degree_delta_dynamic_minus_original",
        "dynamic_osrm_mean_m",
        "dynamic_overlap_precision",
        "test_valid_mean",
        "test_p95",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feature in zip(axes.ravel(), selected):
        rows = bin_table[bin_table["feature"] == feature].copy()
        if rows.empty:
            ax.axis("off")
            continue
        ax.plot(range(len(rows)), rows["dynamic_better_rate"], marker="o", label="better")
        ax.plot(range(len(rows)), rows["dynamic_worse_rate"], marker="o", label="worse")
        ax.set_title(feature)
        ax.set_xlabel("feature quantile bin")
        ax.set_ylim(0, 1)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "better_worse_rate_by_feature_bins.png", dpi=180)
    plt.close(fig)


def main() -> None:
    repo_root = _repo_root()
    basic_ts = _basic_ts_dir(repo_root)
    result_dir = _default_result_dir(repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--node-analysis-dir", default=str(result_dir))
    parser.add_argument("--output-dir", default=str(repo_root / "mvp_experiments/active/adaptive_threshold_dynamic_weight/results/gwnet_dynamic_threshold_general_patterns_20260525"))
    parser.add_argument("--osrm-distance", default="/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/distance_matrices/SD/SD_osrm_shortest_distance_m.npy")
    parser.add_argument("--straight-distance", default="/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/distance_matrices/SD/SD_straight_distance_m.npy")
    parser.add_argument("--dynamic-cfg", default="baselines/GWNet/SD_dynamic_threshold.py")
    parser.add_argument("--dynamic-ckpt", default=str(basic_ts / "checkpoints/DynamicThresholdGraphWaveNet/SD_dynamic_threshold_hard_exp_tanh_gwnet_100_12_12_addaptadj_dynamic_threshold_hard_addaptadj_20260522_gpu1/94d620cfd224e6d2cf7cd2c344b1a3cd/DynamicThresholdGraphWaveNet_best_val_MAE.pt"))
    parser.add_argument("--dynamic-run-tag", default="dynamic_threshold_hard_addaptadj_20260522_gpu1")
    parser.add_argument("--dynamic-addaptadj", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_df = pd.read_csv(Path(args.node_analysis_dir) / "per_node_mae_degree.csv")
    meta = pd.read_csv(basic_ts / "datasets/SD/meta.csv")
    meta["FwyBase"] = meta["Fwy"].astype(str).str.replace(r"-[NSEW]$", "", regex=True)
    desc = json.loads((basic_ts / "datasets/SD/desc.json").read_text(encoding="utf-8"))
    data = np.memmap(basic_ts / "datasets/SD/data.dat", dtype="float32", mode="r", shape=tuple(desc["shape"]))
    flow = np.asarray(data[..., 0])
    total_len = flow.shape[0]
    ratio = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    valid_len = int(total_len * ratio[1])
    test_len = int(total_len * ratio[2])
    train_len = total_len - valid_len - test_len
    input_len = desc["regular_settings"]["INPUT_LEN"]
    test_flow = flow[train_len + valid_len + input_len :]

    original_adj = _load_original_adj(basic_ts)
    osrm_m = np.load(args.osrm_distance).astype("float32")[: len(node_df), : len(node_df)]
    straight_m = np.load(args.straight_distance).astype("float32")[: len(node_df), : len(node_df)]
    np.fill_diagonal(osrm_m, 0.0)
    np.fill_diagonal(straight_m, 0.0)

    feature_table = pd.concat(
        [
            node_df.reset_index(drop=True),
            meta.reset_index(drop=True),
            _node_flow_stats(test_flow, "test"),
            _node_flow_stats(flow, "full"),
            _static_neighbor_features(original_adj, osrm_m, straight_m, meta),
            _dynamic_graph_features(args, basic_ts, original_adj, osrm_m, straight_m, meta),
        ],
        axis=1,
    )
    feature_table["group"] = np.where(
        feature_table["delta_dynamic_minus_original"] < 0,
        "dynamic_better",
        np.where(feature_table["delta_dynamic_minus_original"] > 0, "dynamic_worse", "tie"),
    )
    low_q, high_q = feature_table["delta_dynamic_minus_original"].quantile([0.1, 0.9])
    feature_table["strong_group"] = np.where(
        feature_table["delta_dynamic_minus_original"] <= low_q,
        "strong_better",
        np.where(feature_table["delta_dynamic_minus_original"] >= high_q, "strong_worse", "middle"),
    )

    feature_cols = [
        "original_out_degree",
        "original_in_degree",
        "dynamic_out_degree_mean",
        "dynamic_out_degree_std",
        "dynamic_in_degree_mean",
        "degree_delta_dynamic_minus_original",
        "dynamic_osrm_mean_m",
        "dynamic_straight_mean_m",
        "dynamic_osrm_max_mean_m",
        "dynamic_added_osrm_mean_m",
        "dynamic_added_degree",
        "dynamic_overlap_precision",
        "dynamic_overlap_recall",
        "dynamic_same_fwy_share",
        "dynamic_same_dir_share",
        "original_osrm_mean_m",
        "original_straight_mean_m",
        "original_same_fwy_share",
        "original_same_dir_share",
        "Lanes",
        "test_valid_mean",
        "test_valid_std",
        "test_cv",
        "test_zero_rate",
        "test_p50",
        "test_p95",
        "test_lag1_absdiff",
        "test_daily_amp",
        "full_valid_mean",
        "full_cv",
        "full_zero_rate",
    ]

    group_summary = _group_summary(feature_table, feature_cols)
    correlations = _correlation_table(feature_table, feature_cols)
    bin_table = _bin_prevalence(
        feature_table,
        [
            "dynamic_out_degree_mean",
            "degree_delta_dynamic_minus_original",
            "dynamic_osrm_mean_m",
            "dynamic_overlap_precision",
            "dynamic_same_fwy_share",
            "test_valid_mean",
            "test_p95",
            "test_zero_rate",
            "Lanes",
        ],
    )
    cases = _case_table(feature_table)

    categorical_rows = []
    for column in ["FwyBase", "Direction", "Lanes"]:
        ct = pd.crosstab(feature_table[column], feature_table["group"])
        for value, row in ct.iterrows():
            total = row.sum()
            categorical_rows.append(
                {
                    "feature": column,
                    "value": value,
                    "node_count": int(total),
                    "dynamic_better_rate": float(row.get("dynamic_better", 0) / total),
                    "dynamic_worse_rate": float(row.get("dynamic_worse", 0) / total),
                    "tie_rate": float(row.get("tie", 0) / total),
                }
            )
    categorical = pd.DataFrame(categorical_rows)

    feature_table.to_csv(out_dir / "node_feature_table.csv", index=False)
    group_summary.to_csv(out_dir / "feature_group_summary.csv", index=False)
    correlations.to_csv(out_dir / "feature_correlations.csv", index=False)
    bin_table.to_csv(out_dir / "feature_bin_prevalence.csv", index=False)
    categorical.to_csv(out_dir / "categorical_prevalence.csv", index=False)
    cases.to_csv(out_dir / "case_study_nodes.csv", index=False)

    summary = {
        "node_count": int(len(feature_table)),
        "dynamic_better_count": int((feature_table["group"] == "dynamic_better").sum()),
        "dynamic_worse_count": int((feature_table["group"] == "dynamic_worse").sum()),
        "tie_count": int((feature_table["group"] == "tie").sum()),
        "strong_better_delta_threshold": float(low_q),
        "strong_worse_delta_threshold": float(high_q),
        "delta_summary": _summary(feature_table["delta_dynamic_minus_original"]),
        "dynamic_degree_summary": _summary(feature_table["dynamic_out_degree_mean"]),
        "dynamic_osrm_mean_summary_m": _summary(feature_table["dynamic_osrm_mean_m"]),
        "dynamic_overlap_precision_summary": _summary(feature_table["dynamic_overlap_precision"]),
        "top_effect_size_features": group_summary.head(12).to_dict(orient="records"),
        "top_correlation_features": correlations.head(12).to_dict(orient="records"),
    }
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _plot_outputs(out_dir, feature_table, group_summary, bin_table)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()

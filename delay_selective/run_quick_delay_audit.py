#!/usr/bin/env python3
"""Quick delay observability audit for road-segment traffic speed data.

This script is intentionally lightweight: it samples graph edges, computes
directed lagged correlations, and writes parseable CSV/JSON summaries for a
server-side sanity check before any model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick delay observability audit on graph traffic speed data."
    )
    parser.add_argument("--dataset-npz", required=True, help="Dataset NPZ containing edges and metadata.")
    parser.add_argument("--targets-npy", default=None, help="Optional targets .npy loaded with mmap.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON/plot outputs.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-time-steps", type=int, default=10000)
    parser.add_argument("--max-edges", type=int, default=1500)
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum positive lag in timesteps.")
    parser.add_argument("--min-pair-coverage", type=float, default=0.80)
    parser.add_argument("--residualize", default="time_of_day", choices=["none", "mean", "time_of_day"])
    parser.add_argument("--min-corr", type=float, default=0.20)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distance-bins-m",
        default="0,50,100,200,500,1000,2000,inf",
        help="Comma-separated centroid-distance bins in meters.",
    )
    return parser.parse_args()


def as_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def load_targets(dataset: Any, targets_npy: str | None) -> np.ndarray:
    if targets_npy:
        return np.load(targets_npy, mmap_mode="r")
    return dataset["targets"]


def select_timestamps(dataset: Any, split: str, max_time_steps: int) -> np.ndarray:
    if split == "all":
        total = dataset["unix_timestamps"].shape[0]
        idx = np.arange(total, dtype=np.int64)
    else:
        idx = np.asarray(dataset[f"{split}_timestamps"], dtype=np.int64)
    if max_time_steps and max_time_steps > 0:
        idx = idx[: min(max_time_steps, idx.shape[0])]
    return idx


def get_feature_index(names: np.ndarray, name: str) -> int | None:
    decoded = [str(x) for x in names.tolist()]
    return decoded.index(name) if name in decoded else None


def node_centroids(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    features = np.asarray(dataset["spatial_node_features"])[0]
    names = np.asarray(dataset["spatial_node_feature_names"])

    x0_i = get_feature_index(names, "x_coordinate_start")
    y0_i = get_feature_index(names, "y_coordinate_start")
    x1_i = get_feature_index(names, "x_coordinate_end")
    y1_i = get_feature_index(names, "y_coordinate_end")
    length_i = get_feature_index(names, "length")

    if None in (x0_i, y0_i, x1_i, y1_i):
        raise ValueError("Coordinate features are missing from spatial_node_features.")

    cx = (features[:, x0_i] + features[:, x1_i]) / 2.0
    cy = (features[:, y0_i] + features[:, y1_i]) / 2.0
    length = features[:, length_i] if length_i is not None else None
    return cx, cy, length


def parse_bins(spec: str) -> np.ndarray:
    values: list[float] = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        if item in {"inf", "infinity"}:
            values.append(float("inf"))
        else:
            values.append(float(item))
    if len(values) < 2:
        raise ValueError("At least two distance-bin boundaries are required.")
    return np.asarray(values, dtype=float)


def residualize_matrix(x: np.ndarray, unix_timestamps: np.ndarray, mode: str) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if mode == "none":
        return x
    if mode == "mean":
        return x - np.nanmean(x, axis=0, keepdims=True)
    if mode != "time_of_day":
        raise ValueError(f"Unknown residualize mode: {mode}")

    if unix_timestamps.shape[0] < 2:
        return x - np.nanmean(x, axis=0, keepdims=True)

    step = int(np.median(np.diff(unix_timestamps)))
    step = max(step, 1)
    slots_per_day = max(1, int(round(86400 / step)))
    slots = ((unix_timestamps % 86400) // step).astype(np.int64)
    slots = np.clip(slots, 0, slots_per_day - 1)

    baseline = np.full((slots_per_day, x.shape[1]), np.nan, dtype=np.float32)
    global_mean = np.nanmean(x, axis=0)
    for slot in np.unique(slots):
        mask = slots == slot
        baseline[slot] = np.nanmean(x[mask], axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, global_mean[None, :])
    return x - baseline[slots]


def corr_columns(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(a) & np.isfinite(b)
    count = valid.sum(axis=0).astype(np.float64)

    a0 = np.where(valid, a, 0.0).astype(np.float64, copy=False)
    b0 = np.where(valid, b, 0.0).astype(np.float64, copy=False)
    safe_count = np.maximum(count, 1.0)

    mean_a = a0.sum(axis=0) / safe_count
    mean_b = b0.sum(axis=0) / safe_count

    da = np.where(valid, a - mean_a[None, :], 0.0)
    db = np.where(valid, b - mean_b[None, :], 0.0)
    cov = (da * db).sum(axis=0)
    var_a = (da * da).sum(axis=0)
    var_b = (db * db).sum(axis=0)
    denom = np.sqrt(var_a * var_b)

    corr = np.full(a.shape[1], np.nan, dtype=np.float64)
    ok = (count >= 3) & (denom > 0)
    corr[ok] = cov[ok] / denom[ok]
    return corr, count


def sample_edges(
    edges: np.ndarray,
    x: np.ndarray,
    max_edges: int,
    min_pair_coverage: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    src_cov = np.isfinite(x[:, src]).mean(axis=0)
    dst_cov = np.isfinite(x[:, dst]).mean(axis=0)
    pair_cov = np.minimum(src_cov, dst_cov)
    eligible = np.flatnonzero(pair_cov >= min_pair_coverage)
    if eligible.shape[0] == 0:
        raise ValueError("No edges satisfy the minimum pair coverage threshold.")

    if max_edges and eligible.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(eligible, size=max_edges, replace=False))
    else:
        chosen = eligible
    return chosen, pair_cov[chosen]


def compute_lag_scores(
    x: np.ndarray,
    edges: np.ndarray,
    edge_indices: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    src = edges[edge_indices, 0].astype(np.int64)
    dst = edges[edge_indices, 1].astype(np.int64)
    source = x[:, src]
    target = x[:, dst]

    corr_by_lag = np.full((max_lag + 1, edge_indices.shape[0]), np.nan, dtype=np.float64)
    count_by_lag = np.zeros((max_lag + 1, edge_indices.shape[0]), dtype=np.float64)

    for lag in range(max_lag + 1):
        if lag == 0:
            a = source
            b = target
        else:
            a = source[:-lag]
            b = target[lag:]
        corr, count = corr_columns(a, b)
        corr_by_lag[lag] = corr
        count_by_lag[lag] = count
    return corr_by_lag, count_by_lag


def summarize_bins(
    distances: np.ndarray,
    best_lags: np.ndarray,
    improvements: np.ndarray,
    best_corrs: np.ndarray,
    bins: np.ndarray,
    min_corr: float,
    min_improvement: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    high_conf_nonzero = (best_lags > 0) & (best_corrs >= min_corr) & (improvements >= min_improvement)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if math.isinf(hi):
            mask = distances >= lo
            label = f"[{lo:g}, inf)"
        else:
            mask = (distances >= lo) & (distances < hi)
            label = f"[{lo:g}, {hi:g})"
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "distance_bin_m": label,
                    "num_edges": 0,
                    "near_zero_ratio": None,
                    "high_conf_nonzero_ratio": None,
                    "median_best_lag": None,
                    "median_improvement": None,
                }
            )
            continue
        rows.append(
            {
                "distance_bin_m": label,
                "num_edges": n,
                "near_zero_ratio": float(np.mean(best_lags[mask] == 0)),
                "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero[mask])),
                "median_best_lag": float(np.nanmedian(best_lags[mask])),
                "median_improvement": float(np.nanmedian(improvements[mask])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(output_dir: Path, corr_by_lag: np.ndarray, best_lags: np.ndarray, distances: np.ndarray) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    made: list[str] = []

    plt.figure(figsize=(7, 4))
    plt.hist(best_lags, bins=np.arange(best_lags.max() + 2) - 0.5, edgecolor="black")
    plt.xlabel("Best lag (timesteps)")
    plt.ylabel("Edge count")
    plt.title("Best lag distribution")
    out = output_dir / "best_lag_hist.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    made.append(str(out))

    plt.figure(figsize=(7, 4))
    plt.scatter(distances, best_lags, s=8, alpha=0.35)
    plt.xlabel("Edge centroid distance (m)")
    plt.ylabel("Best lag (timesteps)")
    plt.title("Best lag vs edge distance")
    out = output_dir / "best_lag_vs_distance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    made.append(str(out))

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(corr_by_lag.shape[0]), np.nanmean(corr_by_lag, axis=1), marker="o")
    plt.xlabel("Lag (timesteps)")
    plt.ylabel("Mean edge correlation")
    plt.title("Mean lagged correlation")
    out = output_dir / "mean_corr_by_lag.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    made.append(str(out))

    return made


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(args.dataset_npz, allow_pickle=True)
    targets = load_targets(dataset, args.targets_npy)
    timestamp_idx = select_timestamps(dataset, args.split, args.max_time_steps)
    unix_timestamps = np.asarray(dataset["unix_timestamps"])[timestamp_idx]
    edges = np.asarray(dataset["edges"], dtype=np.int64)

    x = np.asarray(targets[timestamp_idx, :], dtype=np.float32)
    x = residualize_matrix(x, unix_timestamps, args.residualize)

    edge_indices, pair_coverage = sample_edges(
        edges=edges,
        x=x,
        max_edges=args.max_edges,
        min_pair_coverage=args.min_pair_coverage,
        seed=args.seed,
    )
    corr_by_lag, count_by_lag = compute_lag_scores(
        x=x,
        edges=edges,
        edge_indices=edge_indices,
        max_lag=args.max_lag,
    )

    has_any_corr = np.isfinite(corr_by_lag).any(axis=0)
    corr_for_argmax = np.where(np.isfinite(corr_by_lag), corr_by_lag, -np.inf)
    best_lags = np.argmax(corr_for_argmax, axis=0)
    best_lags = np.where(has_any_corr, best_lags, 0)
    best_corrs = corr_by_lag[best_lags, np.arange(edge_indices.shape[0])]
    zero_corrs = corr_by_lag[0]
    improvements = best_corrs - zero_corrs

    cx, cy, lengths = node_centroids(dataset)
    sampled_edges = edges[edge_indices]
    src = sampled_edges[:, 0]
    dst = sampled_edges[:, 1]
    centroid_distances = np.sqrt((cx[src] - cx[dst]) ** 2 + (cy[src] - cy[dst]) ** 2)

    timestep_seconds = int(np.median(np.diff(np.asarray(dataset["unix_timestamps"]))))
    high_conf_nonzero = (
        (best_lags > 0)
        & (best_corrs >= args.min_corr)
        & (improvements >= args.min_improvement)
    )

    edge_rows: list[dict[str, Any]] = []
    for local_i, edge_i in enumerate(edge_indices):
        row = {
            "edge_index": int(edge_i),
            "source": int(src[local_i]),
            "target": int(dst[local_i]),
            "pair_coverage": float(pair_coverage[local_i]),
            "centroid_distance_m": float(centroid_distances[local_i]),
            "source_length_m": float(lengths[src[local_i]]) if lengths is not None else None,
            "target_length_m": float(lengths[dst[local_i]]) if lengths is not None else None,
            "best_lag_steps": int(best_lags[local_i]),
            "best_lag_seconds": int(best_lags[local_i] * timestep_seconds),
            "zero_lag_corr": float(zero_corrs[local_i]),
            "best_lag_corr": float(best_corrs[local_i]),
            "corr_improvement": float(improvements[local_i]),
            "high_conf_nonzero_delay": bool(high_conf_nonzero[local_i]),
            "valid_pairs_at_best_lag": int(count_by_lag[best_lags[local_i], local_i]),
        }
        for lag in range(args.max_lag + 1):
            row[f"corr_lag_{lag}"] = float(corr_by_lag[lag, local_i])
        edge_rows.append(row)

    bins = parse_bins(args.distance_bins_m)
    bin_rows = summarize_bins(
        distances=centroid_distances,
        best_lags=best_lags,
        improvements=improvements,
        best_corrs=best_corrs,
        bins=bins,
        min_corr=args.min_corr,
        min_improvement=args.min_improvement,
    )

    write_csv(output_dir / "edge_delay_scores.csv", edge_rows)
    write_csv(output_dir / "distance_bin_summary.csv", bin_rows)
    plots = maybe_plot(output_dir, corr_by_lag, best_lags, centroid_distances)

    summary = {
        "dataset_npz": args.dataset_npz,
        "targets_npy": args.targets_npy,
        "split": args.split,
        "num_timestamps_used": int(timestamp_idx.shape[0]),
        "num_edges_total": int(edges.shape[0]),
        "num_edges_sampled": int(edge_indices.shape[0]),
        "timestep_seconds": int(timestep_seconds),
        "max_lag_steps": int(args.max_lag),
        "max_lag_minutes": float(args.max_lag * timestep_seconds / 60.0),
        "residualize": args.residualize,
        "min_pair_coverage": float(args.min_pair_coverage),
        "min_corr": float(args.min_corr),
        "min_improvement": float(args.min_improvement),
        "near_zero_ratio": float(np.mean(best_lags == 0)),
        "nonzero_ratio": float(np.mean(best_lags > 0)),
        "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero)),
        "median_best_lag_steps": float(np.nanmedian(best_lags)),
        "median_corr_improvement": float(np.nanmedian(improvements)),
        "mean_corr_by_lag": [float(x) for x in np.nanmean(corr_by_lag, axis=1)],
        "distance_bin_summary": bin_rows,
        "plots": plots,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

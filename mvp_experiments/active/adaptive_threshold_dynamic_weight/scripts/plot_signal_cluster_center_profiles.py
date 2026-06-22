#!/usr/bin/env python3
"""Plot raw signal-center profiles for KMeans cluster assignments.

The KMeans centers stored by sklearn live in standardized embedding space. This
script maps cluster assignments back to the original traffic-flow windows and
computes raw node-average profiles per cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["SD", "GLA", "GBA"])
    parser.add_argument("--embeddings", nargs="+", default=["time", "freq"])
    parser.add_argument("--basicts-datasets-dir", type=Path, default=Path("BasicTS/datasets"))
    parser.add_argument(
        "--cluster-output-dir",
        type=Path,
        default=Path("mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/signal_kmeans_clusters_12step_week"),
    )
    parser.add_argument("--feature-index", type=int, default=0)
    parser.add_argument("--window-steps", type=int, default=12)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--selected-windows", nargs="+", type=int, default=[0, 2, 3, 5, 6, 7])
    parser.add_argument("--period-name", default="analysis period")
    parser.add_argument("--per-node-zscore", action="store_true", default=True)
    parser.add_argument("--raw-scale", action="store_true", help="Plot raw flow instead of node-wise z-scored shape.")
    return parser.parse_args()


def load_desc(dataset_dir: Path) -> dict:
    with (dataset_dir / "desc.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_window(dataset_dir: Path, desc: dict, start: int, length: int, feature_index: int) -> np.ndarray:
    shape = tuple(desc["shape"])
    mmap = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    return np.asarray(mmap[start : start + length, :, feature_index], dtype=np.float64)


def node_zscore(signal_tn: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.nanmean(signal_tn, axis=0, keepdims=True)
    std = np.nanstd(signal_tn, axis=0, keepdims=True)
    return (signal_tn - mean) / np.maximum(std, eps)


def time_label(window_index: int, window_steps: int, frequency: int) -> str:
    steps_per_day = int(round(24 * 60 / frequency))
    start = window_index * window_steps
    end = start + window_steps
    day = int(start // steps_per_day) + 1
    h0 = (start % steps_per_day) * frequency / 60.0
    h1 = (end % steps_per_day) * frequency / 60.0
    if h1 == 0.0 and window_steps > 0:
        h1 = 24.0
    return f"w{window_index:02d} D{day} {h0:04.1f}-{h1:04.1f}h"


def window_label(window_steps: int) -> str:
    return f"{window_steps}-step window"


def compute_centers(
    dataset_dir: Path,
    desc: dict,
    assignments: pd.DataFrame,
    feature_index: int,
    window_steps: int,
    num_clusters: int,
    raw_scale: bool,
) -> pd.DataFrame:
    rows = []
    grouped = assignments.groupby("window_index", sort=True)
    for window_index, group in grouped:
        start = int(group["start_step"].iloc[0])
        signal = load_window(dataset_dir, desc, start, window_steps, feature_index)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if not raw_scale:
            signal = node_zscore(signal)
        labels = group.sort_values("node_index")["cluster"].to_numpy(dtype=int)
        if labels.shape[0] != signal.shape[1]:
            raise ValueError(f"window={window_index}: labels={labels.shape[0]} nodes, signal={signal.shape[1]}")
        for cluster in range(num_clusters):
            mask = labels == cluster
            if not np.any(mask):
                center = np.full(window_steps, np.nan)
                spread = np.full(window_steps, np.nan)
                count = 0
            else:
                vals = signal[:, mask]
                center = np.nanmean(vals, axis=1)
                spread = np.nanstd(vals, axis=1)
                count = int(mask.sum())
            for step in range(window_steps):
                rows.append(
                    {
                        "window_index": int(window_index),
                        "cluster": int(cluster),
                        "step": int(step),
                        "center": float(center[step]),
                        "std": float(spread[step]),
                        "count": count,
                    }
                )
    return pd.DataFrame(rows)


def plot_selected_windows(
    centers: pd.DataFrame,
    dataset: str,
    embedding: str,
    frequency: int,
    window_steps: int,
    selected_windows: list[int],
    num_clusters: int,
    raw_scale: bool,
    period_name: str,
    output: Path,
) -> None:
    selected = [idx for idx in selected_windows if idx in set(centers["window_index"])]
    cols = min(3, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.7 * rows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("tab10", num_clusters)
    x = np.arange(window_steps)
    for ax_idx, ax in enumerate(axes.ravel()):
        if ax_idx >= len(selected):
            ax.axis("off")
            continue
        window_index = selected[ax_idx]
        sub = centers[centers["window_index"] == window_index]
        for cluster in range(num_clusters):
            line = sub[sub["cluster"] == cluster].sort_values("step")
            if line.empty:
                continue
            ax.plot(x, line["center"], color=cmap(cluster), linewidth=1.7, label=f"C{cluster}")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(time_label(window_index, window_steps, frequency), fontsize=10, pad=7)
        ax.set_xlabel(f"step in {window_label(window_steps)}")
        ax.set_ylabel("flow" if raw_scale else "node-wise z-score")
        ax.grid(True, alpha=0.22)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False)
    fig.suptitle(f"{dataset} {embedding}: cluster center profiles in selected {window_label(window_steps)}s", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_period_average(
    centers: pd.DataFrame,
    dataset: str,
    embedding: str,
    window_steps: int,
    num_clusters: int,
    raw_scale: bool,
    period_name: str,
    output: Path,
) -> None:
    agg = (
        centers.groupby(["cluster", "step"], as_index=False)
        .agg(center_mean=("center", "mean"), center_std=("center", "std"), count_mean=("count", "mean"))
        .sort_values(["cluster", "step"])
    )
    cols = 4
    rows = int(np.ceil(num_clusters / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.0 * cols, 3.0 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("tab10", num_clusters)
    x = np.arange(window_steps)
    for cluster, ax in enumerate(axes.ravel()[:num_clusters]):
        line = agg[agg["cluster"] == cluster]
        mean = line["center_mean"].to_numpy()
        std = np.nan_to_num(line["center_std"].to_numpy(), nan=0.0)
        ax.plot(x, mean, color=cmap(cluster), linewidth=2.0)
        ax.fill_between(x, mean - std, mean + std, color=cmap(cluster), alpha=0.18, linewidth=0)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(f"C{cluster}, avg nodes={line['count_mean'].mean():.1f}", fontsize=10)
        ax.grid(True, alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel(f"step in {window_label(window_steps)}")
    for ax in axes[:, 0]:
        ax.set_ylabel("flow" if raw_scale else "node-wise z-score")
    fig.suptitle(f"{dataset} {embedding}: {period_name}-average raw signal shape by cluster", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mode = "raw" if args.raw_scale else "zshape"
    all_summary_rows = []
    for dataset in args.datasets:
        dataset_dir = args.basicts_datasets_dir / dataset
        desc = load_desc(dataset_dir)
        frequency = int(desc.get("frequency (minutes)", 15))
        for embedding in args.embeddings:
            base = args.cluster_output_dir / dataset / f"{dataset}_{embedding}_k{args.num_clusters}"
            assignment_path = base.with_name(base.name + "_assignments.csv")
            if not assignment_path.exists():
                raise FileNotFoundError(assignment_path)
            assignments = pd.read_csv(assignment_path)
            centers = compute_centers(
                dataset_dir=dataset_dir,
                desc=desc,
                assignments=assignments,
                feature_index=args.feature_index,
                window_steps=args.window_steps,
                num_clusters=args.num_clusters,
                raw_scale=args.raw_scale,
            )
            centers_path = base.with_name(base.name + f"_center_profiles_{mode}.csv")
            centers.to_csv(centers_path, index=False)
            selected_path = base.with_name(base.name + f"_center_profiles_selected_{mode}.png")
            period_path = base.with_name(base.name + f"_center_profiles_periodavg_{mode}.png")
            plot_selected_windows(
                centers=centers,
                dataset=dataset,
                embedding=embedding,
                frequency=frequency,
                window_steps=args.window_steps,
                selected_windows=args.selected_windows,
                num_clusters=args.num_clusters,
                raw_scale=args.raw_scale,
                period_name=args.period_name,
                output=selected_path,
            )
            plot_period_average(
                centers=centers,
                dataset=dataset,
                embedding=embedding,
                window_steps=args.window_steps,
                num_clusters=args.num_clusters,
                raw_scale=args.raw_scale,
                period_name=args.period_name,
                output=period_path,
            )
            all_summary_rows.append(
                {
                    "dataset": dataset,
                    "embedding": embedding,
                    "mode": mode,
                    "centers_csv": str(centers_path),
                    "selected_plot": str(selected_path),
                    "periodavg_plot": str(period_path),
                }
            )
            print(f"wrote {selected_path}")
            print(f"wrote {period_path}")
    summary = pd.DataFrame(all_summary_rows)
    summary_path = args.cluster_output_dir / f"center_profile_summary_{mode}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

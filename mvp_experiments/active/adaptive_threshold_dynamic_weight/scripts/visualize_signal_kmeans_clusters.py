#!/usr/bin/env python3
"""Visualize time-varying signal clusters for LargeST-style datasets.

This script is diagnostic only. It reads BasicTS data.dat / desc.json / meta.csv
files, clusters node signals over sliding windows, and writes maps/statistics.
It does not build graphs or change any training configuration.

Default setting follows the dynamic-threshold discussion: use a fixed-length
history segment as a node embedding and visualize how clusters evolve across a
chosen period.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_dir: Path
    desc_path: Path
    data_path: Path
    meta_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SD", "GLA", "GBA"],
        help="Dataset names under --basicts-datasets-dir.",
    )
    parser.add_argument(
        "--basicts-datasets-dir",
        type=Path,
        default=Path("BasicTS/datasets"),
        help="Directory containing BasicTS datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/signal_kmeans_clusters"))
    parser.add_argument("--embeddings", nargs="+", choices=["time", "freq"], default=["time", "freq"])
    parser.add_argument("--feature-index", type=int, default=0, help="Traffic feature channel to cluster.")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument(
        "--week-steps",
        type=int,
        default=None,
        help="Total analysis span in steps. Historical name; can represent a week, month, etc. Defaults to 7 days.",
    )
    parser.add_argument("--window-steps", type=int, default=12, help="Default: 12-step history segment.")
    parser.add_argument("--stride-steps", type=int, default=None, help="Defaults to --window-steps.")
    parser.add_argument("--max-windows", type=int, default=None, help="Defaults to all windows in the analysis span.")
    parser.add_argument("--period-name", default="analysis period", help="Text used in plot titles.")
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--freq-components", type=int, default=6, help="Number of low-frequency FFT magnitudes.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument(
        "--no-per-node-zscore",
        action="store_true",
        help="Disable node-wise z-score before embedding. Default clusters by shape.",
    )
    parser.add_argument(
        "--downsample-time",
        type=int,
        default=1,
        help="Average consecutive time steps before time-domain clustering.",
    )
    return parser.parse_args()


def load_dataset_spec(root: Path, name: str) -> DatasetSpec:
    dataset_dir = root / name
    spec = DatasetSpec(
        name=name,
        dataset_dir=dataset_dir,
        desc_path=dataset_dir / "desc.json",
        data_path=dataset_dir / "data.dat",
        meta_path=dataset_dir / "meta.csv",
    )
    missing = [path for path in (spec.desc_path, spec.data_path, spec.meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{name}: missing {missing}")
    return spec


def load_desc(spec: DatasetSpec) -> dict:
    with spec.desc_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_signal_window(spec: DatasetSpec, desc: dict, start: int, length: int, feature_index: int) -> np.ndarray:
    shape = tuple(desc["shape"])
    mmap = np.memmap(spec.data_path, dtype="float32", mode="r", shape=shape)
    end = min(start + length, shape[0])
    if start < 0 or start >= shape[0] or end <= start:
        raise ValueError(f"Bad window start={start}, length={length}, data length={shape[0]}")
    if feature_index < 0 or feature_index >= shape[2]:
        raise ValueError(f"feature_index={feature_index} outside feature dim={shape[2]}")
    return np.asarray(mmap[start:end, :, feature_index], dtype=np.float32)


def node_zscore(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std = np.nanstd(signal, axis=0, keepdims=True)
    return (signal - mean) / np.maximum(std, eps)


def make_time_embedding(signal_tn: np.ndarray, downsample: int) -> np.ndarray:
    """Return node x time embedding."""
    signal = signal_tn
    if downsample > 1:
        usable = (signal.shape[0] // downsample) * downsample
        signal = signal[:usable].reshape(usable // downsample, downsample, signal.shape[1]).mean(axis=1)
    return signal.T


def make_freq_embedding(signal_tn: np.ndarray, num_components: int) -> np.ndarray:
    """Return node x frequency-magnitude embedding."""
    centered = signal_tn - np.nanmean(signal_tn, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    magnitudes = np.abs(spectrum)[1 : num_components + 1]
    return magnitudes.T


def clean_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float64)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(embedding)


def fit_kmeans(embedding: np.ndarray, num_clusters: int, seed: int, n_init: int, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    labels = model.fit_predict(embedding)
    return labels.astype(int), model.cluster_centers_


def align_labels(
    labels: np.ndarray,
    centers: np.ndarray,
    previous_centers: np.ndarray | None,
    num_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    if previous_centers is None:
        return labels, centers
    cost = np.linalg.norm(centers[:, None, :] - previous_centers[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(row): int(col) for row, col in zip(rows, cols)}
    aligned = np.array([mapping.get(int(label), int(label)) for label in labels], dtype=int)
    aligned_centers = np.zeros_like(previous_centers)
    for old, new in mapping.items():
        aligned_centers[new] = centers[old]
    unused = set(range(num_clusters)) - set(mapping.values())
    for cluster_id in unused:
        aligned_centers[cluster_id] = previous_centers[cluster_id]
    return aligned, aligned_centers


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")


def load_meta(spec: DatasetSpec) -> pd.DataFrame:
    meta = pd.read_csv(spec.meta_path)
    lat_col = find_column(meta, ("Lat", "lat", "latitude"))
    lon_col = find_column(meta, ("Lng", "lng", "lon", "longitude"))
    id_col = find_column(meta, ("ID", "id", "sensor_id", "node_id"))
    out = meta.copy()
    out["_node_index"] = np.arange(len(out))
    out["_id"] = out[id_col].astype(str)
    out["_lat"] = out[lat_col].astype(float)
    out["_lon"] = out[lon_col].astype(float)
    return out


def plot_cluster_maps(
    meta: pd.DataFrame,
    labels_by_window: list[np.ndarray],
    titles: list[str],
    dataset: str,
    embedding_name: str,
    output: Path,
    num_clusters: int,
) -> None:
    n = len(labels_by_window)
    cols = min(8 if n > 16 else 4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    cmap = plt.get_cmap("tab20", num_clusters)
    lon = meta["_lon"].to_numpy()
    lat = meta["_lat"].to_numpy()
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n:
            ax.axis("off")
            continue
        labels = labels_by_window[idx]
        sc = ax.scatter(lon, lat, c=labels, s=8, cmap=cmap, vmin=-0.5, vmax=num_clusters - 0.5, alpha=0.88, linewidths=0)
        ax.set_title(titles[idx], fontsize=10)
        ax.set_xlabel("Lng")
        ax.set_ylabel("Lat")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.5)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.82, ticks=np.arange(num_clusters))
    cbar.set_label("Aligned KMeans cluster")
    fig.suptitle(f"{dataset} {embedding_name} KMeans clusters", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _window_axis_label(window_steps: int) -> str:
    return f"{window_steps}-step window index"


def plot_cluster_animation(
    meta: pd.DataFrame,
    labels_by_window: list[np.ndarray],
    titles: list[str],
    dataset: str,
    embedding_name: str,
    output: Path,
    num_clusters: int,
) -> bool:
    try:
        from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    cmap = plt.get_cmap("tab20", num_clusters)
    lon = meta["_lon"].to_numpy()
    lat = meta["_lat"].to_numpy()
    tmp_dir = output.with_suffix("")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    x_pad = max((lon.max() - lon.min()) * 0.04, 1e-6)
    y_pad = max((lat.max() - lat.min()) * 0.04, 1e-6)
    for idx, labels in enumerate(labels_by_window):
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        sc = ax.scatter(lon, lat, c=labels, s=8, cmap=cmap, vmin=-0.5, vmax=num_clusters - 0.5, alpha=0.88, linewidths=0)
        ax.set_xlim(lon.min() - x_pad, lon.max() + x_pad)
        ax.set_ylim(lat.min() - y_pad, lat.max() + y_pad)
        ax.set_title(f"{dataset} {embedding_name}: {titles[idx]}", fontsize=11)
        ax.set_xlabel("Lng")
        ax.set_ylabel("Lat")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.5)
        fig.colorbar(sc, ax=ax, shrink=0.82, ticks=np.arange(num_clusters), label="Cluster")
        frame_path = tmp_dir / f"frame_{idx:03d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    adaptive_palette = getattr(Image, "Palette", Image).ADAPTIVE
    images = [Image.open(path).convert("P", palette=adaptive_palette) for path in frame_paths]
    images[0].save(output, save_all=True, append_images=images[1:], duration=320, loop=0, optimize=False)
    return True


def plot_cluster_fraction(count_df: pd.DataFrame, dataset: str, embedding_name: str, output: Path) -> None:
    pivot = count_df.pivot(index="cluster", columns="window_index", values="fraction").sort_index()
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * pivot.shape[1]), max(4, 0.42 * pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(np.arange(pivot.shape[1]), labels=[str(col) for col in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]), labels=[str(idx) for idx in pivot.index])
    ax.set_xlabel("Window index")
    ax.set_ylabel("Cluster")
    ax.set_title(f"{dataset} {embedding_name} cluster fraction")
    fig.colorbar(im, ax=ax, label="Node fraction")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_node_timeline(labels_by_window: list[np.ndarray], dataset: str, embedding_name: str, output: Path, num_clusters: int) -> None:
    label_matrix = np.stack(labels_by_window, axis=1)
    order = np.lexsort(tuple(label_matrix[:, col] for col in reversed(range(label_matrix.shape[1]))))
    sorted_matrix = label_matrix[order]
    fig, ax = plt.subplots(figsize=(max(5, 0.75 * label_matrix.shape[1]), 7))
    im = ax.imshow(sorted_matrix, aspect="auto", interpolation="nearest", cmap=plt.get_cmap("tab20", num_clusters), vmin=-0.5, vmax=num_clusters - 0.5)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Nodes sorted by cluster sequence")
    ax.set_title(f"{dataset} {embedding_name} node cluster timeline")
    ax.set_xticks(np.arange(label_matrix.shape[1]))
    fig.colorbar(im, ax=ax, label="Cluster")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ari_curve(
    stability_df: pd.DataFrame,
    dataset: str,
    embedding_name: str,
    output: Path,
    window_steps: int,
    period_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = stability_df["window_index"].to_numpy()
    ax.plot(x, stability_df["ari_vs_first"], marker="o", linewidth=1.2, markersize=3, label="ARI vs first")
    ax.plot(x, stability_df["ari_vs_previous"], marker="s", linewidth=1.2, markersize=3, label="ARI vs previous")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(_window_axis_label(window_steps))
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(f"{dataset} {embedding_name} cluster stability over {period_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_counts(labels_by_window: list[np.ndarray], num_clusters: int) -> pd.DataFrame:
    rows = []
    total = len(labels_by_window[0])
    for window_idx, labels in enumerate(labels_by_window):
        counts = np.bincount(labels, minlength=num_clusters)
        for cluster_id, count in enumerate(counts):
            rows.append(
                {
                    "window_index": window_idx,
                    "cluster": cluster_id,
                    "count": int(count),
                    "fraction": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def summarize_stability(labels_by_window: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    first = labels_by_window[0]
    for idx, labels in enumerate(labels_by_window):
        rows.append(
            {
                "window_index": idx,
                "ari_vs_first": float(adjusted_rand_score(first, labels)),
                "ari_vs_previous": float("nan") if idx == 0 else float(adjusted_rand_score(labels_by_window[idx - 1], labels)),
                "changed_fraction_vs_previous": float("nan")
                if idx == 0
                else float(np.mean(labels_by_window[idx - 1] != labels)),
            }
        )
    return pd.DataFrame(rows)


def run_one_embedding(
    spec: DatasetSpec,
    desc: dict,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    embedding_name: str,
    output_dir: Path,
) -> dict[str, float | str | int]:
    frequency = int(desc.get("frequency (minutes)", 15))
    steps_per_day = int(round(24 * 60 / frequency))
    window_steps = args.window_steps or steps_per_day
    stride_steps = args.stride_steps or window_steps
    week_steps = args.week_steps or 7 * steps_per_day
    available_windows = int(np.floor((week_steps - window_steps) / stride_steps)) + 1
    max_windows = available_windows if args.max_windows is None else min(args.max_windows, available_windows)
    if max_windows <= 0:
        raise ValueError("No windows to process; check week/window/stride settings.")

    labels_by_window: list[np.ndarray] = []
    titles: list[str] = []
    assignment_rows = []
    previous_centers = None
    last_aligned_centers = None

    for window_idx in range(max_windows):
        start = args.start_step + window_idx * stride_steps
        raw = load_signal_window(spec, desc, start, window_steps, args.feature_index)
        signal = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if not args.no_per_node_zscore:
            signal = node_zscore(signal)
        if embedding_name == "time":
            embedding = make_time_embedding(signal, args.downsample_time)
        elif embedding_name == "freq":
            embedding = make_freq_embedding(signal, args.freq_components)
        else:
            raise ValueError(embedding_name)
        embedding = clean_embedding(embedding)
        labels, centers = fit_kmeans(
            embedding,
            num_clusters=args.num_clusters,
            seed=args.seed + window_idx,
            n_init=args.n_init,
            max_iter=args.max_iter,
        )
        labels, aligned_centers = align_labels(labels, centers, previous_centers, args.num_clusters)
        previous_centers = aligned_centers
        last_aligned_centers = aligned_centers
        labels_by_window.append(labels)
        hour_start = (start % steps_per_day) * frequency / 60.0
        day_start = int(start // steps_per_day) + 1
        hour_end = ((start + window_steps) % steps_per_day) * frequency / 60.0
        if hour_end == 0.0 and window_steps > 0:
            hour_end = 24.0
        titles.append(f"w{window_idx:02d}, day {day_start}, {hour_start:04.1f}-{hour_end:04.1f}h")
        for node_idx, label in enumerate(labels):
            assignment_rows.append(
                {
                    "dataset": spec.name,
                    "embedding": embedding_name,
                    "window_index": window_idx,
                    "start_step": start,
                    "end_step": start + window_steps,
                    "node_index": node_idx,
                    "node_id": meta.iloc[node_idx]["_id"],
                    "cluster": int(label),
                }
            )

    prefix = output_dir / f"{spec.name}_{embedding_name}_k{args.num_clusters}"
    plot_cluster_maps(
        meta=meta,
        labels_by_window=labels_by_window,
        titles=titles,
        dataset=spec.name,
        embedding_name=embedding_name,
        output=prefix.with_name(prefix.name + "_maps_contact_sheet.png"),
        num_clusters=args.num_clusters,
    )
    count_df = summarize_counts(labels_by_window, args.num_clusters)
    stability_df = summarize_stability(labels_by_window)
    assignment_df = pd.DataFrame(assignment_rows)
    count_df.to_csv(prefix.with_name(prefix.name + "_cluster_counts.csv"), index=False)
    stability_df.to_csv(prefix.with_name(prefix.name + "_stability.csv"), index=False)
    assignment_df.to_csv(prefix.with_name(prefix.name + "_assignments.csv"), index=False)
    plot_cluster_fraction(count_df, spec.name, embedding_name, prefix.with_name(prefix.name + "_cluster_fraction.png"))
    plot_node_timeline(
        labels_by_window,
        spec.name,
        embedding_name,
        prefix.with_name(prefix.name + "_node_timeline.png"),
        args.num_clusters,
    )
    plot_ari_curve(
        stability_df,
        spec.name,
        embedding_name,
        prefix.with_name(prefix.name + "_stability_curve.png"),
        window_steps,
        args.period_name,
    )
    wrote_animation = plot_cluster_animation(
        meta,
        labels_by_window,
        titles,
        spec.name,
        embedding_name,
        prefix.with_name(prefix.name + "_cluster_animation.gif"),
        args.num_clusters,
    )
    if last_aligned_centers is not None:
        np.save(prefix.with_name(prefix.name + "_last_centers.npy"), last_aligned_centers)

    return {
        "dataset": spec.name,
        "embedding": embedding_name,
        "num_nodes": int(desc["num_nodes"]),
        "num_windows": int(max_windows),
        "frequency_minutes": int(frequency),
        "window_steps": int(window_steps),
        "stride_steps": int(stride_steps),
        "analysis_span_steps": int(week_steps),
        "period_name": str(args.period_name),
        "mean_ari_vs_previous": float(stability_df["ari_vs_previous"].dropna().mean()),
        "mean_changed_fraction_vs_previous": float(stability_df["changed_fraction_vs_previous"].dropna().mean()),
        "maps_contact_sheet": str(prefix.with_name(prefix.name + "_maps_contact_sheet.png")),
        "cluster_animation": str(prefix.with_name(prefix.name + "_cluster_animation.gif")) if wrote_animation else "",
        "cluster_fraction": str(prefix.with_name(prefix.name + "_cluster_fraction.png")),
        "node_timeline": str(prefix.with_name(prefix.name + "_node_timeline.png")),
        "stability_curve": str(prefix.with_name(prefix.name + "_stability_curve.png")),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for dataset in args.datasets:
        spec = load_dataset_spec(args.basicts_datasets_dir, dataset)
        desc = load_desc(spec)
        meta = load_meta(spec)
        if int(desc["num_nodes"]) != len(meta):
            raise ValueError(f"{dataset}: desc nodes={desc['num_nodes']} but meta rows={len(meta)}")
        dataset_out = args.output_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        for embedding_name in args.embeddings:
            summary = run_one_embedding(spec, desc, meta, args, embedding_name, dataset_out)
            all_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    summary_df = pd.DataFrame(all_summaries)
    summary_path = args.output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

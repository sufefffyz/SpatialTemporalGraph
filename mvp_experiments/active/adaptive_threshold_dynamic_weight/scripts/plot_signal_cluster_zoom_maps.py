#!/usr/bin/env python3
"""Plot zoomed maps for dense signal-cluster regions.

This diagnostic script reads existing KMeans assignment CSVs from
``visualize_signal_kmeans_clusters.py`` and metadata from BasicTS datasets.
It does not recompute clusters and does not affect training configs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class Region:
    index: int
    x0: float
    x1: float
    y0: float
    y1: float
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basicts-datasets-dir", type=Path, default=Path("BasicTS/datasets"))
    parser.add_argument(
        "--cluster-output-dir",
        type=Path,
        default=Path("mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/signal_kmeans_clusters_1day_month"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/signal_kmeans_clusters_1day_month_zoom"),
    )
    parser.add_argument("--datasets", nargs="+", default=["GLA", "GBA"])
    parser.add_argument("--embeddings", nargs="+", choices=["time", "freq"], default=["time", "freq"])
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--selected-windows", nargs="+", type=int, default=[0, 7, 14, 21, 28, 29])
    parser.add_argument("--grid-size", type=int, default=6, help="Grid used to find dense map regions.")
    parser.add_argument("--top-regions", type=int, default=4)
    parser.add_argument("--region-pad-ratio", type=float, default=0.28)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")


def load_meta(dataset_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(dataset_dir / "meta.csv")
    lat_col = find_column(meta, ("Lat", "lat", "latitude"))
    lon_col = find_column(meta, ("Lng", "lng", "lon", "longitude"))
    out = meta.copy()
    out["_lat"] = out[lat_col].astype(float)
    out["_lon"] = out[lon_col].astype(float)
    return out


def load_assignments(cluster_output_dir: Path, dataset: str, embedding: str, num_clusters: int) -> pd.DataFrame:
    path = cluster_output_dir / dataset / f"{dataset}_{embedding}_k{num_clusters}_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def select_dense_regions(lon: np.ndarray, lat: np.ndarray, grid_size: int, top_regions: int, pad_ratio: float) -> list[Region]:
    x_edges = np.linspace(float(lon.min()), float(lon.max()), grid_size + 1)
    y_edges = np.linspace(float(lat.min()), float(lat.max()), grid_size + 1)
    counts, _, _ = np.histogram2d(lon, lat, bins=[x_edges, y_edges])
    candidates: list[tuple[int, int, int]] = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            count = int(counts[ix, iy])
            if count > 0:
                candidates.append((count, ix, iy))
    candidates.sort(reverse=True)

    regions: list[Region] = []
    used: set[tuple[int, int]] = set()
    for count, ix, iy in candidates:
        if len(regions) >= top_regions:
            break
        if (ix, iy) in used:
            continue
        x0, x1 = float(x_edges[ix]), float(x_edges[ix + 1])
        y0, y1 = float(y_edges[iy]), float(y_edges[iy + 1])
        x_pad = (x1 - x0) * pad_ratio
        y_pad = (y1 - y0) * pad_ratio
        regions.append(
            Region(
                index=len(regions),
                x0=max(float(lon.min()), x0 - x_pad),
                x1=min(float(lon.max()), x1 + x_pad),
                y0=max(float(lat.min()), y0 - y_pad),
                y1=min(float(lat.max()), y1 + y_pad),
                count=count,
            )
        )
        used.add((ix, iy))
    return regions


def marker_size(num_points: int) -> float:
    return float(np.clip(95.0 / np.sqrt(max(num_points, 1)), 2.2, 12.0))


def plot_region_overview(
    lon: np.ndarray,
    lat: np.ndarray,
    regions: list[Region],
    dataset: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(lon, lat, s=3.0, color="#9aa0a6", alpha=0.35, linewidths=0)
    for region in regions:
        rect = Rectangle(
            (region.x0, region.y0),
            region.x1 - region.x0,
            region.y1 - region.y0,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(
            region.x0,
            region.y1,
            f"R{region.index}: {region.count}",
            color="#d62728",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.set_title(f"{dataset} dense regions for zoomed cluster maps")
    ax.set_xlabel("Lng")
    ax.set_ylabel("Lat")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_region_contact_sheet(
    meta: pd.DataFrame,
    assignments: pd.DataFrame,
    region: Region,
    dataset: str,
    embedding: str,
    selected_windows: list[int],
    num_clusters: int,
    output: Path,
    alpha: float,
    dpi: int,
) -> None:
    lon = meta["_lon"].to_numpy()
    lat = meta["_lat"].to_numpy()
    mask = (lon >= region.x0) & (lon <= region.x1) & (lat >= region.y0) & (lat <= region.y1)
    node_indices = np.where(mask)[0]
    if len(node_indices) == 0:
        return

    labels_by_window = {}
    for window_idx, group in assignments.groupby("window_index"):
        labels_by_window[int(window_idx)] = group.sort_values("node_index")["cluster"].to_numpy()

    cols = min(3, len(selected_windows))
    rows = int(np.ceil(len(selected_windows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 4.4 * rows), squeeze=False)
    cmap = plt.get_cmap("tab20", num_clusters)
    size = marker_size(len(node_indices))

    for plot_idx, ax in enumerate(axes.ravel()):
        if plot_idx >= len(selected_windows):
            ax.axis("off")
            continue
        window_idx = selected_windows[plot_idx]
        labels = labels_by_window[window_idx][node_indices]
        sc = ax.scatter(
            lon[node_indices],
            lat[node_indices],
            c=labels,
            s=size,
            cmap=cmap,
            vmin=-0.5,
            vmax=num_clusters - 0.5,
            alpha=alpha,
            linewidths=0,
        )
        ax.set_xlim(region.x0, region.x1)
        ax.set_ylim(region.y0, region.y1)
        ax.set_title(f"day {window_idx + 1}", fontsize=10)
        ax.set_xlabel("Lng")
        ax.set_ylabel("Lat")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.5)

    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.82, ticks=np.arange(num_clusters), label="Cluster")
    fig.suptitle(
        f"{dataset} {embedding} R{region.index}: {len(node_indices)} nodes, selected daily windows",
        fontsize=13,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for dataset in args.datasets:
        dataset_dir = args.basicts_datasets_dir / dataset
        meta = load_meta(dataset_dir)
        lon = meta["_lon"].to_numpy()
        lat = meta["_lat"].to_numpy()
        regions = select_dense_regions(lon, lat, args.grid_size, args.top_regions, args.region_pad_ratio)
        dataset_out = args.output_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        overview_path = dataset_out / f"{dataset}_dense_regions_overview.png"
        plot_region_overview(lon, lat, regions, dataset, overview_path)
        for embedding in args.embeddings:
            assignments = load_assignments(args.cluster_output_dir, dataset, embedding, args.num_clusters)
            for region in regions:
                out_path = dataset_out / f"{dataset}_{embedding}_k{args.num_clusters}_region{region.index}_zoom_maps.png"
                plot_region_contact_sheet(
                    meta=meta,
                    assignments=assignments,
                    region=region,
                    dataset=dataset,
                    embedding=embedding,
                    selected_windows=args.selected_windows,
                    num_clusters=args.num_clusters,
                    output=out_path,
                    alpha=args.alpha,
                    dpi=args.dpi,
                )
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "embedding": embedding,
                        "region": region.index,
                        "hist_count": region.count,
                        "x0": region.x0,
                        "x1": region.x1,
                        "y0": region.y0,
                        "y1": region.y1,
                        "plot": str(out_path),
                    }
                )
                print(f"wrote {out_path}")
        print(f"wrote {overview_path}")
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "zoom_map_summary.csv", index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Daily signal clustering diagnostics for traffic datasets.

This script is analysis-only. It reads BasicTS-style ``data.dat`` and
``desc.json`` files, builds daily node profiles, runs fixed-K and adaptive
overcluster-merge clustering, and writes figures/statistics. It does not touch
forecasting models or graph construction code.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


FALLBACK_META = {
    "PEMS-BAY": ["BasicTS/datasets/raw_data/PEMS-BAY/graph_sensor_locations_bay_ordered.csv"],
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_dir: Path
    desc_path: Path
    data_path: Path
    meta_path: Path | None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["METR-LA", "PEMS-BAY", "PEMSD3_2025_full_phys", "PEMSD4_2025_full_phys"],
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "BasicTS" / "datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "mvp_experiments" / "active" / "adaptive_signal_clustering" / "outputs",
    )
    parser.add_argument("--run-name", default="traffic_daily_zshape_30d")
    parser.add_argument("--feature-index", type=int, default=0)
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--num-days", type=int, default=30)
    parser.add_argument(
        "--target-steps-per-day",
        type=int,
        default=96,
        help="Resample each daily profile to this length. Use 0 to keep native frequency.",
    )
    parser.add_argument("--methods", nargs="+", choices=["fixed", "scan", "merge"], default=["fixed", "scan", "merge"])
    parser.add_argument("--fixed-k", type=int, default=8)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--rho-merge", type=float, default=1.5)
    parser.add_argument("--radius-quantile", type=float, default=0.75)
    parser.add_argument("--min-cluster-ratio", type=float, default=0.005)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--global-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--metric-sample-size", type=int, default=2000)
    parser.add_argument("--scan-sample-size", type=int, default=1500)
    parser.add_argument("--max-map-days", type=int, default=8)
    parser.add_argument("--skip-maps", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def find_meta_path(repo_root: Path, dataset_dir: Path, dataset: str) -> Path | None:
    direct_candidates = [
        dataset_dir / "meta.csv",
        dataset_dir / "sensor_catalog.csv",
        dataset_dir / "nodes.csv",
    ]
    for path in direct_candidates:
        if path.exists():
            return path
    for rel in FALLBACK_META.get(dataset, []):
        path = repo_root / rel
        if path.exists():
            return path
    return None


def load_dataset_spec(repo_root: Path, dataset_root: Path, name: str) -> DatasetSpec:
    dataset_dir = dataset_root / name
    spec = DatasetSpec(
        name=name,
        dataset_dir=dataset_dir,
        desc_path=dataset_dir / "desc.json",
        data_path=dataset_dir / "data.dat",
        meta_path=find_meta_path(repo_root, dataset_dir, name),
    )
    missing = [path for path in [spec.desc_path, spec.data_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{name}: missing {missing}")
    return spec


def column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = column_lookup(df.columns)
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_meta(spec: DatasetSpec, num_nodes: int) -> pd.DataFrame | None:
    if spec.meta_path is None:
        return None
    meta = pd.read_csv(spec.meta_path)
    if "node_index" in meta.columns:
        meta = meta.sort_values("node_index").reset_index(drop=True)
    if len(meta) != num_nodes:
        return None

    lat_col = find_column(meta, ["lat", "latitude", "Lat", "Latitude"])
    lon_col = find_column(meta, ["lon", "lng", "longitude", "Lng", "Longitude"])
    if lat_col is None or lon_col is None:
        return None

    id_col = find_column(meta, ["ID", "id", "sensor_id", "node_id", "graph_node"]) or meta.columns[0]
    out = meta.copy()
    out["_node_index"] = np.arange(len(out))
    out["_id"] = out[id_col].astype(str)
    out["_lat"] = out[lat_col].astype(float)
    out["_lon"] = out[lon_col].astype(float)
    return out


def day_steps_from_desc(desc: dict) -> int:
    freq = int(desc.get("frequency (minutes)", 5))
    return int(round(24 * 60 / freq))


def load_daily_raw_profiles(
    spec: DatasetSpec,
    desc: dict,
    feature_index: int,
    start_day: int,
    num_days: int,
    target_steps_per_day: int,
) -> np.ndarray:
    shape = tuple(desc["shape"])
    native_steps = day_steps_from_desc(desc)
    if feature_index < 0 or feature_index >= shape[2]:
        raise ValueError(f"{spec.name}: feature_index={feature_index} outside feature dim={shape[2]}")
    total_days = shape[0] // native_steps
    if start_day < 0 or start_day >= total_days:
        raise ValueError(f"{spec.name}: start_day={start_day} outside total_days={total_days}")
    actual_days = min(num_days, total_days - start_day)
    mmap = np.memmap(spec.data_path, dtype="float32", mode="r", shape=shape)
    raw = np.asarray(
        mmap[start_day * native_steps : (start_day + actual_days) * native_steps, :, feature_index],
        dtype=np.float32,
    )
    raw = raw.reshape(actual_days, native_steps, shape[1])
    target_steps = target_steps_per_day or native_steps
    if target_steps == native_steps:
        profiles = raw.transpose(0, 2, 1)
    elif native_steps % target_steps == 0:
        factor = native_steps // target_steps
        profiles = raw.reshape(actual_days, target_steps, factor, shape[1]).mean(axis=2).transpose(0, 2, 1)
    else:
        profiles = np.empty((actual_days, shape[1], target_steps), dtype=np.float32)
        old_x = np.linspace(0.0, 1.0, native_steps)
        new_x = np.linspace(0.0, 1.0, target_steps)
        for day in range(actual_days):
            day_values = raw[day]
            for node in range(shape[1]):
                profiles[day, node] = np.interp(new_x, old_x, day_values[:, node])
    return np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def zshape_profiles(raw_profiles: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = raw_profiles.mean(axis=2, keepdims=True)
    std = raw_profiles.std(axis=2, keepdims=True)
    return ((raw_profiles - mean) / np.maximum(std, eps)).astype(np.float32)


def standardize_embedding(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))


def fit_kmeans(x: np.ndarray, k: int, seed: int, n_init: int, max_iter: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter, algorithm="lloyd")
    return model.fit_predict(x).astype(int)


def reindex_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(int(v) for v in np.unique(labels) if v >= 0)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping.get(int(v), -1) for v in labels], dtype=int)


def centers_from_labels(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(int(v) for v in np.unique(labels) if v >= 0)
    centers = []
    counts = []
    for label in unique:
        mask = labels == label
        centers.append(x[mask].mean(axis=0))
        counts.append(int(mask.sum()))
    return np.vstack(centers), np.asarray(counts, dtype=int)


def merge_tiny_clusters(x: np.ndarray, labels: np.ndarray, min_cluster_ratio: float) -> np.ndarray:
    labels = reindex_labels(labels)
    min_size = max(1, int(math.ceil(min_cluster_ratio * len(labels))))
    while True:
        centers, counts = centers_from_labels(x, labels)
        tiny = np.where(counts < min_size)[0]
        if len(tiny) == 0 or len(counts) <= 1:
            return reindex_labels(labels)
        target = int(tiny[np.argmin(counts[tiny])])
        dist = np.linalg.norm(centers - centers[target], axis=1)
        dist[target] = np.inf
        nearest = int(np.argmin(dist))
        labels[labels == target] = nearest
        labels = reindex_labels(labels)


def cluster_radii(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, q: float) -> np.ndarray:
    radii = np.zeros(len(centers), dtype=np.float64)
    for label in range(len(centers)):
        vals = x[labels == label]
        if len(vals) == 0:
            radii[label] = 0.0
        else:
            d = np.linalg.norm(vals - centers[label], axis=1)
            radii[label] = float(np.quantile(d, q))
    return radii


def overcluster_merge(
    x: np.ndarray,
    k_max: int,
    rho_merge: float,
    radius_quantile: float,
    min_cluster_ratio: float,
    seed: int,
    n_init: int,
    max_iter: int,
) -> np.ndarray:
    k0 = min(k_max, len(x))
    labels = fit_kmeans(x, k0, seed, n_init, max_iter)
    labels = merge_tiny_clusters(x, labels, min_cluster_ratio)
    eps = 1e-8
    for _ in range(k0):
        labels = reindex_labels(labels)
        centers, counts = centers_from_labels(x, labels)
        if len(centers) <= 2:
            break
        radii = cluster_radii(x, labels, centers, radius_quantile)
        dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        denom = radii[:, None] + radii[None, :] + eps
        separation = dist / denom
        np.fill_diagonal(separation, np.inf)
        pair = np.unravel_index(int(np.argmin(separation)), separation.shape)
        if not np.isfinite(separation[pair]) or float(separation[pair]) >= rho_merge:
            break
        keep = int(pair[0]) if counts[pair[0]] >= counts[pair[1]] else int(pair[1])
        drop = int(pair[1]) if keep == int(pair[0]) else int(pair[0])
        labels[labels == drop] = keep
        labels = merge_tiny_clusters(x, labels, min_cluster_ratio)
    return reindex_labels(labels)


def inter_intra_ratio(x: np.ndarray, labels: np.ndarray, q: float) -> float:
    centers, _ = centers_from_labels(x, labels)
    if len(centers) <= 1:
        return float("nan")
    radii = cluster_radii(x, labels, centers, q)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    denom = radii[:, None] + radii[None, :] + 1e-8
    ratio = dist / denom
    np.fill_diagonal(ratio, np.inf)
    return float(np.min(ratio))


def score_adaptive_k(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_size: int,
    radius_quantile: float,
    min_cluster_ratio: float,
) -> tuple[float, dict[str, float]]:
    labels = reindex_labels(labels)
    k = len(np.unique(labels))
    counts = np.bincount(labels, minlength=k)
    max_cluster_ratio = float(counts.max() / counts.sum())
    min_cluster_ratio_actual = float(counts.min() / counts.sum())
    if min_cluster_ratio_actual < min_cluster_ratio:
        return -float("inf"), {
            "score": -float("inf"),
            "mean_silhouette": float("nan"),
            "bottom20_silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "min_inter_intra_ratio": float("nan"),
            "max_cluster_ratio": max_cluster_ratio,
            "min_cluster_ratio": min_cluster_ratio_actual,
        }
    rng = np.random.default_rng(seed)
    if 0 < sample_size < len(x):
        idx = rng.choice(len(x), size=sample_size, replace=False)
        x_eval = x[idx]
        labels_eval = labels[idx]
    else:
        x_eval = x
        labels_eval = labels
    if len(np.unique(labels_eval)) <= 1:
        return -float("inf"), {
            "score": -float("inf"),
            "mean_silhouette": float("nan"),
            "bottom20_silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "min_inter_intra_ratio": float("nan"),
            "max_cluster_ratio": max_cluster_ratio,
            "min_cluster_ratio": min_cluster_ratio_actual,
        }
    sil = silhouette_samples(x_eval, labels_eval)
    mean_sil = float(np.mean(sil))
    bottom20 = float(np.quantile(sil, 0.2))
    db = float(davies_bouldin_score(x, labels))
    ratio = inter_intra_ratio(x, labels, radius_quantile)
    imbalance = max_cluster_ratio - 1.0 / max(1, k)
    score = mean_sil + 0.5 * bottom20 + 0.2 * math.log(max(ratio, 1e-8)) - 0.2 * db - 0.1 * imbalance
    return float(score), {
        "score": float(score),
        "mean_silhouette": mean_sil,
        "bottom20_silhouette": bottom20,
        "davies_bouldin": db,
        "min_inter_intra_ratio": float(ratio),
        "max_cluster_ratio": max_cluster_ratio,
        "min_cluster_ratio": min_cluster_ratio_actual,
    }


def adaptive_k_scan(
    x: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    n_init: int,
    max_iter: int,
    sample_size: int,
    radius_quantile: float,
    min_cluster_ratio: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    best_score = -float("inf")
    best_labels: np.ndarray | None = None
    for k in range(max(2, k_min), min(k_max, len(x) - 1) + 1):
        labels = fit_kmeans(x, k, seed + k, n_init, max_iter)
        score, parts = score_adaptive_k(
            x=x,
            labels=labels,
            seed=seed + 1009 * k,
            sample_size=sample_size,
            radius_quantile=radius_quantile,
            min_cluster_ratio=min_cluster_ratio,
        )
        rows.append({"k": k, **parts})
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_labels is None:
        best_labels = fit_kmeans(x, min(max(2, k_min), len(x)), seed, n_init, max_iter)
    return reindex_labels(best_labels), pd.DataFrame(rows)


def confidence_from_centers(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers, _ = centers_from_labels(x, labels)
    if len(centers) <= 1:
        return np.ones(len(x), dtype=np.float64)
    dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    best_two = np.partition(dist, kth=1, axis=1)[:, :2]
    d1 = best_two[:, 0]
    d2 = best_two[:, 1]
    return (d2 - d1) / (d2 + 1e-8)


def cluster_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_size: int,
) -> dict[str, float | int]:
    valid = labels >= 0
    x_valid = x[valid]
    labels_valid = labels[valid]
    k = len(np.unique(labels_valid))
    counts = np.bincount(reindex_labels(labels_valid), minlength=k) if k > 0 else np.array([], dtype=int)
    if k <= 1 or len(x_valid) <= k:
        silhouette = float("nan")
        db = float("nan")
        ch = float("nan")
    else:
        ss = min(sample_size, len(x_valid)) if sample_size > 0 else None
        silhouette = float(silhouette_score(x_valid, labels_valid, sample_size=ss, random_state=seed))
        db = float(davies_bouldin_score(x_valid, labels_valid))
        ch = float(calinski_harabasz_score(x_valid, labels_valid))
    frac = counts / max(1, counts.sum()) if len(counts) else np.array([])
    entropy = float(-(frac * np.log(frac + 1e-12)).sum() / max(1e-12, np.log(max(2, len(counts))))) if len(frac) else 0.0
    return {
        "k": int(k),
        "mean_silhouette_sampled": silhouette,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "cluster_size_entropy": entropy,
        "max_cluster_ratio": float(frac.max()) if len(frac) else float("nan"),
        "uncertain_ratio": float(np.mean(labels < 0)),
    }


def global_align_labels(
    profiles: np.ndarray,
    local_labels_by_day: np.ndarray,
    global_k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    all_centers = []
    rows = []
    for day in range(local_labels_by_day.shape[0]):
        labels = local_labels_by_day[day]
        for local_label in sorted(int(v) for v in np.unique(labels) if v >= 0):
            mask = labels == local_label
            if not np.any(mask):
                continue
            all_centers.append(profiles[day, mask].mean(axis=0))
            rows.append({"day": day, "local_cluster": local_label, "size": int(mask.sum())})
    if not all_centers:
        raise ValueError("No local clusters to align.")
    center_matrix = standardize_embedding(np.vstack(all_centers))
    k = min(global_k, len(all_centers))
    global_center_labels = fit_kmeans(center_matrix, k, seed, n_init, max_iter)
    mapping = {}
    for row, global_label in zip(rows, global_center_labels):
        mapping[(int(row["day"]), int(row["local_cluster"]))] = int(global_label)
        row["global_cluster"] = int(global_label)
    aligned = np.full_like(local_labels_by_day, fill_value=-1, dtype=int)
    for day in range(local_labels_by_day.shape[0]):
        for local_label in np.unique(local_labels_by_day[day]):
            if local_label < 0:
                continue
            aligned[day, local_labels_by_day[day] == local_label] = mapping[(day, int(local_label))]
    global_profiles = []
    for global_label in range(k):
        vals = []
        for day in range(aligned.shape[0]):
            mask = aligned[day] == global_label
            if np.any(mask):
                vals.append(profiles[day, mask])
        global_profiles.append(np.vstack(vals).mean(axis=0))
    return aligned, np.vstack(global_profiles), pd.DataFrame(rows)


def stability_table(labels_by_day: np.ndarray) -> pd.DataFrame:
    rows = []
    first = labels_by_day[0]
    for day in range(labels_by_day.shape[0]):
        labels = labels_by_day[day]
        prev = float("nan") if day == 0 else float(adjusted_rand_score(labels_by_day[day - 1], labels))
        changed = float("nan") if day == 0 else float(np.mean(labels_by_day[day - 1] != labels))
        rows.append(
            {
                "day": int(day),
                "ari_vs_first": float(adjusted_rand_score(first, labels)),
                "ari_vs_previous": prev,
                "changed_fraction_vs_previous": changed,
            }
        )
    return pd.DataFrame(rows)


def select_days(num_days: int, max_days: int) -> list[int]:
    if num_days <= max_days:
        return list(range(num_days))
    return sorted(set(int(v) for v in np.linspace(0, num_days - 1, max_days)))


def plot_map_contact(
    meta: pd.DataFrame,
    labels_by_day: np.ndarray,
    dataset: str,
    method: str,
    output: Path,
    max_days: int,
) -> None:
    days = select_days(labels_by_day.shape[0], max_days)
    cols = min(4, len(days))
    rows = int(math.ceil(len(days) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows), squeeze=False)
    lon = meta["_lon"].to_numpy()
    lat = meta["_lat"].to_numpy()
    k = max(1, int(labels_by_day[labels_by_day >= 0].max()) + 1) if np.any(labels_by_day >= 0) else 1
    cmap = plt.get_cmap("tab20", max(20, k))
    size = max(2.0, min(18.0, 4500.0 / max(1, len(meta))))
    for ax_idx, ax in enumerate(axes.ravel()):
        if ax_idx >= len(days):
            ax.axis("off")
            continue
        day = days[ax_idx]
        labels = labels_by_day[day]
        uncertain = labels < 0
        if np.any(~uncertain):
            ax.scatter(lon[~uncertain], lat[~uncertain], c=labels[~uncertain], s=size, cmap=cmap, vmin=-0.5, vmax=k - 0.5, alpha=0.9, linewidths=0)
        if np.any(uncertain):
            ax.scatter(lon[uncertain], lat[uncertain], c="#bdbdbd", s=size, alpha=0.35, linewidths=0)
        ax.set_title(f"Day {day + 1}")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.5)
    fig.suptitle(f"{dataset} {method}: global-aligned daily clusters", fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_center_profiles(global_profiles: np.ndarray, dataset: str, method: str, output: Path) -> None:
    k = len(global_profiles)
    cols = min(4, k)
    rows = int(math.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), squeeze=False, sharex=True, sharey=True)
    x = np.arange(global_profiles.shape[1])
    cmap = plt.get_cmap("tab20", max(20, k))
    for idx, ax in enumerate(axes.ravel()):
        if idx >= k:
            ax.axis("off")
            continue
        ax.plot(x, global_profiles[idx], color=cmap(idx), linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(f"Global C{idx}")
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("15-min step in day")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-shape")
    fig.suptitle(f"{dataset} {method}: global cluster center profiles", fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fraction_heatmap(labels_by_day: np.ndarray, dataset: str, method: str, output: Path) -> pd.DataFrame:
    rows = []
    k = max(1, int(labels_by_day[labels_by_day >= 0].max()) + 1) if np.any(labels_by_day >= 0) else 1
    for day in range(labels_by_day.shape[0]):
        labels = labels_by_day[day]
        total = len(labels)
        for cluster in range(k):
            rows.append({"day": day, "cluster": cluster, "fraction": float(np.mean(labels == cluster))})
        rows.append({"day": day, "cluster": -1, "fraction": float(np.mean(labels < 0))})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="cluster", columns="day", values="fraction").sort_index()
    fig, ax = plt.subplots(figsize=(max(7.0, 0.32 * labels_by_day.shape[0]), max(3.2, 0.35 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(np.arange(labels_by_day.shape[0]), labels=[str(i + 1) for i in range(labels_by_day.shape[0])], rotation=90)
    ax.set_yticks(np.arange(len(pivot)), labels=[str(v) for v in pivot.index])
    ax.set_xlabel("Day")
    ax.set_ylabel("Global cluster (-1 = uncertain)")
    ax.set_title(f"{dataset} {method}: cluster fractions")
    fig.colorbar(im, ax=ax, label="node fraction")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_stability(stability: pd.DataFrame, dataset: str, method: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(stability["day"] + 1, stability["ari_vs_first"], marker="o", markersize=3, linewidth=1.4, label="ARI vs day 1")
    ax.plot(stability["day"] + 1, stability["ari_vs_previous"], marker="s", markersize=3, linewidth=1.4, label="ARI vs previous day")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Day")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(f"{dataset} {method}: cluster stability")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selected_k(metrics: pd.DataFrame, dataset: str, method: str, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 3.4))
    ax1.plot(metrics["day"] + 1, metrics["k"], marker="o", markersize=3, linewidth=1.4, color="#1f77b4")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("selected K")
    ax1.set_title(f"{dataset} {method}: selected K by day")
    ax1.grid(True, alpha=0.25)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_hist(confidence: np.ndarray, dataset: str, method: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(confidence.reshape(-1), bins=50, color="#4e79a7", alpha=0.86)
    ax.set_xlabel("margin confidence")
    ax.set_ylabel("node-day count")
    ax.set_title(f"{dataset} {method}: assignment confidence")
    ax.grid(True, alpha=0.2)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_method(
    method: str,
    dataset: str,
    profiles: np.ndarray,
    meta: pd.DataFrame | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | int | str]:
    labels_by_day = []
    confidence_by_day = []
    metric_rows = []
    scan_score_tables = []
    for day in range(profiles.shape[0]):
        print(f"[{dataset}][{method}] day {day + 1}/{profiles.shape[0]}", flush=True)
        x = standardize_embedding(profiles[day])
        if method == "fixed":
            labels = fit_kmeans(x, min(args.fixed_k, len(x)), args.seed + day, args.n_init, args.max_iter)
            confidence = confidence_from_centers(x, labels)
        elif method == "scan":
            labels, score_table = adaptive_k_scan(
                x=x,
                k_min=args.k_min,
                k_max=args.k_max,
                seed=args.seed + day,
                n_init=args.n_init,
                max_iter=args.max_iter,
                sample_size=args.scan_sample_size,
                radius_quantile=args.radius_quantile,
                min_cluster_ratio=args.min_cluster_ratio,
            )
            score_table.insert(0, "day", day)
            scan_score_tables.append(score_table)
            confidence = confidence_from_centers(x, labels)
        elif method == "merge":
            labels = overcluster_merge(
                x=x,
                k_max=args.k_max,
                rho_merge=args.rho_merge,
                radius_quantile=args.radius_quantile,
                min_cluster_ratio=args.min_cluster_ratio,
                seed=args.seed + day,
                n_init=args.n_init,
                max_iter=args.max_iter,
            )
            confidence = confidence_from_centers(x, labels)
            labels = labels.copy()
            labels[confidence < args.confidence_threshold] = -1
        else:
            raise ValueError(method)
        labels = reindex_labels(labels)
        labels_by_day.append(labels)
        confidence_by_day.append(confidence)
        row = {"dataset": dataset, "method": method, "day": day}
        row.update(cluster_metrics(x, labels, args.seed + day, args.metric_sample_size))
        row["mean_confidence"] = float(np.mean(confidence))
        row["bottom20_confidence"] = float(np.quantile(confidence, 0.2))
        metric_rows.append(row)

    labels_arr = np.stack(labels_by_day, axis=0)
    confidence_arr = np.stack(confidence_by_day, axis=0)
    aligned, global_profiles, local_global = global_align_labels(
        profiles=profiles,
        local_labels_by_day=labels_arr,
        global_k=args.global_k,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "labels_local.npy", labels_arr)
    np.save(method_dir / "labels_global.npy", aligned)
    np.save(method_dir / "confidence.npy", confidence_arr)
    np.save(method_dir / "global_center_profiles.npy", global_profiles)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(method_dir / "metrics_daily.csv", index=False)
    if scan_score_tables:
        pd.concat(scan_score_tables, ignore_index=True).to_csv(method_dir / "scan_scores_daily.csv", index=False)
    local_global.to_csv(method_dir / "local_to_global_clusters.csv", index=False)
    stability = stability_table(aligned)
    stability.to_csv(method_dir / "stability_daily.csv", index=False)
    fractions = plot_fraction_heatmap(aligned, dataset, method, method_dir / "cluster_fraction_heatmap.png")
    fractions.to_csv(method_dir / "cluster_fraction_daily.csv", index=False)
    plot_center_profiles(global_profiles, dataset, method, method_dir / "global_center_profiles.png")
    plot_stability(stability, dataset, method, method_dir / "stability_curve.png")
    plot_selected_k(metrics, dataset, method, method_dir / "selected_k_daily.png")
    plot_confidence_hist(confidence_arr, dataset, method, method_dir / "confidence_hist.png")
    if meta is not None and not args.skip_maps:
        plot_map_contact(meta, aligned, dataset, method, method_dir / "cluster_maps_contact_sheet.png", args.max_map_days)

    summary = {
        "dataset": dataset,
        "method": method,
        "num_days": int(profiles.shape[0]),
        "num_nodes": int(profiles.shape[1]),
        "profile_steps": int(profiles.shape[2]),
        "mean_k": float(metrics["k"].mean()),
        "mean_silhouette_sampled": float(metrics["mean_silhouette_sampled"].mean()),
        "mean_davies_bouldin": float(metrics["davies_bouldin"].mean()),
        "mean_cluster_size_entropy": float(metrics["cluster_size_entropy"].mean()),
        "mean_max_cluster_ratio": float(metrics["max_cluster_ratio"].mean()),
        "mean_uncertain_ratio": float(metrics["uncertain_ratio"].mean()),
        "mean_confidence": float(metrics["mean_confidence"].mean()),
        "mean_ari_vs_previous": float(stability["ari_vs_previous"].dropna().mean()),
        "mean_changed_fraction_vs_previous": float(stability["changed_fraction_vs_previous"].dropna().mean()),
        "output_dir": str(method_dir),
    }
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_dataset(spec: DatasetSpec, args: argparse.Namespace, run_dir: Path) -> list[dict[str, float | int | str]]:
    desc = read_json(spec.desc_path)
    num_nodes = int(desc["num_nodes"])
    dataset_dir = run_dir / spec.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.name}] build daily profiles", flush=True)
    raw = load_daily_raw_profiles(
        spec=spec,
        desc=desc,
        feature_index=args.feature_index,
        start_day=args.start_day,
        num_days=args.num_days,
        target_steps_per_day=args.target_steps_per_day,
    )
    profiles = zshape_profiles(raw)
    np.save(dataset_dir / "daily_profiles_raw.npy", raw)
    np.save(dataset_dir / "daily_profiles_time_zshape.npy", profiles)
    meta = load_meta(spec, num_nodes)
    meta_status = "ok" if meta is not None else "missing_or_unusable"
    if meta is not None:
        meta.to_csv(dataset_dir / "resolved_meta.csv", index=False)
    data_summary = {
        "dataset": spec.name,
        "dataset_dir": str(spec.dataset_dir),
        "desc_shape": desc["shape"],
        "frequency_minutes": int(desc.get("frequency (minutes)", 5)),
        "native_steps_per_day": day_steps_from_desc(desc),
        "profile_steps_per_day": int(profiles.shape[2]),
        "start_day": int(args.start_day),
        "num_days": int(profiles.shape[0]),
        "num_nodes": int(profiles.shape[1]),
        "feature_index": int(args.feature_index),
        "meta_path": str(spec.meta_path) if spec.meta_path else "",
        "meta_status": meta_status,
    }
    (dataset_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    summaries = []
    for method in args.methods:
        print(f"[{spec.name}] start method={method}", flush=True)
        summaries.append(run_method(method, spec.name, profiles, meta, args, dataset_dir))
    return summaries


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    run_dir = (args.output_dir / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.datasets:
        spec = load_dataset_spec(args.repo_root, args.dataset_root, name)
        dataset_summaries = run_dataset(spec, args, run_dir)
        summaries.extend(dataset_summaries)
        for summary in dataset_summaries:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"wrote {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

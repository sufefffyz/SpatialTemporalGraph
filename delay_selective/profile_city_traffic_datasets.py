#!/usr/bin/env python3
"""Profile city-traffic-M speed/volume datasets for delay-selective validation.

The profiler scans target value distributions, missingness, temporal coverage,
graph degree distributions, edge distance distributions, and available road
metadata. It also flags when a file is a road-type subgraph rather than the
canonical city-traffic-M dataset described in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npformat


PAPER_CITY_TRAFFIC_M = {
    "paper": "Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks",
    "dataset": "city-traffic-M",
    "nodes": 53530,
    "edges": 121236,
    "timestamps": 35449,
    "train_timestamps": 26208,
    "val_timestamps": 4032,
    "test_timestamps": 5209,
    "frequency_seconds": 300,
    "timezone": "UTC+5",
    "target_variables": ["speed", "volume"],
    "node_definition": "road segment",
    "edge_definition": "directed road adjacency when movement is permitted by traffic rules",
    "static_road_attributes": 26,
}

DEFAULT_DATASETS = [
    ("speed_category_1_0", "data/city_traffic_m_speed__category__1_0.npz"),
    ("volume_category_1_0", "data/city_traffic_m_volume__category__1_0.npz"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile city-traffic-M speed/volume datasets.")
    parser.add_argument("--output-dir", default="delay_selective/outputs/dataset_profile")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as name:path. Can be repeated. Defaults to local category=1.0 subgraphs.",
    )
    parser.add_argument(
        "--target-array",
        action="append",
        default=[],
        help="Optional target override as name:path_to_targets.npy. Useful for memmapped product targets.",
    )
    parser.add_argument("--max-time-steps", type=int, default=0, help="0 means full time axis.")
    parser.add_argument("--sample-time-steps", type=int, default=8000)
    parser.add_argument("--sample-nodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hist-bins", type=int, default=100)
    parser.add_argument(
        "--target-max-load-gb",
        type=float,
        default=64.0,
        help=(
            "Skip target value sampling from compressed NPZ when estimated uncompressed "
            "target size exceeds this limit. Ignored for --target-array memmaps."
        ),
    )
    parser.add_argument(
        "--force-load-targets",
        action="store_true",
        help="Force loading targets from NPZ even if they exceed --target-max-load-gb.",
    )
    parser.add_argument(
        "--distance-bins-m",
        default="0,25,50,100,200,500,1000,2000,5000,inf",
        help="Comma-separated edge centroid-distance bins in meters.",
    )
    return parser.parse_args()


def parse_named_paths(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in items:
        if ":" not in spec:
            raise ValueError(f"Expected name:path, got {spec!r}")
        name, path = spec.split(":", 1)
        out[name] = path
    return out


def dataset_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.dataset:
        return [(name, path) for name, path in DEFAULT_DATASETS if Path(path).exists()]
    specs: list[tuple[str, str]] = []
    for spec in args.dataset:
        if ":" not in spec:
            raise ValueError(f"--dataset must be name:path, got {spec!r}")
        name, path = spec.split(":", 1)
        specs.append((name, path))
    return specs


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, obj: Any) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=to_jsonable)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_bins(spec: str) -> np.ndarray:
    values = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        values.append(float("inf") if item in {"inf", "infinity"} else float(item))
    if len(values) < 2:
        raise ValueError("Need at least two bin boundaries.")
    return np.asarray(values, dtype=float)


def read_npy_header_from_npz(npz_file: zipfile.ZipFile, member_name: str) -> tuple[tuple[int, ...], bool, np.dtype]:
    with npz_file.open(member_name, "r") as f:
        version = npformat.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = npformat.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = npformat.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy format version: {version}")
    return tuple(int(v) for v in shape), bool(fortran_order), np.dtype(dtype)


def npz_header_summary(path: str) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    with zipfile.ZipFile(path, mode="r") as zf:
        for member in zf.namelist():
            if not member.endswith(".npy"):
                continue
            key = Path(member).stem
            shape, fortran_order, dtype = read_npy_header_from_npz(zf, member)
            arrays[key] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": fortran_order,
                "estimated_uncompressed_bytes": int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
                if shape
                else int(dtype.itemsize),
            }
    return arrays


def feature_names(dataset: Any) -> list[str]:
    return [str(x) for x in np.asarray(dataset["spatial_node_feature_names"]).tolist()]


def feature_index(names: list[str], name: str) -> int | None:
    return names.index(name) if name in names else None


def scalar_or_list(value: Any, max_items: int = 30) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    flat = arr.reshape(-1)
    if flat.size <= max_items:
        return flat.tolist()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "head": flat[:max_items].tolist(),
    }


def time_indices(num_steps: int, max_time_steps: int, sample_time_steps: int, seed: int) -> np.ndarray:
    limit = num_steps if max_time_steps <= 0 else min(num_steps, max_time_steps)
    if sample_time_steps <= 0 or sample_time_steps >= limit:
        return np.arange(limit, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(limit, dtype=np.int64), size=sample_time_steps, replace=False))


def node_indices(num_nodes: int, sample_nodes: int, seed: int) -> np.ndarray:
    if sample_nodes <= 0 or sample_nodes >= num_nodes:
        return np.arange(num_nodes, dtype=np.int64)
    rng = np.random.default_rng(seed + 1009)
    return np.sort(rng.choice(np.arange(num_nodes, dtype=np.int64), size=sample_nodes, replace=False))


def finite_quantiles(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    qs = np.percentile(finite, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def histogram_rows(values: np.ndarray, bins: int, dataset_name: str, field: str) -> list[dict[str, Any]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    counts, edges = np.histogram(finite, bins=bins)
    return [
        {
            "dataset": dataset_name,
            "field": field,
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(count),
        }
        for i, count in enumerate(counts)
    ]


def load_targets_sample(
    dataset: Any,
    dataset_name: str,
    target_shape: tuple[int, int],
    target_nbytes: int,
    target_overrides: dict[str, str],
    time_idx: np.ndarray,
    node_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if dataset_name in target_overrides:
        target_path = target_overrides[dataset_name]
        targets = np.load(target_path, mmap_mode="r")
        sample = np.asarray(targets[np.ix_(time_idx, node_idx)], dtype=np.float64)
        return sample, {
            "target_source": target_path,
            "target_source_type": "npy_memmap",
            "target_sampling_skipped": False,
        }

    estimated_gb = target_nbytes / (1024.0**3)
    if estimated_gb > args.target_max_load_gb and not args.force_load_targets:
        return None, {
            "target_source": "npz:targets",
            "target_source_type": "compressed_npz",
            "target_sampling_skipped": True,
            "skip_reason": (
                f"estimated targets array is {estimated_gb:.2f} GiB, above "
                f"--target-max-load-gb={args.target_max_load_gb:.2f}; use --force-load-targets "
                "or --target-array name:path_to_targets.npy on the server"
            ),
        }

    targets = dataset["targets"]
    if tuple(targets.shape) != target_shape:
        raise ValueError(f"Header shape {target_shape} does not match loaded targets shape {targets.shape}")
    sample = np.asarray(targets[np.ix_(time_idx, node_idx)], dtype=np.float64)
    return sample, {
        "target_source": "npz:targets",
        "target_source_type": "compressed_npz",
        "target_sampling_skipped": False,
    }


def temporal_summary(unix_timestamps: np.ndarray) -> dict[str, Any]:
    diffs = np.diff(unix_timestamps.astype(np.int64))
    summary = {
        "num_timestamps": int(unix_timestamps.shape[0]),
        "start_unix": int(unix_timestamps[0]) if unix_timestamps.size else None,
        "end_unix": int(unix_timestamps[-1]) if unix_timestamps.size else None,
        "duration_days": float((unix_timestamps[-1] - unix_timestamps[0]) / 86400.0)
        if unix_timestamps.size > 1
        else None,
    }
    if diffs.size:
        unique, counts = np.unique(diffs, return_counts=True)
        summary.update(
            {
                "median_step_seconds": int(np.median(diffs)),
                "min_step_seconds": int(np.min(diffs)),
                "max_step_seconds": int(np.max(diffs)),
                "num_unique_steps": int(unique.shape[0]),
                "top_step_counts": [
                    {"step_seconds": int(step), "count": int(count)}
                    for step, count in sorted(
                        zip(unique, counts), key=lambda x: int(x[1]), reverse=True
                    )[:10]
                ],
            }
        )
    return summary


def metadata_summary(dataset: Any, headers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "keys": list(dataset.files),
        "array_headers": headers,
        "scalar_metadata": {},
    }
    for key in dataset.files:
        if key == "targets":
            continue
        arr = np.asarray(dataset[key])
        if arr.shape == () or key.endswith("_names") or key.startswith("subgraph_"):
            out["scalar_metadata"][key] = scalar_or_list(arr)
    return out


def canonical_fit_summary(
    dataset_name: str,
    path: str,
    num_steps: int,
    num_nodes: int,
    num_edges: int,
    split_sizes: dict[str, int],
    dataset: Any,
) -> dict[str, Any]:
    subgraph_keys = [key for key in dataset.files if key.startswith("subgraph_")]
    is_category_subset = "__category__" in Path(path).name or "__category__" in dataset_name
    expected = PAPER_CITY_TRAFFIC_M
    mismatches = {
        "nodes": {"observed": num_nodes, "expected": expected["nodes"], "matches": num_nodes == expected["nodes"]},
        "edges": {"observed": num_edges, "expected": expected["edges"], "matches": num_edges == expected["edges"]},
        "timestamps": {
            "observed": num_steps,
            "expected": expected["timestamps"],
            "matches": num_steps == expected["timestamps"],
        },
        "train_timestamps": {
            "observed": split_sizes["train"],
            "expected": expected["train_timestamps"],
            "matches": split_sizes["train"] == expected["train_timestamps"],
        },
        "val_timestamps": {
            "observed": split_sizes["val"],
            "expected": expected["val_timestamps"],
            "matches": split_sizes["val"] == expected["val_timestamps"],
        },
        "test_timestamps": {
            "observed": split_sizes["test"],
            "expected": expected["test_timestamps"],
            "matches": split_sizes["test"] == expected["test_timestamps"],
        },
    }
    warnings = []
    if subgraph_keys:
        warnings.append("file contains subgraph_* metadata; treat it as a derived subgraph, not canonical paper data")
    if is_category_subset:
        warnings.append("filename/name indicates a category-specific road-type subgraph")
    if num_nodes != expected["nodes"] or num_edges != expected["edges"]:
        warnings.append("observed node/edge counts differ from paper city-traffic-M full graph")
    return {
        "paper_expected_city_traffic_m": expected,
        "is_likely_category_subgraph": bool(is_category_subset or subgraph_keys),
        "subgraph_metadata_keys": subgraph_keys,
        "canonical_count_checks": mismatches,
        "warnings": warnings,
    }


def categorical_feature_rows(dataset: Any, dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = feature_names(dataset)
    features = np.asarray(dataset["spatial_node_features"])[0]
    for name in names:
        idx = feature_index(names, name)
        if idx is None:
            continue
        col = features[:, idx]
        unique, counts = np.unique(col, return_counts=True)
        if unique.shape[0] <= 40:
            for value, count in zip(unique, counts):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "feature": name,
                        "value": str(value),
                        "count": int(count),
                        "ratio": float(count / col.shape[0]) if col.shape[0] else 0.0,
                    }
                )
    return rows


def numeric_feature_rows(dataset: Any, dataset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = feature_names(dataset)
    features = np.asarray(dataset["spatial_node_features"])[0]
    for name in names:
        idx = feature_index(names, name)
        if idx is None:
            continue
        col = features[:, idx].astype(np.float64, copy=False)
        finite = col[np.isfinite(col)]
        unique_count = int(np.unique(finite).shape[0]) if finite.size else 0
        stats = finite_quantiles(col)
        stats.update(
            {
                "dataset": dataset_name,
                "feature": name,
                "unique_count": unique_count,
                "nan_ratio": float(np.isnan(col).mean()),
            }
        )
        rows.append(stats)
    return rows


def degree_summary(edges: np.ndarray, num_nodes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    total_deg = in_deg + out_deg

    summary = {
        "num_nodes": int(num_nodes),
        "num_edges": int(edges.shape[0]),
        "num_nodes_with_in_edges": int(np.sum(in_deg > 0)),
        "num_nodes_with_out_edges": int(np.sum(out_deg > 0)),
        "num_isolated_nodes": int(np.sum(total_deg == 0)),
        "in_degree": finite_quantiles(in_deg.astype(float)),
        "out_degree": finite_quantiles(out_deg.astype(float)),
        "total_degree": finite_quantiles(total_deg.astype(float)),
    }
    rows = []
    for name, deg in [("in", in_deg), ("out", out_deg), ("total", total_deg)]:
        unique, counts = np.unique(deg, return_counts=True)
        for value, count in zip(unique, counts):
            rows.append({"degree_type": name, "degree": int(value), "num_nodes": int(count)})
    return summary, rows


def edge_geometry_summary(dataset: Any, bins: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = feature_names(dataset)
    features = np.asarray(dataset["spatial_node_features"])[0]
    edges = np.asarray(dataset["edges"], dtype=np.int64)

    x0_i = feature_index(names, "x_coordinate_start")
    y0_i = feature_index(names, "y_coordinate_start")
    x1_i = feature_index(names, "x_coordinate_end")
    y1_i = feature_index(names, "y_coordinate_end")
    length_i = feature_index(names, "length")

    if None in (x0_i, y0_i, x1_i, y1_i):
        return {"has_coordinates": False}, []

    cx = (features[:, x0_i] + features[:, x1_i]) / 2.0
    cy = (features[:, y0_i] + features[:, y1_i]) / 2.0
    src = edges[:, 0]
    dst = edges[:, 1]
    centroid_dist = np.sqrt((cx[src] - cx[dst]) ** 2 + (cy[src] - cy[dst]) ** 2)

    summary = {
        "has_coordinates": True,
        "centroid_distance_m": finite_quantiles(centroid_dist.astype(float)),
    }
    if length_i is not None:
        lengths = features[:, length_i].astype(float)
        summary["node_length_m"] = finite_quantiles(lengths)
        summary["source_target_length_sum_m"] = finite_quantiles(lengths[src] + lengths[dst])

    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if math.isinf(hi):
            mask = centroid_dist >= lo
            label = f"[{lo:g}, inf)"
        else:
            mask = (centroid_dist >= lo) & (centroid_dist < hi)
            label = f"[{lo:g}, {hi:g})"
        rows.append(
            {
                "distance_bin_m": label,
                "num_edges": int(mask.sum()),
                "edge_ratio": float(mask.mean()) if mask.size else 0.0,
            }
        )
    return summary, rows


def missingness_rows(sample: np.ndarray, dataset_name: str) -> list[dict[str, Any]]:
    finite_mask = np.isfinite(sample)
    node_valid_ratio = finite_mask.mean(axis=0)
    time_valid_ratio = finite_mask.mean(axis=1)
    rows = []
    for group, values in [
        ("node_nan_ratio", 1.0 - node_valid_ratio),
        ("time_nan_ratio", 1.0 - time_valid_ratio),
        ("node_valid_ratio", node_valid_ratio),
        ("time_valid_ratio", time_valid_ratio),
    ]:
        stats = finite_quantiles(values.astype(float))
        stats.update({"dataset": dataset_name, "field": group})
        rows.append(stats)
    return rows


def target_sample_summary(sample: np.ndarray | None, sample_meta: dict[str, Any]) -> dict[str, Any]:
    if sample is None:
        return {**sample_meta, "finite_count": 0}
    values = sample.reshape(-1)
    stats = finite_quantiles(values)
    stats.update(
        {
            **sample_meta,
            "sample_values": int(values.shape[0]),
            "nan_count": int(np.isnan(values).sum()),
            "nan_ratio": float(np.isnan(values).mean()),
            "zero_count": int(np.sum(values == 0)),
            "zero_ratio": float(np.mean(values == 0)),
        }
    )
    return stats


def profile_one_dataset(
    name: str,
    path: str,
    output_dir: Path,
    args: argparse.Namespace,
    bins: np.ndarray,
    target_overrides: dict[str, str],
) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    headers = npz_header_summary(path)
    if "targets" not in headers:
        raise KeyError(f"Dataset file has no targets array: {path}")
    target_shape = tuple(int(v) for v in headers["targets"]["shape"])
    if len(target_shape) != 2:
        raise ValueError(f"Expected targets shape [time, node], got {target_shape}")
    num_steps, num_nodes = target_shape
    target_nbytes = int(headers["targets"]["estimated_uncompressed_bytes"])

    dataset = np.load(path, allow_pickle=True, mmap_mode="r")
    unix_timestamps = np.asarray(dataset["unix_timestamps"])
    edges = np.asarray(dataset["edges"], dtype=np.int64)
    split_sizes = {
        "train": int(np.asarray(dataset["train_timestamps"]).shape[0]),
        "val": int(np.asarray(dataset["val_timestamps"]).shape[0]),
        "test": int(np.asarray(dataset["test_timestamps"]).shape[0]),
    }

    time_idx = time_indices(num_steps, args.max_time_steps, args.sample_time_steps, args.seed)
    node_idx = node_indices(num_nodes, args.sample_nodes, args.seed)
    sample, sample_meta = load_targets_sample(
        dataset=dataset,
        dataset_name=name,
        target_shape=(num_steps, num_nodes),
        target_nbytes=target_nbytes,
        target_overrides=target_overrides,
        time_idx=time_idx,
        node_idx=node_idx,
        args=args,
    )
    sample_meta.update(
        {
            "sample_time_steps": int(time_idx.shape[0]),
            "sample_nodes": int(node_idx.shape[0]),
            "estimated_uncompressed_targets_gib": float(target_nbytes / (1024.0**3)),
        }
    )

    degree_stats, degree_rows = degree_summary(edges, num_nodes)
    edge_geo_stats, edge_distance_rows = edge_geometry_summary(dataset, bins)
    canonical = canonical_fit_summary(name, path, num_steps, num_nodes, edges.shape[0], split_sizes, dataset)
    target_stats = target_sample_summary(sample, sample_meta)

    summary = {
        "dataset": name,
        "path": path,
        "targets_shape": [int(num_steps), int(num_nodes)],
        "target_distribution_sample": target_stats,
        "temporal": temporal_summary(unix_timestamps),
        "metadata": metadata_summary(dataset, headers),
        "paper_context": canonical,
        "degree": degree_stats,
        "edge_geometry": edge_geo_stats,
        "split_sizes": split_sizes,
    }

    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(dataset_dir / "summary.json", summary)
    if sample is not None:
        values = sample.reshape(-1)
        write_csv(dataset_dir / "target_histogram.csv", histogram_rows(values, args.hist_bins, name, "targets"))
        write_csv(dataset_dir / "missingness_summary.csv", missingness_rows(sample, name))
    write_csv(dataset_dir / "degree_distribution.csv", [{**r, "dataset": name} for r in degree_rows])
    write_csv(dataset_dir / "edge_distance_distribution.csv", [{**r, "dataset": name} for r in edge_distance_rows])
    write_csv(dataset_dir / "numeric_spatial_feature_summary.csv", numeric_feature_rows(dataset, name))
    write_csv(dataset_dir / "low_cardinality_spatial_feature_counts.csv", categorical_feature_rows(dataset, name))

    return summary


def comparison_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        dist = s["target_distribution_sample"]
        temporal = s["temporal"]
        degree = s["degree"]
        edge_geo = s["edge_geometry"]
        paper_context = s["paper_context"]
        rows.append(
            {
                "dataset": s["dataset"],
                "path": s["path"],
                "num_timestamps": s["targets_shape"][0],
                "num_nodes": s["targets_shape"][1],
                "num_edges": degree["num_edges"],
                "is_likely_category_subgraph": paper_context["is_likely_category_subgraph"],
                "paper_warnings": " | ".join(paper_context["warnings"]),
                "duration_days": temporal.get("duration_days"),
                "median_step_seconds": temporal.get("median_step_seconds"),
                "target_sampling_skipped": dist.get("target_sampling_skipped"),
                "target_mean": dist.get("mean"),
                "target_std": dist.get("std"),
                "target_p50": dist.get("p50"),
                "target_p95": dist.get("p95"),
                "target_nan_ratio": dist.get("nan_ratio"),
                "target_zero_ratio": dist.get("zero_ratio"),
                "isolated_nodes": degree.get("num_isolated_nodes"),
                "median_total_degree": degree["total_degree"].get("p50"),
                "edge_centroid_distance_p50_m": edge_geo.get("centroid_distance_m", {}).get("p50")
                if edge_geo.get("has_coordinates")
                else None,
                "edge_centroid_distance_p95_m": edge_geo.get("centroid_distance_m", {}).get("p95")
                if edge_geo.get("has_coordinates")
                else None,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bins = parse_bins(args.distance_bins_m)
    target_overrides = parse_named_paths(args.target_array)
    specs = dataset_specs(args)
    if not specs:
        raise FileNotFoundError("No dataset specs were provided and no default local subgraph files exist.")

    summaries = []
    for name, path in specs:
        print(f"[profile] {name}: {path}", flush=True)
        summaries.append(profile_one_dataset(name, path, out, args, bins, target_overrides))

    rows = comparison_rows(summaries)
    write_json(out / "all_dataset_profile_summary.json", summaries)
    write_csv(out / "dataset_comparison.csv", rows)

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Profile outputs written to: {out}")


if __name__ == "__main__":
    main()

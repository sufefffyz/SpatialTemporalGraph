#!/usr/bin/env python3
"""Delay-effect audit for LargeST-style SD/GLA/GBA 5-minute datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_CANDIDATES = {
    "SD": [
        "BasicTS/datasets/SD_5min_full",
        "BasicTS/datasets/SD_5min",
    ],
    "GLA": [
        "BasicTS/datasets/GLA_5min_full",
        "BasicTS/datasets/GLA_5min",
    ],
    "GBA": [
        "BasicTS/datasets/GBA_5min_full",
        "BasicTS/datasets/GBA_5min",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute lagged-correlation delay effects on SD/GLA/GBA 5-minute datasets."
    )
    parser.add_argument("--output-dir", default="delay_selective/outputs/largest_5min_delay_audit")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as NAME:PATH. Can be repeated. Defaults to SD/GLA/GBA 5min candidates.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-file", default="split_indices_full.npz")
    parser.add_argument("--max-time-steps", type=int, default=20160, help="0 means use the whole selected split.")
    parser.add_argument("--max-edges", type=int, default=20000, help="0 means score all eligible graph edges.")
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum lag in 5-minute steps.")
    parser.add_argument(
        "--graph-variants",
        nargs="+",
        default=["distthre"],
        choices=["distthre", "physical_dir", "physical_bidir"],
        help=(
            "Graph variants to audit. Default is distthre, the LargeST built-in "
            "road-network distance graph."
        ),
    )
    parser.add_argument("--residualize", default="time_of_day", choices=["none", "mean", "time_of_day"])
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.80,
        help=(
            "Minimum correlation confidence for accepting a non-zero delay. "
            "The earlier exploratory default was 0.20; delay-selective runs "
            "use a stricter 0.80 threshold."
        ),
    )
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--min-edge-std", type=float, default=1e-6)
    parser.add_argument("--score-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distance-bins-km",
        default="0,1,2,5,10,20,50,100,inf",
        help="Comma-separated edge distance bins in kilometers when metadata has Lat/Lng.",
    )
    parser.add_argument(
        "--allow-non-5min",
        action="store_true",
        help="Do not reject datasets whose desc.json frequency is not 5 minutes.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any default/requested dataset path is missing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_dataset_specs(items: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for spec in items:
        if ":" not in spec:
            raise ValueError(f"--dataset expects NAME:PATH, got {spec!r}")
        name, path = spec.split(":", 1)
        specs[name.strip().upper()] = Path(path).expanduser()
    return specs


def discover_default_datasets(require_all: bool) -> tuple[dict[str, Path], list[dict[str, str]]]:
    specs: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []
    for name, candidates in DEFAULT_DATASET_CANDIDATES.items():
        found = None
        for raw in candidates:
            path = Path(raw)
            if path.exists():
                found = path
                break
        if found is None:
            skipped.append(
                {
                    "dataset": name,
                    "reason": "missing_5min_dataset_dir",
                    "checked": ";".join(candidates),
                }
            )
            if require_all:
                raise FileNotFoundError(f"{name} 5min dataset not found in: {candidates}")
        else:
            specs[name] = found
    return specs, skipped


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def unwrap_adj(payload: Any) -> np.ndarray:
    if isinstance(payload, tuple) and len(payload) == 3:
        payload = payload[2]
    elif isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    adj = np.asarray(payload, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adj.shape}")
    adj = adj.copy()
    np.fill_diagonal(adj, 0.0)
    return adj


def load_graph_variants(dataset_dir: Path, requested_variants: list[str]) -> dict[str, np.ndarray]:
    candidates = {
        "distthre": [
            dataset_dir / "adj_mx_largeST_original.pkl",
            dataset_dir / "adj_mx.pkl",
        ],
        "physical_dir": [
            dataset_dir / "adj_mx_physical_directed.pkl",
            dataset_dir / "adj_mx_physical_forward.pkl",
        ],
    }
    graphs: dict[str, np.ndarray] = {}
    requested = set(requested_variants)
    for name, paths in candidates.items():
        if name not in requested:
            continue
        for path in paths:
            if path.exists():
                graphs[name] = unwrap_adj(read_pickle(path))
                break
    if "physical_bidir" in requested and "physical_dir" not in graphs:
        for path in candidates["physical_dir"]:
            if path.exists():
                graphs["physical_dir"] = unwrap_adj(read_pickle(path))
                break
    if "physical_bidir" in requested and "physical_dir" in graphs:
        graphs["physical_bidir"] = np.maximum(graphs["physical_dir"], graphs["physical_dir"].T)
    if "physical_dir" not in requested:
        graphs.pop("physical_dir", None)
    if not graphs:
        raise FileNotFoundError(
            f"No requested adjacency variants {requested_variants} found in {dataset_dir}"
        )
    return graphs


def select_time_indices(desc: dict[str, Any], dataset_dir: Path, split_file: str, split: str) -> np.ndarray:
    num_time_steps = int(desc["num_time_steps"])
    if split == "all":
        return np.arange(num_time_steps, dtype=np.int64)

    split_path = dataset_dir / split_file
    if split_path.exists():
        splits = np.load(split_path)
        key = f"{split}_idx"
        if key not in splits:
            raise KeyError(f"{split_path} does not contain {key}")
        return np.asarray(splits[key], dtype=np.int64)

    ratios = desc.get("regular_settings", {}).get("TRAIN_VAL_TEST_RATIO", [0.6, 0.2, 0.2])
    train_len = int(num_time_steps * float(ratios[0]))
    val_len = int(num_time_steps * float(ratios[1]))
    if split == "train":
        return np.arange(0, train_len, dtype=np.int64)
    if split == "val":
        return np.arange(train_len, train_len + val_len, dtype=np.int64)
    if split == "test":
        return np.arange(train_len + val_len, num_time_steps, dtype=np.int64)
    raise ValueError(f"Unsupported split: {split}")


def load_flow_matrix(dataset_dir: Path, desc: dict[str, Any], time_indices: np.ndarray, max_time_steps: int) -> np.ndarray:
    shape = tuple(int(x) for x in desc["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected BasicTS data shape [T,N,C], got {shape}")
    if max_time_steps > 0:
        time_indices = time_indices[: min(max_time_steps, time_indices.shape[0])]
    data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    return np.asarray(data[time_indices, :, 0], dtype=np.float32)


def residualize_matrix(x: np.ndarray, mode: str, steps_per_day: int) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if mode == "none":
        return x
    if mode == "mean":
        return x - np.nanmean(x, axis=0, keepdims=True)
    if mode != "time_of_day":
        raise ValueError(f"Unknown residualization mode: {mode}")

    slots = np.arange(x.shape[0], dtype=np.int64) % max(1, steps_per_day)
    baseline = np.full((max(1, steps_per_day), x.shape[1]), np.nan, dtype=np.float32)
    global_mean = np.nanmean(x, axis=0)
    for slot in np.unique(slots):
        baseline[slot] = np.nanmean(x[slots == slot], axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, global_mean[None, :])
    return x - baseline[slots]


def corr_columns(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(a) & np.isfinite(b)
    count = valid.sum(axis=0).astype(np.float64)
    safe_count = np.maximum(count, 1.0)

    a0 = np.where(valid, a, 0.0).astype(np.float64, copy=False)
    b0 = np.where(valid, b, 0.0).astype(np.float64, copy=False)
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


def select_edges(
    adj: np.ndarray,
    x: np.ndarray,
    max_edges: int,
    min_edge_std: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    edges = np.argwhere(adj > 0).astype(np.int64)
    if edges.size == 0:
        return edges, 0
    node_std = np.nanstd(x, axis=0)
    eligible_mask = (
        np.isfinite(node_std[edges[:, 0]])
        & np.isfinite(node_std[edges[:, 1]])
        & (node_std[edges[:, 0]] > min_edge_std)
        & (node_std[edges[:, 1]] > min_edge_std)
    )
    eligible = edges[eligible_mask]
    num_eligible = int(eligible.shape[0])
    if max_edges > 0 and eligible.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(np.arange(eligible.shape[0]), size=max_edges, replace=False))
        eligible = eligible[chosen]
    return eligible, num_eligible


def compute_lag_scores(
    x: np.ndarray,
    edges: np.ndarray,
    max_lag: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr_by_lag = np.full((max_lag + 1, edges.shape[0]), np.nan, dtype=np.float64)
    count_by_lag = np.zeros((max_lag + 1, edges.shape[0]), dtype=np.float64)
    chunk_size = max(1, chunk_size)
    for start in range(0, edges.shape[0], chunk_size):
        end = min(edges.shape[0], start + chunk_size)
        src_idx = edges[start:end, 0]
        dst_idx = edges[start:end, 1]
        src = x[:, src_idx]
        dst = x[:, dst_idx]
        for lag in range(max_lag + 1):
            if lag == 0:
                a = src
                b = dst
            else:
                a = src[:-lag]
                b = dst[lag:]
            corr, count = corr_columns(a, b)
            corr_by_lag[lag, start:end] = corr
            count_by_lag[lag, start:end] = count
    return corr_by_lag, count_by_lag


def parse_bins(spec: str) -> np.ndarray:
    values = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        values.append(float("inf") if item in {"inf", "infinity"} else float(item))
    if len(values) < 2:
        raise ValueError("Need at least two bin boundaries.")
    return np.asarray(values, dtype=float)


def load_lat_lng(dataset_dir: Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray] | None:
    meta_path = dataset_dir / "meta.csv"
    if not meta_path.exists():
        return None
    with meta_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = {name.lower(): name for name in (reader.fieldnames or [])}
        lat_key = fields.get("lat") or fields.get("latitude")
        lng_key = fields.get("lng") or fields.get("lon") or fields.get("longitude")
        if lat_key is None or lng_key is None:
            return None
        lat = []
        lng = []
        for row in reader:
            lat.append(float(row[lat_key]))
            lng.append(float(row[lng_key]))
    if len(lat) != num_nodes:
        return None
    return np.asarray(lat, dtype=np.float64), np.asarray(lng, dtype=np.float64)


def edge_distances_km(lat_lng: tuple[np.ndarray, np.ndarray] | None, edges: np.ndarray) -> np.ndarray | None:
    if lat_lng is None:
        return None
    lat, lng = lat_lng
    src = edges[:, 0]
    dst = edges[:, 1]
    mean_lat = np.deg2rad((lat[src] + lat[dst]) / 2.0)
    dx = (lng[dst] - lng[src]) * 111.320 * np.cos(mean_lat)
    dy = (lat[dst] - lat[src]) * 110.540
    return np.sqrt(dx * dx + dy * dy)


def summarize_distance_bins(
    distances_km: np.ndarray | None,
    best_lags: np.ndarray,
    improvements: np.ndarray,
    best_corrs: np.ndarray,
    bins: np.ndarray,
    min_corr: float,
    min_improvement: float,
) -> list[dict[str, Any]]:
    if distances_km is None:
        return []
    rows: list[dict[str, Any]] = []
    high_conf_nonzero = (best_lags > 0) & (best_corrs >= min_corr) & (improvements >= min_improvement)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if math.isinf(hi):
            mask = distances_km >= lo
            label = f"[{lo:g}, inf)"
        else:
            mask = (distances_km >= lo) & (distances_km < hi)
            label = f"[{lo:g}, {hi:g})"
        n = int(mask.sum())
        rows.append(
            {
                "distance_bin_km": label,
                "num_edges": n,
                "near_zero_ratio": float(np.mean(best_lags[mask] == 0)) if n else None,
                "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero[mask])) if n else None,
                "median_best_lag_steps": float(np.nanmedian(best_lags[mask])) if n else None,
                "median_best_lag_minutes": float(5 * np.nanmedian(best_lags[mask])) if n else None,
                "median_corr_improvement": float(np.nanmedian(improvements[mask])) if n else None,
            }
        )
    return rows


def maybe_plot_dataset(output_dir: Path, edge_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs = sorted({row["graph"] for row in edge_rows})

    fig, axes = plt.subplots(1, max(1, len(graphs)), figsize=(4.0 * max(1, len(graphs)), 3.0), squeeze=False)
    for ax, graph in zip(axes[0], graphs):
        lags = [int(row["best_lag_steps"]) for row in edge_rows if row["graph"] == graph]
        ax.hist(lags, bins=np.arange(max(lags) + 2) - 0.5, edgecolor="white", color="#4E79A7")
        ax.set_title(graph)
        ax.set_xlabel("Best lag (5-min steps)")
        ax.set_ylabel("Edges")
    fig.tight_layout()
    path = output_dir / "best_lag_hist.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(str(path))

    labels = [row["graph"] for row in summary_rows]
    high_conf = [row["high_conf_nonzero_ratio"] for row in summary_rows]
    near_zero = [row["near_zero_ratio"] for row in summary_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.0 * len(labels)), 3.0))
    ax.bar(x - 0.18, near_zero, width=0.36, label="Best lag = 0", color="#59A14F")
    ax.bar(x + 0.18, high_conf, width=0.36, label="High-conf nonzero", color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Edge ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "delay_effect_ratios.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(str(path))
    return made


def audit_one_dataset(
    name: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    output_root: Path,
    bins_km: np.ndarray,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    desc = load_json(dataset_dir / "desc.json")
    frequency = int(desc.get("frequency (minutes)", -1))
    if frequency != 5 and not args.allow_non_5min:
        raise ValueError(
            f"{name} frequency is {frequency} minutes in {dataset_dir / 'desc.json'}, "
            "but this audit is restricted to 5-minute versions."
        )
    steps_per_day = max(1, int(round(1440 / max(1, frequency))))
    time_indices = select_time_indices(desc, dataset_dir, args.split_file, args.split)
    flow = load_flow_matrix(dataset_dir, desc, time_indices, args.max_time_steps)
    flow = residualize_matrix(flow, args.residualize, steps_per_day)
    graphs = load_graph_variants(dataset_dir, args.graph_variants)
    lat_lng = load_lat_lng(dataset_dir, int(desc["num_nodes"]))

    dataset_out = output_root / name
    all_edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []

    for graph_name, adj in graphs.items():
        edges, num_eligible_edges = select_edges(
            adj=adj,
            x=flow,
            max_edges=args.max_edges,
            min_edge_std=args.min_edge_std,
            seed=args.seed,
        )
        if edges.shape[0] == 0:
            summary_rows.append(
                {
                    "dataset": name,
                    "graph": graph_name,
                    "num_graph_edges": int((adj > 0).sum()),
                    "num_eligible_edges": int(num_eligible_edges),
                    "num_edges_scored": 0,
                    "error": "no eligible nonconstant edges",
                }
            )
            continue

        corr_by_lag, count_by_lag = compute_lag_scores(
            x=flow,
            edges=edges,
            max_lag=args.max_lag,
            chunk_size=args.score_chunk_size,
        )
        has_any_corr = np.isfinite(corr_by_lag).any(axis=0)
        corr_for_argmax = np.where(np.isfinite(corr_by_lag), corr_by_lag, -np.inf)
        best_lags = np.argmax(corr_for_argmax, axis=0)
        best_lags = np.where(has_any_corr, best_lags, 0)
        best_corrs = corr_by_lag[best_lags, np.arange(edges.shape[0])]
        zero_corrs = corr_by_lag[0]
        improvements = best_corrs - zero_corrs
        high_conf_nonzero = (
            (best_lags > 0)
            & (best_corrs >= args.min_corr)
            & (improvements >= args.min_improvement)
        )
        distances_km = edge_distances_km(lat_lng, edges)

        mean_corr_by_lag = [float(v) for v in np.nanmean(corr_by_lag, axis=1)]
        summary_rows.append(
            {
                "dataset": name,
                "dataset_dir": str(dataset_dir),
                "graph": graph_name,
                "frequency_minutes": frequency,
                "split": args.split,
                "num_time_steps_used": int(flow.shape[0]),
                "num_nodes": int(flow.shape[1]),
                "num_graph_edges": int((adj > 0).sum()),
                "num_eligible_edges": int(num_eligible_edges),
                "num_edges_scored": int(edges.shape[0]),
                "max_lag_steps": int(args.max_lag),
                "max_lag_minutes": int(args.max_lag * frequency),
                "residualize": args.residualize,
                "near_zero_ratio": float(np.mean(best_lags == 0)),
                "nonzero_ratio": float(np.mean(best_lags > 0)),
                "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero)),
                "median_best_lag_steps": float(np.nanmedian(best_lags)),
                "median_best_lag_minutes": float(frequency * np.nanmedian(best_lags)),
                "median_zero_lag_corr": float(np.nanmedian(zero_corrs)),
                "median_best_lag_corr": float(np.nanmedian(best_corrs)),
                "median_corr_improvement": float(np.nanmedian(improvements)),
                "mean_corr_by_lag": json.dumps(mean_corr_by_lag),
            }
        )

        graph_distance_rows = summarize_distance_bins(
            distances_km=distances_km,
            best_lags=best_lags,
            improvements=improvements,
            best_corrs=best_corrs,
            bins=bins_km,
            min_corr=args.min_corr,
            min_improvement=args.min_improvement,
        )
        for row in graph_distance_rows:
            row.update({"dataset": name, "graph": graph_name})
        distance_rows.extend(graph_distance_rows)

        for i, (src, dst) in enumerate(edges):
            row = {
                "dataset": name,
                "graph": graph_name,
                "source_index": int(src),
                "target_index": int(dst),
                "edge_distance_km": float(distances_km[i]) if distances_km is not None else None,
                "best_lag_steps": int(best_lags[i]),
                "best_lag_minutes": int(best_lags[i] * frequency),
                "zero_lag_corr": float(zero_corrs[i]),
                "best_lag_corr": float(best_corrs[i]),
                "corr_improvement": float(improvements[i]),
                "high_conf_nonzero_delay": bool(high_conf_nonzero[i]),
                "valid_pairs_at_best_lag": int(count_by_lag[best_lags[i], i]),
            }
            for lag in range(args.max_lag + 1):
                row[f"corr_lag_{lag}"] = float(corr_by_lag[lag, i])
            all_edge_rows.append(row)

    write_csv(dataset_out / "edge_delay_scores.csv", all_edge_rows)
    write_csv(dataset_out / "summary.csv", summary_rows)
    write_csv(dataset_out / "distance_bin_summary.csv", distance_rows)
    plots = maybe_plot_dataset(dataset_out, all_edge_rows, summary_rows)
    dataset_summary = {
        "dataset": name,
        "dataset_dir": str(dataset_dir),
        "desc": desc,
        "summary_rows": summary_rows,
        "distance_bin_rows": distance_rows,
        "plots": plots,
    }
    write_json(dataset_out / "summary.json", dataset_summary)
    return dataset_summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    bins_km = parse_bins(args.distance_bins_km)

    if args.dataset:
        dataset_specs = parse_dataset_specs(args.dataset)
        skipped: list[dict[str, str]] = []
        if args.require_all:
            for name, path in dataset_specs.items():
                if not path.exists():
                    raise FileNotFoundError(f"{name} dataset path does not exist: {path}")
    else:
        dataset_specs, skipped = discover_default_datasets(require_all=args.require_all)

    results = []
    errors = []
    for name, path in dataset_specs.items():
        if not path.exists():
            item = {"dataset": name, "reason": "path_missing", "checked": str(path)}
            skipped.append(item)
            if args.require_all:
                raise FileNotFoundError(item)
            continue
        print(f"[delay-audit] {name}: {path}", flush=True)
        try:
            results.append(audit_one_dataset(name, path, args, output_root, bins_km))
        except Exception as exc:
            errors.append({"dataset": name, "path": str(path), "error": repr(exc)})
            if args.require_all:
                raise

    combined_summary_rows: list[dict[str, Any]] = []
    combined_distance_rows: list[dict[str, Any]] = []
    for result in results:
        combined_summary_rows.extend(result["summary_rows"])
        combined_distance_rows.extend(result["distance_bin_rows"])
    write_csv(output_root / "all_summary.csv", combined_summary_rows)
    write_csv(output_root / "all_distance_bin_summary.csv", combined_distance_rows)
    write_json(
        output_root / "run_summary.json",
        {
            "args": vars(args),
            "num_datasets_completed": len(results),
            "completed_datasets": [result["dataset"] for result in results],
            "skipped": skipped,
            "errors": errors,
            "output_dir": str(output_root),
        },
    )
    print(
        json.dumps(
            {
                "completed_datasets": [result["dataset"] for result in results],
                "skipped": skipped,
                "errors": errors,
                "output_dir": str(output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

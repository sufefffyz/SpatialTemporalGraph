#!/usr/bin/env python3
"""Build Gaussian-kernel global-threshold graphs from distance matrices.

This script turns a precomputed OSRM distance matrix into BasicTS-compatible
adjacency matrices. It keeps the graph construction independent from the model
code so the same graph can be reused by GWNet, DCRNN, STGCN, or other BasicTS
baselines.

The main setting is the LargeST-style global threshold:

    S_ij = exp(-(D_ij / sigma)^2)

Then a global score threshold is chosen by quantile so the off-diagonal average
degree matches a target degree. By default, target degrees are beta multiples of
the reference graph average degree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_csv_floats(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset label, e.g. SD.")
    parser.add_argument("--distance", required=True, type=Path, help="OSRM distance matrix .npy in meters.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for graph outputs.")
    parser.add_argument("--reference-adj", type=Path, help="Reference adjacency, e.g. LargeST SD adj_mx.pkl.")
    parser.add_argument("--node-ids", type=Path, help="Optional node ID CSV from build_distance_matrices.py.")
    parser.add_argument(
        "--degree-ratios",
        default="0.25,0.5,1.0,1.5,2.0",
        help="Beta values multiplied by the reference graph average degree.",
    )
    parser.add_argument(
        "--target-degrees",
        default=None,
        help="Comma-separated absolute target average degrees. Appended after degree ratios.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian sigma in meters. Overrides --sigma-mode when provided.",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=("reference-median", "reference-trimmed-mean", "all-finite-std"),
        default=None,
        help="How to estimate sigma when --sigma is not provided.",
    )
    parser.add_argument(
        "--selection",
        choices=("exact", "threshold"),
        default="exact",
        help="exact keeps exactly the target number of ranked edges; threshold keeps all ties.",
    )
    parser.add_argument(
        "--candidate-distance",
        type=Path,
        default=None,
        help="Optional candidate distance matrix, usually straight-line distance.",
    )
    parser.add_argument(
        "--candidate-max-m",
        type=float,
        default=None,
        help="Only consider candidates whose --candidate-distance is <= this value.",
    )
    parser.add_argument("--no-self-loops", action="store_true", help="Do not set diagonal entries to 1.")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def dump_pickle(path: Path, obj: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite.")
    with path.open("wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_adjacency(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    obj = load_pickle(path)
    if isinstance(obj, (tuple, list)) and len(obj) >= 3:
        return np.asarray(obj[-1])
    return np.asarray(obj)


def load_node_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or "node_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a node_id column.")
        return [str(row["node_id"]) for row in reader]


def basicts_payload(adj: np.ndarray, node_ids: list[str] | None) -> Any:
    if node_ids is None:
        return adj
    if len(node_ids) != adj.shape[0]:
        raise ValueError(f"node_ids length {len(node_ids)} != adjacency size {adj.shape[0]}.")
    return (node_ids, {node_id: i for i, node_id in enumerate(node_ids)}, adj)


def offdiag_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def graph_stats(adj: np.ndarray) -> dict[str, Any]:
    n = adj.shape[0]
    mask = offdiag_mask(n)
    support = (adj != 0) & mask
    degrees = support.sum(axis=1)
    values = adj[support]
    stats: dict[str, Any] = {
        "num_nodes": int(n),
        "edge_count": int(support.sum()),
        "avg_out_degree": float(support.sum() / n),
        "density_no_diag": float(support.sum() / (n * (n - 1))),
        "isolated_rows": int((degrees == 0).sum()),
        "min_out_degree": int(degrees.min()) if degrees.size else 0,
        "median_out_degree": float(np.median(degrees)) if degrees.size else 0.0,
        "max_out_degree": int(degrees.max()) if degrees.size else 0,
    }
    if values.size:
        quantiles = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        stats["weight_quantiles"] = [float(item) for item in quantiles]
    return stats


def estimate_sigma(distance: np.ndarray, reference_adj: np.ndarray | None, mode: str | None, base_mask: np.ndarray) -> tuple[float, dict[str, Any]]:
    if mode is None:
        mode = "reference-median" if reference_adj is not None else "all-finite-std"

    if mode in {"reference-median", "reference-trimmed-mean"}:
        if reference_adj is None:
            raise ValueError(f"--sigma-mode {mode} requires --reference-adj.")
        if reference_adj.shape != distance.shape:
            raise ValueError(f"reference adjacency shape {reference_adj.shape} != distance shape {distance.shape}.")
        ref_mask = (reference_adj > 0) & offdiag_mask(distance.shape[0])
        weights = reference_adj[ref_mask].astype(np.float64)
        dists = distance[ref_mask].astype(np.float64)
        denom = np.sqrt(-np.log(weights))
        valid = np.isfinite(dists) & (dists > 0.0) & np.isfinite(denom) & (denom > 1e-12)
        sigmas = dists[valid] / denom[valid]
        if sigmas.size == 0:
            raise ValueError("Could not estimate sigma from reference graph and distance matrix.")
        if mode == "reference-median":
            sigma = float(np.median(sigmas))
        else:
            sigmas_sorted = np.sort(sigmas)
            lo = int(0.05 * sigmas_sorted.size)
            hi = int(0.95 * sigmas_sorted.size)
            sigma = float(np.mean(sigmas_sorted[lo:hi]))
        return sigma, {
            "mode": mode,
            "source_edge_count": int(ref_mask.sum()),
            "valid_edge_count": int(sigmas.size),
            "sigma_quantiles_m": [float(item) for item in np.percentile(sigmas, [0, 1, 5, 25, 50, 75, 95, 99, 100])],
        }

    values = distance[base_mask]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("No finite positive distances available for all-finite-std sigma.")
    sigma = float(np.std(values))
    return sigma, {"mode": mode, "finite_positive_count": int(values.size)}


def format_degree_tag(degree: float) -> str:
    rounded = f"{degree:.2f}".replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", rounded)


def build_graph_for_degree(
    scores: np.ndarray,
    available_flat: np.ndarray,
    target_degree: float,
    selection: str,
    self_loops: bool,
    dtype: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = scores.shape[0]
    available_count = int(available_flat.size)
    target_edges = int(round(target_degree * n))
    target_edges = max(0, min(target_edges, available_count))

    adj = np.zeros(scores.shape, dtype=dtype)
    if target_edges > 0:
        flat_scores = scores.ravel()[available_flat]
        order = np.argsort(flat_scores, kind="mergesort")[::-1]
        selected_available = available_flat[order[:target_edges]]
        threshold = float(flat_scores[order[target_edges - 1]])
        if selection == "threshold":
            selected_available = available_flat[flat_scores >= threshold]
        adj.ravel()[selected_available] = scores.ravel()[selected_available].astype(dtype, copy=False)
    else:
        threshold = math.inf

    if self_loops:
        np.fill_diagonal(adj, 1.0)

    support = (adj != 0) & offdiag_mask(n)
    actual_edges = int(support.sum())
    metadata = {
        "target_avg_degree": float(target_degree),
        "target_edge_count": int(target_edges),
        "actual_edge_count": actual_edges,
        "actual_avg_degree": float(actual_edges / n),
        "threshold_similarity": float(threshold),
        "threshold_distance_m": float(math.sqrt(-math.log(threshold)) * 1.0) if 0.0 < threshold <= 1.0 else None,
        "score_upper_quantile": float(1.0 - (target_edges / available_count)) if available_count else None,
    }
    return adj, metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distance = np.load(args.distance)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"distance matrix must be square, got {distance.shape}.")
    n = distance.shape[0]
    base_mask = offdiag_mask(n) & np.isfinite(distance) & (distance >= 0.0)

    if args.candidate_distance is not None or args.candidate_max_m is not None:
        if args.candidate_distance is None or args.candidate_max_m is None:
            raise ValueError("--candidate-distance and --candidate-max-m must be provided together.")
        candidate_distance = np.load(args.candidate_distance)
        if candidate_distance.shape != distance.shape:
            raise ValueError(f"candidate distance shape {candidate_distance.shape} != distance shape {distance.shape}.")
        base_mask &= np.isfinite(candidate_distance) & (candidate_distance <= args.candidate_max_m)

    reference_adj = load_adjacency(args.reference_adj)
    if reference_adj is not None and reference_adj.shape != distance.shape:
        raise ValueError(f"reference adjacency shape {reference_adj.shape} != distance shape {distance.shape}.")

    node_ids = load_node_ids(args.node_ids)
    if node_ids is not None and len(node_ids) != n:
        raise ValueError(f"node ID count {len(node_ids)} != distance size {n}.")

    reference_stats = graph_stats(reference_adj) if reference_adj is not None else None
    reference_degree = reference_stats["avg_out_degree"] if reference_stats else None

    target_degrees: list[tuple[str, float]] = []
    for ratio in parse_csv_floats(args.degree_ratios):
        if reference_degree is None:
            raise ValueError("--degree-ratios requires --reference-adj.")
        target_degrees.append((f"beta{ratio:.2f}".replace(".", "p"), ratio * float(reference_degree)))
    for degree in parse_csv_floats(args.target_degrees):
        target_degrees.append((f"deg{degree:.2f}".replace(".", "p"), degree))
    if not target_degrees:
        if reference_degree is None:
            raise ValueError("Provide --target-degrees or --reference-adj with --degree-ratios.")
        target_degrees.append(("beta1p00", float(reference_degree)))

    if args.sigma is not None:
        sigma = float(args.sigma)
        sigma_info = {"mode": "explicit", "sigma_m": sigma}
    else:
        sigma, sigma_info = estimate_sigma(distance, reference_adj, args.sigma_mode, base_mask)
        sigma_info["sigma_m"] = sigma
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError(f"Invalid sigma: {sigma}.")

    scores = np.zeros(distance.shape, dtype=np.float64)
    scores[base_mask] = np.exp(-np.square(distance[base_mask].astype(np.float64) / sigma))
    available_flat = np.flatnonzero(base_mask)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance": str(args.distance),
        "candidate_distance": str(args.candidate_distance) if args.candidate_distance else None,
        "candidate_max_m": args.candidate_max_m,
        "reference_adj": str(args.reference_adj) if args.reference_adj else None,
        "reference_stats": reference_stats,
        "sigma": sigma_info,
        "available_offdiag_candidates": int(available_flat.size),
        "selection": args.selection,
        "self_loops": not args.no_self_loops,
        "graphs": [],
    }

    for label, degree in target_degrees:
        adj, metadata = build_graph_for_degree(
            scores,
            available_flat,
            degree,
            args.selection,
            not args.no_self_loops,
            args.dtype,
        )
        degree_tag = format_degree_tag(metadata["actual_avg_degree"])
        out_name = f"{args.dataset}_osrm_gaussian_global_{label}_deg{degree_tag}.pkl"
        out_path = args.output_dir / out_name
        dump_pickle(out_path, basicts_payload(adj, node_ids), args.overwrite)
        metadata.update(
            {
                "label": label,
                "path": str(out_path),
                "stats": graph_stats(adj),
            }
        )
        # Convert the score threshold back to meters for readability.
        threshold = metadata["threshold_similarity"]
        if threshold is not None and 0.0 < threshold <= 1.0:
            metadata["threshold_distance_m"] = float(sigma * math.sqrt(-math.log(threshold)))
        summary["graphs"].append(metadata)

    summary_path = args.output_dir / f"{args.dataset}_osrm_gaussian_global_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists. Use --overwrite.")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interpretable three-step regional delay audit for LargeST 5-minute data.

The goal is deliberately diagnostic, not a full model:

1. Partition the existing local road graph into topology-contiguous groups.
2. Aggregate node-level traffic signals into group-level time series.
3. Re-run delay estimation on directed inter-group edges only.

This operationalizes the current delay-selective hypothesis: keep local
near-neighbor interactions synchronous, and only inspect delay at a coarser
regional scale where non-zero lags may be observable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_largest_5min_delay_audit import (
        DEFAULT_DATASET_CANDIDATES,
        discover_default_datasets,
        edge_distances_km,
        load_graph_variants,
        load_json,
        load_lat_lng,
        parse_bins,
        parse_dataset_specs,
        write_csv,
        write_json,
    )
    from .run_largest_5min_delay_multimethod_audit import (
        METHOD_CHOICES,
        WINDOW_CHOICES,
        get_window_specs,
        load_window_flow,
        nan_quantile,
        prepare_method_flow,
        score_delay,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_largest_5min_delay_audit import (
        DEFAULT_DATASET_CANDIDATES,
        discover_default_datasets,
        edge_distances_km,
        load_graph_variants,
        load_json,
        load_lat_lng,
        parse_bins,
        parse_dataset_specs,
        write_csv,
        write_json,
    )
    from run_largest_5min_delay_multimethod_audit import (
        METHOD_CHOICES,
        WINDOW_CHOICES,
        get_window_specs,
        load_window_flow,
        nan_quantile,
        prepare_method_flow,
        score_delay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build interpretable topology groups and audit inter-group delay on LargeST 5min data."
    )
    parser.add_argument(
        "--output-dir",
        default="delay_selective/outputs/largest_5min_interpretable_group_delay",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as NAME:PATH. Can be repeated. Defaults to discovered SD/GLA/GBA 5min bundles.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-file", default="split_indices_full.npz")
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=20160,
        help="Only used when --windows includes train_prefix.",
    )
    parser.add_argument(
        "--graph-variant",
        default="distthre",
        choices=["distthre", "physical_dir", "physical_bidir"],
        help="Graph used for local grouping and directed inter-group boundary edges.",
    )
    parser.add_argument(
        "--grouping",
        default="topology_bfs",
        choices=["topology_bfs", "louvain"],
        help="Interpretable grouping method. Louvain falls back to topology_bfs if networkx is unavailable.",
    )
    parser.add_argument("--target-group-size", type=int, default=32)
    parser.add_argument("--max-group-size", type=int, default=96)
    parser.add_argument("--louvain-resolution", type=float, default=1.0)
    parser.add_argument("--pooling", default="median", choices=["mean", "median"])
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["month", "week_daily"],
        choices=WINDOW_CHOICES,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mcc_5min_resid", "stdde_spline_fft_mcc", "lift_fft_abs"],
        choices=METHOD_CHOICES,
    )
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum lag in native 5-minute steps.")
    parser.add_argument("--interp-minutes", type=int, default=1)
    parser.add_argument("--residualize", default="time_of_day", choices=["none", "mean", "time_of_day"])
    parser.add_argument("--min-corr", type=float, default=0.80)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--min-edge-std", type=float, default=1e-6)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--month-start-day", type=int, default=0)
    parser.add_argument("--month-num-days", type=int, default=31)
    parser.add_argument("--week-start-day", type=int, default=0)
    parser.add_argument("--week-num-days", type=int, default=7)
    parser.add_argument(
        "--min-boundary-edges",
        type=int,
        default=1,
        help="Minimum original directed graph edges required to keep an inter-group edge.",
    )
    parser.add_argument(
        "--max-group-edges",
        type=int,
        default=0,
        help="0 means score all eligible directed inter-group edges.",
    )
    parser.add_argument("--stable-day-ratio", type=float, default=0.5)
    parser.add_argument("--lag-agreement-ratio", type=float, default=0.5)
    parser.add_argument(
        "--distance-bins-km",
        default="0,1,2,5,10,20,50,100,inf",
        help="Group centroid distance bins in kilometers.",
    )
    parser.add_argument("--allow-non-5min", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def compact_assignment(assignment: np.ndarray) -> np.ndarray:
    values = sorted(int(v) for v in np.unique(assignment))
    mapping = {value: idx for idx, value in enumerate(values)}
    return np.asarray([mapping[int(v)] for v in assignment], dtype=np.int64)


def adjacency_lists(sym_adj: np.ndarray) -> list[list[int]]:
    rows, cols = np.nonzero(sym_adj)
    neighbors: list[list[int]] = [[] for _ in range(sym_adj.shape[0])]
    for src, dst in zip(rows, cols):
        if src != dst:
            neighbors[int(src)].append(int(dst))
    degrees = np.asarray([len(x) for x in neighbors], dtype=np.int64)
    for node, items in enumerate(neighbors):
        items.sort(key=lambda v: (-degrees[v], v))
    return neighbors


def bfs_chunks(
    allowed_nodes: np.ndarray,
    neighbors: list[list[int]],
    target_size: int,
    degrees: np.ndarray,
) -> list[list[int]]:
    allowed = set(int(x) for x in allowed_nodes)
    chunks: list[list[int]] = []
    while allowed:
        seed = max(allowed, key=lambda n: (degrees[n], -n))
        queue: deque[int] = deque([seed])
        allowed.remove(seed)
        chunk = [seed]
        while queue and len(chunk) < target_size:
            cur = queue.popleft()
            for nb in neighbors[cur]:
                if nb in allowed:
                    allowed.remove(nb)
                    chunk.append(nb)
                    queue.append(nb)
                    if len(chunk) >= target_size:
                        break
        chunks.append(chunk)
    return chunks


def topology_bfs_groups(sym_adj: np.ndarray, target_group_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    neighbors = adjacency_lists(sym_adj)
    degrees = np.asarray([len(x) for x in neighbors], dtype=np.int64)
    nodes = np.arange(sym_adj.shape[0], dtype=np.int64)
    chunks = bfs_chunks(nodes, neighbors, max(1, target_group_size), degrees)
    assignment = np.empty(sym_adj.shape[0], dtype=np.int64)
    for gid, chunk in enumerate(chunks):
        assignment[np.asarray(chunk, dtype=np.int64)] = gid
    return compact_assignment(assignment), {
        "actual_grouping": "topology_bfs",
        "note": "Dependency-free contiguous BFS chunks on the symmetrized local graph.",
    }


def louvain_groups(
    adj: np.ndarray,
    sym_adj: np.ndarray,
    target_group_size: int,
    max_group_size: int,
    resolution: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import networkx as nx
    except Exception as exc:
        assignment, meta = topology_bfs_groups(sym_adj, target_group_size)
        meta["requested_grouping"] = "louvain"
        meta["fallback_reason"] = f"networkx_unavailable: {exc!r}"
        return assignment, meta

    if not hasattr(nx.community, "louvain_communities"):
        assignment, meta = topology_bfs_groups(sym_adj, target_group_size)
        meta["requested_grouping"] = "louvain"
        meta["fallback_reason"] = "networkx_has_no_louvain_communities"
        return assignment, meta

    graph = nx.Graph()
    graph.add_nodes_from(range(adj.shape[0]))
    weights = np.maximum(adj, adj.T)
    rows, cols = np.nonzero(np.triu(weights > 0, k=1))
    for src, dst in zip(rows, cols):
        graph.add_edge(int(src), int(dst), weight=float(weights[src, dst]))
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )

    neighbors = adjacency_lists(sym_adj)
    degrees = np.asarray([len(x) for x in neighbors], dtype=np.int64)
    chunks: list[list[int]] = []
    for community in communities:
        nodes = np.asarray(sorted(int(x) for x in community), dtype=np.int64)
        if max_group_size > 0 and nodes.shape[0] > max_group_size:
            chunks.extend(bfs_chunks(nodes, neighbors, max(1, target_group_size), degrees))
        else:
            chunks.append(nodes.tolist())

    assignment = np.empty(adj.shape[0], dtype=np.int64)
    for gid, chunk in enumerate(chunks):
        assignment[np.asarray(chunk, dtype=np.int64)] = gid
    return compact_assignment(assignment), {
        "actual_grouping": "louvain",
        "louvain_resolution": resolution,
        "oversized_communities_split_with": "topology_bfs",
    }


def build_groups(adj: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    sym_adj = np.maximum(adj, adj.T) > 0
    if args.grouping == "louvain":
        return louvain_groups(
            adj=adj,
            sym_adj=sym_adj,
            target_group_size=args.target_group_size,
            max_group_size=args.max_group_size,
            resolution=args.louvain_resolution,
            seed=args.seed,
        )
    return topology_bfs_groups(sym_adj=sym_adj, target_group_size=args.target_group_size)


def group_centroids(
    assignment: np.ndarray,
    lat_lng: tuple[np.ndarray, np.ndarray] | None,
    num_groups: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if lat_lng is None:
        return None
    lat, lng = lat_lng
    group_lat = np.full(num_groups, np.nan, dtype=np.float64)
    group_lng = np.full(num_groups, np.nan, dtype=np.float64)
    for gid in range(num_groups):
        mask = assignment == gid
        if mask.any():
            group_lat[gid] = float(np.nanmean(lat[mask]))
            group_lng[gid] = float(np.nanmean(lng[mask]))
    return group_lat, group_lng


def centroid_distances_km(
    centroids: tuple[np.ndarray, np.ndarray] | None,
    edges: np.ndarray,
) -> np.ndarray | None:
    if centroids is None:
        return None
    lat, lng = centroids
    src = edges[:, 0]
    dst = edges[:, 1]
    mean_lat = np.deg2rad((lat[src] + lat[dst]) / 2.0)
    dx = (lng[dst] - lng[src]) * 111.320 * np.cos(mean_lat)
    dy = (lat[dst] - lat[src]) * 110.540
    return np.sqrt(dx * dx + dy * dy)


def group_assignment_rows(
    assignment: np.ndarray,
    lat_lng: tuple[np.ndarray, np.ndarray] | None,
) -> list[dict[str, Any]]:
    rows = []
    lat = lng = None
    if lat_lng is not None:
        lat, lng = lat_lng
    for node, gid in enumerate(assignment):
        row = {"node_index": node, "group_id": int(gid)}
        if lat is not None and lng is not None:
            row["lat"] = float(lat[node])
            row["lng"] = float(lng[node])
        rows.append(row)
    return rows


def summarize_grouping(
    adj: np.ndarray,
    assignment: np.ndarray,
    centroids: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    num_groups = int(assignment.max()) + 1
    sizes = np.bincount(assignment, minlength=num_groups)
    edges = np.argwhere(adj > 0).astype(np.int64)
    src_group = assignment[edges[:, 0]]
    dst_group = assignment[edges[:, 1]]
    intra = src_group == dst_group
    out_boundary = np.bincount(src_group[~intra], minlength=num_groups)
    in_boundary = np.bincount(dst_group[~intra], minlength=num_groups)
    internal = np.bincount(src_group[intra], minlength=num_groups)

    lat = lng = None
    if centroids is not None:
        lat, lng = centroids

    rows: list[dict[str, Any]] = []
    for gid in range(num_groups):
        row = {
            "group_id": gid,
            "num_nodes": int(sizes[gid]),
            "internal_directed_edges": int(internal[gid]),
            "outgoing_boundary_edges": int(out_boundary[gid]),
            "incoming_boundary_edges": int(in_boundary[gid]),
        }
        if lat is not None and lng is not None:
            row["centroid_lat"] = float(lat[gid])
            row["centroid_lng"] = float(lng[gid])
        rows.append(row)

    summary = {
        "num_nodes": int(adj.shape[0]),
        "num_groups": num_groups,
        "num_graph_edges": int(edges.shape[0]),
        "intra_edge_ratio": float(np.mean(intra)) if edges.shape[0] else None,
        "inter_edge_ratio": float(np.mean(~intra)) if edges.shape[0] else None,
        "group_size_mean": float(np.mean(sizes)) if sizes.size else None,
        "group_size_median": float(np.median(sizes)) if sizes.size else None,
        "group_size_p90": float(np.quantile(sizes, 0.9)) if sizes.size else None,
        "group_size_max": int(np.max(sizes)) if sizes.size else None,
    }
    return summary, rows


def build_inter_group_edges(
    adj: np.ndarray,
    assignment: np.ndarray,
    lat_lng: tuple[np.ndarray, np.ndarray] | None,
    centroids: tuple[np.ndarray, np.ndarray] | None,
    min_boundary_edges: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    raw_edges = np.argwhere(adj > 0).astype(np.int64)
    src_group = assignment[raw_edges[:, 0]]
    dst_group = assignment[raw_edges[:, 1]]
    inter = src_group != dst_group
    raw_edges = raw_edges[inter]
    src_group = src_group[inter]
    dst_group = dst_group[inter]

    if raw_edges.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), [], {}

    num_groups = int(assignment.max()) + 1
    keys = src_group * num_groups + dst_group
    unique_keys, counts = np.unique(keys, return_counts=True)
    node_distances = edge_distances_km(lat_lng, raw_edges)

    boundary_distance_sum: dict[int, float] = defaultdict(float)
    if node_distances is not None:
        for key, dist in zip(keys, node_distances):
            if np.isfinite(dist):
                boundary_distance_sum[int(key)] += float(dist)

    group_edges = []
    rows = []
    meta: dict[tuple[int, int], dict[str, Any]] = {}
    candidate_edges = np.column_stack([unique_keys // num_groups, unique_keys % num_groups]).astype(np.int64)
    centroid_dist = centroid_distances_km(centroids, candidate_edges)

    count_lookup = {int(k): int(c) for k, c in zip(unique_keys, counts)}
    for idx, (key, count) in enumerate(zip(unique_keys, counts)):
        if int(count) < min_boundary_edges:
            continue
        src = int(key // num_groups)
        dst = int(key % num_groups)
        reverse_key = dst * num_groups + src
        mean_boundary_distance = None
        if node_distances is not None and int(count) > 0:
            mean_boundary_distance = boundary_distance_sum[int(key)] / float(count)
        centroid_distance = None
        if centroid_dist is not None and np.isfinite(centroid_dist[idx]):
            centroid_distance = float(centroid_dist[idx])
        item = {
            "source_group": src,
            "target_group": dst,
            "boundary_edge_count": int(count),
            "reverse_boundary_edge_count": int(count_lookup.get(reverse_key, 0)),
            "mean_boundary_edge_distance_km": mean_boundary_distance,
            "centroid_distance_km": centroid_distance,
        }
        group_edges.append([src, dst])
        rows.append(item)
        meta[(src, dst)] = item
    return np.asarray(group_edges, dtype=np.int64), rows, meta


def aggregate_group_signal(flow: np.ndarray, assignment: np.ndarray, pooling: str) -> np.ndarray:
    num_groups = int(assignment.max()) + 1
    out = np.full((flow.shape[0], num_groups), np.nan, dtype=np.float32)
    for gid in range(num_groups):
        cols = np.flatnonzero(assignment == gid)
        if cols.size == 0:
            continue
        block = flow[:, cols]
        if pooling == "mean":
            out[:, gid] = np.nanmean(block, axis=1).astype(np.float32)
        else:
            out[:, gid] = np.nanmedian(block, axis=1).astype(np.float32)
    return out


def sample_group_edges(
    group_edges: np.ndarray,
    group_flow: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    if group_edges.shape[0] == 0:
        return group_edges, 0
    group_std = np.nanstd(group_flow, axis=0)
    eligible_mask = (
        np.isfinite(group_std[group_edges[:, 0]])
        & np.isfinite(group_std[group_edges[:, 1]])
        & (group_std[group_edges[:, 0]] > args.min_edge_std)
        & (group_std[group_edges[:, 1]] > args.min_edge_std)
    )
    eligible = group_edges[eligible_mask]
    num_eligible = int(eligible.shape[0])
    if args.max_group_edges > 0 and eligible.shape[0] > args.max_group_edges:
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(np.arange(eligible.shape[0]), size=args.max_group_edges, replace=False))
        eligible = eligible[chosen]
    return eligible, num_eligible


def make_delay_rows(
    dataset: str,
    graph: str,
    window_spec: dict[str, Any],
    method: str,
    method_minutes: int,
    group_edges: np.ndarray,
    edge_meta: dict[tuple[int, int], dict[str, Any]],
    corr_by_lag: np.ndarray,
    count_by_lag: np.ndarray,
    best_lags: np.ndarray,
    best_corrs: np.ndarray,
    best_scores: np.ndarray,
    zero_corrs: np.ndarray,
    zero_scores: np.ndarray,
    improvements: np.ndarray,
    effective_lags: np.ndarray,
    corr_filtered: np.ndarray,
    low_improvement_nonzero: np.ndarray,
    high_conf_nonzero: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (src, dst) in enumerate(group_edges):
        meta = edge_meta.get((int(src), int(dst)), {})
        best_lag = int(best_lags[i])
        effective = float(effective_lags[i])
        row = {
            "dataset": dataset,
            "graph": graph,
            "window": window_spec["window"],
            "window_label": window_spec["label"],
            "day_offset": window_spec.get("day_offset"),
            "day_in_window": window_spec.get("day_in_window"),
            "weekday": window_spec.get("weekday"),
            "method": method,
            "source_group": int(src),
            "target_group": int(dst),
            "boundary_edge_count": meta.get("boundary_edge_count"),
            "reverse_boundary_edge_count": meta.get("reverse_boundary_edge_count"),
            "mean_boundary_edge_distance_km": meta.get("mean_boundary_edge_distance_km"),
            "centroid_distance_km": meta.get("centroid_distance_km"),
            "best_lag_steps": best_lag,
            "best_lag_minutes": int(best_lag * method_minutes),
            "effective_lag_steps": effective if np.isfinite(effective) else float("nan"),
            "effective_lag_minutes": effective * method_minutes if np.isfinite(effective) else float("nan"),
            "zero_lag_corr": float(zero_corrs[i]) if np.isfinite(zero_corrs[i]) else None,
            "zero_lag_score": float(zero_scores[i]) if np.isfinite(zero_scores[i]) else None,
            "best_lag_corr": float(best_corrs[i]) if np.isfinite(best_corrs[i]) else None,
            "best_lag_score": float(best_scores[i]) if np.isfinite(best_scores[i]) else None,
            "corr_improvement": float(improvements[i]) if np.isfinite(improvements[i]) else None,
            "corr_filtered_delay": bool(corr_filtered[i]),
            "low_improvement_nonzero_delay": bool(low_improvement_nonzero[i]),
            "high_conf_nonzero_delay": bool(high_conf_nonzero[i]),
            "valid_pairs_at_best_lag": int(count_by_lag[best_lag, i]),
        }
        rows.append(row)
    return rows


def summarize_group_delay(
    dataset: str,
    dataset_dir: Path,
    graph: str,
    window_spec: dict[str, Any],
    method: str,
    method_minutes: int,
    raw_group_flow: np.ndarray,
    group_edges_all: np.ndarray,
    group_edges_scored: np.ndarray,
    num_eligible_edges: int,
    score_mode: str,
    best_lags: np.ndarray,
    best_corrs: np.ndarray,
    best_scores: np.ndarray,
    zero_corrs: np.ndarray,
    zero_scores: np.ndarray,
    improvements: np.ndarray,
    effective_lags: np.ndarray,
    corr_filtered: np.ndarray,
    low_improvement_nonzero: np.ndarray,
    high_conf_nonzero: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    effective_nan = ~np.isfinite(effective_lags)
    effective_zero = np.isfinite(effective_lags) & (effective_lags == 0)
    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "graph": graph,
        "window": window_spec["window"],
        "window_label": window_spec["label"],
        "day_offset": window_spec.get("day_offset"),
        "day_in_window": window_spec.get("day_in_window"),
        "weekday": window_spec.get("weekday"),
        "method": method,
        "score_mode": score_mode,
        "method_step_minutes": int(method_minutes),
        "num_time_steps_used": int(raw_group_flow.shape[0]),
        "num_groups": int(raw_group_flow.shape[1]),
        "num_group_edges_total": int(group_edges_all.shape[0]),
        "num_group_edges_eligible": int(num_eligible_edges),
        "num_group_edges_scored": int(group_edges_scored.shape[0]),
        "min_corr": float(args.min_corr),
        "min_improvement": float(args.min_improvement),
        "raw_best_zero_ratio": float(np.mean(best_lags == 0)) if group_edges_scored.shape[0] else None,
        "corr_filtered_ratio": float(np.mean(corr_filtered)) if group_edges_scored.shape[0] else None,
        "effective_nan_ratio": float(np.mean(effective_nan)) if group_edges_scored.shape[0] else None,
        "effective_zero_ratio": float(np.mean(effective_zero)) if group_edges_scored.shape[0] else None,
        "effective_zero_or_invalid_ratio": float(np.mean(effective_zero | effective_nan))
        if group_edges_scored.shape[0]
        else None,
        "low_improvement_nonzero_ratio": float(np.mean(low_improvement_nonzero))
        if group_edges_scored.shape[0]
        else None,
        "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero)) if group_edges_scored.shape[0] else None,
        "median_raw_best_lag_minutes": nan_quantile(best_lags * method_minutes, 0.5),
        "median_effective_lag_minutes": nan_quantile(effective_lags * method_minutes, 0.5),
        "p90_effective_lag_minutes": nan_quantile(effective_lags * method_minutes, 0.9),
        "median_zero_lag_corr": nan_quantile(zero_corrs, 0.5),
        "median_best_lag_corr": nan_quantile(best_corrs, 0.5),
        "median_zero_lag_score": nan_quantile(zero_scores, 0.5),
        "median_best_score": nan_quantile(best_scores, 0.5),
        "median_corr_improvement": nan_quantile(improvements, 0.5),
    }


def summarize_group_delay_distance_bins(
    rows: list[dict[str, Any]],
    bins: np.ndarray,
) -> list[dict[str, Any]]:
    if not rows or rows[0].get("centroid_distance_km") is None:
        return []
    out = []
    distances = np.asarray([row["centroid_distance_km"] for row in rows], dtype=np.float64)
    high_conf = np.asarray([row["high_conf_nonzero_delay"] for row in rows], dtype=bool)
    low_improve = np.asarray([row["low_improvement_nonzero_delay"] for row in rows], dtype=bool)
    corr_filtered = np.asarray([row["corr_filtered_delay"] for row in rows], dtype=bool)
    effective_lags = np.asarray([row["effective_lag_minutes"] for row in rows], dtype=np.float64)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if math.isinf(hi):
            mask = distances >= lo
            label = f"[{lo:g}, inf)"
        else:
            mask = (distances >= lo) & (distances < hi)
            label = f"[{lo:g}, {hi:g})"
        n = int(mask.sum())
        out.append(
            {
                "distance_bin_km": label,
                "num_group_edges": n,
                "corr_filtered_ratio": float(np.mean(corr_filtered[mask])) if n else None,
                "effective_zero_ratio": float(np.mean(effective_lags[mask] == 0)) if n else None,
                "low_improvement_nonzero_ratio": float(np.mean(low_improve[mask])) if n else None,
                "high_conf_nonzero_ratio": float(np.mean(high_conf[mask])) if n else None,
                "median_effective_lag_minutes": nan_quantile(effective_lags[mask], 0.5) if n else None,
            }
        )
    return out


def stability_rows(edge_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        if row["window"] != "week_daily":
            continue
        key = (
            row["dataset"],
            row["graph"],
            row["method"],
            row["source_group"],
            row["target_group"],
        )
        buckets[key].append(row)

    out = []
    for key, items in sorted(buckets.items()):
        finite_lags = [
            int(round(float(row["effective_lag_minutes"])))
            for row in items
            if row["effective_lag_minutes"] is not None and np.isfinite(float(row["effective_lag_minutes"]))
        ]
        nonzero_lags = [lag for lag in finite_lags if lag > 0]
        mode_lag = None
        lag_agreement = 0.0
        if nonzero_lags:
            mode_lag, mode_count = Counter(nonzero_lags).most_common(1)[0]
            lag_agreement = mode_count / len(nonzero_lags)
        corr_valid_days = sum(not row["corr_filtered_delay"] for row in items)
        high_conf_days = sum(row["high_conf_nonzero_delay"] for row in items)
        effective_nonzero_days = sum(
            row["effective_lag_minutes"] is not None
            and np.isfinite(float(row["effective_lag_minutes"]))
            and float(row["effective_lag_minutes"]) > 0
            for row in items
        )
        num_days = len(items)
        out.append(
            {
                "dataset": key[0],
                "graph": key[1],
                "method": key[2],
                "source_group": key[3],
                "target_group": key[4],
                "num_days_scored": num_days,
                "corr_valid_day_ratio": corr_valid_days / num_days if num_days else None,
                "effective_nonzero_day_ratio": effective_nonzero_days / num_days if num_days else None,
                "high_conf_nonzero_day_ratio": high_conf_days / num_days if num_days else None,
                "mode_nonzero_lag_minutes": mode_lag,
                "nonzero_lag_agreement_ratio": lag_agreement,
                "stable_high_conf_nonzero_delay": (
                    (high_conf_days / num_days >= args.stable_day_ratio)
                    and (lag_agreement >= args.lag_agreement_ratio)
                )
                if num_days
                else False,
            }
        )
    return out


def audit_group_window_method(
    dataset: str,
    dataset_dir: Path,
    graph_name: str,
    raw_group_flow: np.ndarray,
    group_edges_all: np.ndarray,
    edge_meta: dict[tuple[int, int], dict[str, Any]],
    window_spec: dict[str, Any],
    method: str,
    frequency: int,
    steps_per_day: int,
    bins_km: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    group_edges, num_eligible_edges = sample_group_edges(group_edges_all, raw_group_flow, args)
    if group_edges.shape[0] == 0:
        summary = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir),
            "graph": graph_name,
            "window": window_spec["window"],
            "window_label": window_spec["label"],
            "method": method,
            "num_group_edges_total": int(group_edges_all.shape[0]),
            "num_group_edges_eligible": int(num_eligible_edges),
            "num_group_edges_scored": 0,
            "error": "no eligible inter-group edges",
        }
        return [], summary, []

    flow, method_minutes, _method_meta = prepare_method_flow(
        raw_flow=raw_group_flow,
        method=method,
        frequency=frequency,
        steps_per_day=steps_per_day,
        args=args,
    )
    max_lag_steps = max(0, int((args.max_lag * frequency) // method_minutes))
    corr_by_lag, count_by_lag, best_lags, best_corrs, best_scores, improvements, score_mode = score_delay(
        flow=flow,
        edges=group_edges,
        method=method,
        max_lag_steps=max_lag_steps,
        chunk_size=args.score_chunk_size,
    )
    zero_corrs = corr_by_lag[0]
    zero_scores = np.abs(zero_corrs) if score_mode == "absolute_corr" else zero_corrs
    corr_filtered = best_scores < args.min_corr
    low_improvement_nonzero = (
        (best_lags > 0)
        & (best_scores >= args.min_corr)
        & (improvements < args.min_improvement)
    )
    high_conf_nonzero = (
        (best_lags > 0)
        & (best_scores >= args.min_corr)
        & (improvements >= args.min_improvement)
    )
    effective_lags = best_lags.astype(np.float64)
    effective_lags[corr_filtered] = np.nan

    edge_rows = make_delay_rows(
        dataset=dataset,
        graph=graph_name,
        window_spec=window_spec,
        method=method,
        method_minutes=method_minutes,
        group_edges=group_edges,
        edge_meta=edge_meta,
        corr_by_lag=corr_by_lag,
        count_by_lag=count_by_lag,
        best_lags=best_lags,
        best_corrs=best_corrs,
        best_scores=best_scores,
        zero_corrs=zero_corrs,
        zero_scores=zero_scores,
        improvements=improvements,
        effective_lags=effective_lags,
        corr_filtered=corr_filtered,
        low_improvement_nonzero=low_improvement_nonzero,
        high_conf_nonzero=high_conf_nonzero,
    )
    summary = summarize_group_delay(
        dataset=dataset,
        dataset_dir=dataset_dir,
        graph=graph_name,
        window_spec=window_spec,
        method=method,
        method_minutes=method_minutes,
        raw_group_flow=raw_group_flow,
        group_edges_all=group_edges_all,
        group_edges_scored=group_edges,
        num_eligible_edges=num_eligible_edges,
        score_mode=score_mode,
        best_lags=best_lags,
        best_corrs=best_corrs,
        best_scores=best_scores,
        zero_corrs=zero_corrs,
        zero_scores=zero_scores,
        improvements=improvements,
        effective_lags=effective_lags,
        corr_filtered=corr_filtered,
        low_improvement_nonzero=low_improvement_nonzero,
        high_conf_nonzero=high_conf_nonzero,
        args=args,
    )
    distance_rows = summarize_group_delay_distance_bins(edge_rows, bins_km)
    for row in distance_rows:
        row.update(
            {
                "dataset": dataset,
                "graph": graph_name,
                "window": window_spec["window"],
                "window_label": window_spec["label"],
                "method": method,
                "method_step_minutes": int(method_minutes),
            }
        )
    return edge_rows, summary, distance_rows


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
        raise ValueError(f"{name} frequency is {frequency} minutes, expected 5.")
    steps_per_day = max(1, int(round(1440 / max(1, frequency))))
    graph_name = args.graph_variant
    adj = load_graph_variants(dataset_dir, [graph_name])[graph_name]
    lat_lng = load_lat_lng(dataset_dir, int(desc["num_nodes"]))

    assignment, grouping_meta = build_groups(adj, args)
    num_groups = int(assignment.max()) + 1
    centroids = group_centroids(assignment, lat_lng, num_groups)
    grouping_summary, group_rows = summarize_grouping(adj, assignment, centroids)
    group_edges, group_edge_rows, group_edge_meta = build_inter_group_edges(
        adj=adj,
        assignment=assignment,
        lat_lng=lat_lng,
        centroids=centroids,
        min_boundary_edges=args.min_boundary_edges,
    )

    dataset_out = output_root / name
    dataset_out.mkdir(parents=True, exist_ok=True)
    write_csv(dataset_out / "group_assignments.csv", group_assignment_rows(assignment, lat_lng))
    write_csv(dataset_out / "group_summary.csv", group_rows)
    write_csv(dataset_out / "inter_group_edges.csv", group_edge_rows)

    window_specs = get_window_specs(desc, dataset_dir, args, frequency, steps_per_day)
    all_edge_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_distance_rows: list[dict[str, Any]] = []
    group_signal_files: list[str] = []

    for window_spec in window_specs:
        if int(window_spec.get("num_steps", 1)) == 0 and "time_indices" not in window_spec:
            continue
        raw_flow = load_window_flow(dataset_dir, desc, window_spec, args)
        group_flow = aggregate_group_signal(raw_flow, assignment, args.pooling)
        signal_path = dataset_out / f"group_signal_{window_spec['window']}_{window_spec['label']}.npy"
        np.save(signal_path, group_flow)
        group_signal_files.append(str(signal_path))
        print(
            f"[group-delay] {name} window={window_spec['window']} label={window_spec['label']} "
            f"node_steps={raw_flow.shape[0]} groups={group_flow.shape[1]} group_edges={group_edges.shape[0]}",
            flush=True,
        )

        for method in args.methods:
            print(f"[group-delay] {name}/{graph_name}/{window_spec['label']}/{method}", flush=True)
            edge_rows, summary, distance_rows = audit_group_window_method(
                dataset=name,
                dataset_dir=dataset_dir,
                graph_name=graph_name,
                raw_group_flow=group_flow,
                group_edges_all=group_edges,
                edge_meta=group_edge_meta,
                window_spec=window_spec,
                method=method,
                frequency=frequency,
                steps_per_day=steps_per_day,
                bins_km=bins_km,
                args=args,
            )
            all_edge_rows.extend(edge_rows)
            all_summary_rows.append(summary)
            all_distance_rows.extend(distance_rows)

    stable_rows = stability_rows(all_edge_rows, args)
    write_csv(dataset_out / "group_delay_edges.csv", all_edge_rows)
    write_csv(dataset_out / "group_delay_summary.csv", all_summary_rows)
    write_csv(dataset_out / "group_delay_distance_bin_summary.csv", all_distance_rows)
    write_csv(dataset_out / "group_delay_weekly_stability.csv", stable_rows)

    dataset_summary = {
        "dataset": name,
        "dataset_dir": str(dataset_dir),
        "desc": desc,
        "graph": graph_name,
        "grouping": {
            **grouping_meta,
            **grouping_summary,
            "requested_grouping": args.grouping,
            "target_group_size": args.target_group_size,
            "max_group_size": args.max_group_size,
            "pooling": args.pooling,
            "num_inter_group_edges": int(group_edges.shape[0]),
            "min_boundary_edges": int(args.min_boundary_edges),
        },
        "group_signal_files": group_signal_files,
        "summary_rows": all_summary_rows,
        "distance_bin_rows": all_distance_rows,
        "stability_rows": stable_rows,
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
        print(f"[group-delay] dataset={name} path={path}", flush=True)
        try:
            results.append(audit_one_dataset(name, path, args, output_root, bins_km))
        except Exception as exc:
            errors.append({"dataset": name, "path": str(path), "error": repr(exc)})
            if args.require_all:
                raise

    combined_summary_rows: list[dict[str, Any]] = []
    combined_distance_rows: list[dict[str, Any]] = []
    combined_stability_rows: list[dict[str, Any]] = []
    for result in results:
        combined_summary_rows.extend(result["summary_rows"])
        combined_distance_rows.extend(result["distance_bin_rows"])
        combined_stability_rows.extend(result["stability_rows"])
    write_csv(output_root / "all_group_delay_summary.csv", combined_summary_rows)
    write_csv(output_root / "all_group_delay_distance_bin_summary.csv", combined_distance_rows)
    write_csv(output_root / "all_group_delay_weekly_stability.csv", combined_stability_rows)
    write_json(
        output_root / "run_summary.json",
        {
            "args": vars(args),
            "default_dataset_candidates": DEFAULT_DATASET_CANDIDATES,
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

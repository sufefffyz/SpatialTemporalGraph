#!/usr/bin/env python3
"""Build effective-resistance top-k prior graphs for BasicTS.

This graph variant isolates edge selection:

* selection: per receiver node, keep K nodes with the smallest effective
  resistance on an undirected physical conductance graph;
* weight: by default, reuse the OSRM Gaussian distance weight used by the
  OSRM/GSP top-k priors, so fixed-K comparisons differ only in selected edges.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SD")
    parser.add_argument("--distance", required=True, type=Path, help="OSRM distance matrix .npy in meters.")
    parser.add_argument("--base-dataset", required=True, type=Path, help="BasicTS base dataset dir, e.g. BasicTS/datasets/SD.")
    parser.add_argument("--conductance-adj", required=True, type=Path, help="Physical graph used to compute effective resistance.")
    parser.add_argument("--output-graph-dir", required=True, type=Path)
    parser.add_argument("--output-dataset-root", required=True, type=Path, help="Usually BasicTS/datasets.")
    parser.add_argument("--node-ids", type=Path, help="Optional node_id CSV. Defaults to base meta.csv if present.")
    parser.add_argument("--k-list", default="64")
    parser.add_argument("--sigma", type=float, default=None, help="OSRM Gaussian sigma in meters. Default estimates from conductance graph.")
    parser.add_argument("--weight-mode", choices=["osrm_gaussian", "resistance_gaussian"], default="osrm_gaussian")
    parser.add_argument("--min-weight", type=float, default=1e-8, help="Positive floor applied to retained edges to preserve the K-hop support.")
    parser.add_argument("--include-self", action="store_true", default=True)
    parser.add_argument("--copy-adj", action="store_true", help="Copy adj_mx.pkl instead of symlinking.")
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


def load_adjacency(path: Path) -> np.ndarray:
    obj = load_pickle(path)
    if isinstance(obj, (tuple, list)) and len(obj) >= 3:
        obj = obj[-1]
    return np.asarray(obj, dtype=np.float64)


def load_node_ids(path: Path | None, n: int) -> list[str] | None:
    if path is None:
        return None
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"{path} must contain a node_id, ID, or id column.")
        id_column = None
        for candidate in ("node_id", "ID", "id"):
            if candidate in reader.fieldnames:
                id_column = candidate
                break
        if id_column is None:
            raise ValueError(f"{path} must contain a node_id, ID, or id column.")
        node_ids = [str(row[id_column]) for row in reader]
    if len(node_ids) != n:
        raise ValueError(f"node ID count {len(node_ids)} != graph size {n}.")
    return node_ids


def basicts_payload(adj: np.ndarray, node_ids: list[str] | None) -> Any:
    if node_ids is None:
        return adj
    return (node_ids, {node_id: i for i, node_id in enumerate(node_ids)}, adj)


def offdiag_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def graph_stats(adj: np.ndarray) -> dict[str, Any]:
    n = adj.shape[0]
    support = (adj != 0) & offdiag_mask(n)
    degrees = support.sum(axis=1)
    values = adj[support]
    stats: dict[str, Any] = {
        "num_nodes": int(n),
        "edge_count": int(support.sum()),
        "avg_out_degree": float(support.sum() / n),
        "isolated_rows": int((degrees == 0).sum()),
        "min_out_degree": int(degrees.min()) if degrees.size else 0,
        "median_out_degree": float(np.median(degrees)) if degrees.size else 0.0,
        "p90_out_degree": float(np.percentile(degrees, 90)) if degrees.size else 0.0,
        "max_out_degree": int(degrees.max()) if degrees.size else 0,
    }
    if values.size:
        stats["weight_quantiles"] = [float(item) for item in np.percentile(values, [0, 5, 25, 50, 75, 95, 100])]
    return stats


def estimate_osrm_sigma(distance: np.ndarray, reference_adj: np.ndarray) -> tuple[float, dict[str, Any]]:
    n = distance.shape[0]
    mask = (reference_adj > 0) & offdiag_mask(n)
    weights = reference_adj[mask].astype(np.float64)
    dists = distance[mask].astype(np.float64)
    denom = np.sqrt(-np.log(weights))
    valid = np.isfinite(dists) & (dists > 0.0) & np.isfinite(denom) & (denom > 1e-12)
    sigmas = dists[valid] / denom[valid]
    if sigmas.size:
        return float(np.median(sigmas)), {
            "mode": "reference-median",
            "valid_edge_count": int(sigmas.size),
            "sigma_quantiles_m": [float(item) for item in np.percentile(sigmas, [5, 25, 50, 75, 95])],
        }
    finite = distance[offdiag_mask(n) & np.isfinite(distance) & (distance > 0.0)]
    if finite.size == 0:
        raise ValueError("No finite positive distances for sigma estimation.")
    return float(np.std(finite)), {"mode": "all-finite-std", "finite_positive_count": int(finite.size)}


def connected_components(conductance: np.ndarray) -> list[np.ndarray]:
    n = conductance.shape[0]
    neighbors = [np.flatnonzero(conductance[i] > 0.0) for i in range(n)]
    seen = np.zeros(n, dtype=bool)
    components: list[np.ndarray] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        nodes: list[int] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for nb in neighbors[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        components.append(np.asarray(nodes, dtype=np.int64))
    return components


def effective_resistance_matrix(adj: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    conductance = np.maximum(adj, 0.0)
    np.fill_diagonal(conductance, 0.0)
    conductance = 0.5 * (conductance + conductance.T)
    n = conductance.shape[0]
    resistance = np.full((n, n), np.inf, dtype=np.float64)
    eig_info: list[dict[str, Any]] = []

    for comp in connected_components(conductance):
        sub = conductance[np.ix_(comp, comp)]
        if comp.size == 1:
            resistance[comp[0], comp[0]] = 0.0
            eig_info.append({"size": 1, "positive_eigenvalues": 0})
            continue
        degree = sub.sum(axis=1)
        lap = np.diag(degree) - sub
        eigvals, eigvecs = np.linalg.eigh(lap)
        positive = eigvals > max(float(eigvals[-1]) * 1e-10, 1e-12)
        inv_vals = np.zeros_like(eigvals)
        inv_vals[positive] = 1.0 / eigvals[positive]
        lap_pinv = (eigvecs * inv_vals) @ eigvecs.T
        diag = np.diag(lap_pinv)
        sub_res = diag[:, None] + diag[None, :] - 2.0 * lap_pinv
        sub_res = np.maximum(sub_res, 0.0)
        resistance[np.ix_(comp, comp)] = sub_res
        eig_info.append(
            {
                "size": int(comp.size),
                "positive_eigenvalues": int(positive.sum()),
                "lambda_min_positive": float(eigvals[positive].min()) if positive.any() else None,
                "lambda_max": float(eigvals[-1]),
            }
        )

    np.fill_diagonal(resistance, 0.0)
    info = {
        "conductance_edges_undirected": int(np.triu(conductance > 0.0, k=1).sum()),
        "components": len(eig_info),
        "component_sizes": [item["size"] for item in eig_info],
        "eigendecomposition": eig_info,
    }
    finite = resistance[offdiag_mask(n) & np.isfinite(resistance)]
    if finite.size:
        info["resistance_quantiles"] = [float(item) for item in np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])]
    return resistance, info


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size <= k:
        return np.argsort(scores)
    idx = np.argpartition(scores, k - 1)[:k]
    return idx[np.argsort(scores[idx])]


def build_effres_topk(resistance: np.ndarray, distance: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = resistance.shape[0]
    edge_i: list[np.ndarray] = []
    edge_j: list[np.ndarray] = []
    fallback_edges = 0
    for i in range(n):
        row = resistance[i]
        cand = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if cand.size >= k:
            chosen = cand[topk_indices(row[cand], k)]
        else:
            chosen_list = list(cand[topk_indices(row[cand], cand.size)]) if cand.size else []
            missing = k - len(chosen_list)
            if missing > 0:
                fallback = np.flatnonzero(np.isfinite(distance[i]) & (distance[i] > 0.0))
                if chosen_list:
                    fallback = np.setdiff1d(fallback, np.asarray(chosen_list, dtype=np.int64), assume_unique=False)
                fallback = fallback[topk_indices(distance[i, fallback], min(missing, fallback.size))]
                chosen_list.extend(int(item) for item in fallback)
                fallback_edges += int(fallback.size)
            chosen = np.asarray(chosen_list, dtype=np.int64)
        edge_i.append(np.full(chosen.size, i, dtype=np.int64))
        edge_j.append(chosen.astype(np.int64))
    info = {"fallback_osrm_edges": int(fallback_edges)}
    return np.concatenate(edge_i), np.concatenate(edge_j), info


def adjacency_from_edges(
    distance: np.ndarray,
    resistance: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    *,
    sigma: float,
    weight_mode: str,
    include_self: bool,
) -> np.ndarray:
    n = distance.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    if weight_mode == "osrm_gaussian":
        scores = distance[edge_i, edge_j].astype(np.float64)
        weights = np.exp(-np.square(scores / sigma)).astype(np.float32)
    elif weight_mode == "resistance_gaussian":
        scores = resistance[edge_i, edge_j].astype(np.float64)
        finite = scores[np.isfinite(scores) & (scores > 0.0)]
        tau = float(np.median(finite)) if finite.size else 1.0
        weights = np.exp(-np.square(scores / max(tau, 1e-12))).astype(np.float32)
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")
    adj[edge_i, edge_j] = weights
    if include_self:
        np.fill_diagonal(adj, 1.0)
    return adj


def edge_quality(distance: np.ndarray, resistance: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray) -> dict[str, Any]:
    d = distance[edge_i, edge_j].astype(np.float64)
    r = resistance[edge_i, edge_j].astype(np.float64)
    finite_r = r[np.isfinite(r)]
    return {
        "distance_m_mean": float(np.mean(d)),
        "distance_m_median": float(np.median(d)),
        "distance_m_p90": float(np.percentile(d, 90)),
        "resistance_mean": float(np.mean(finite_r)) if finite_r.size else math.inf,
        "resistance_median": float(np.median(finite_r)) if finite_r.size else math.inf,
        "resistance_p90": float(np.percentile(finite_r, 90)) if finite_r.size else math.inf,
    }


def replace_path(path: Path, target: Path, copy: bool = False) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    if copy:
        shutil.copy2(target, path)
    else:
        path.symlink_to(target.resolve())


def stage_dataset(
    base_dataset: Path,
    output_root: Path,
    dataset_name: str,
    adj_path: Path,
    graph_variant: dict[str, Any],
    copy_adj: bool,
    overwrite: bool,
) -> Path:
    dataset_dir = output_root / dataset_name
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"{dataset_dir} exists. Use --overwrite.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_desc = json.loads((base_dataset / "desc.json").read_text(encoding="utf-8"))
    desc = dict(base_desc)
    desc["name"] = dataset_name
    desc["graph_variant"] = graph_variant
    (dataset_dir / "desc.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    for filename in ("data.dat", "meta.csv"):
        source = base_dataset / filename
        if source.exists():
            replace_path(dataset_dir / filename, source)
    replace_path(dataset_dir / "adj_mx.pkl", adj_path, copy=copy_adj)
    return dataset_dir


def main() -> None:
    args = parse_args()
    args.output_graph_dir.mkdir(parents=True, exist_ok=True)
    args.output_dataset_root.mkdir(parents=True, exist_ok=True)

    distance = np.load(args.distance)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"distance matrix must be square, got {distance.shape}.")
    conductance_adj = load_adjacency(args.conductance_adj)
    if conductance_adj.shape != distance.shape:
        raise ValueError(f"conductance adjacency shape {conductance_adj.shape} != distance shape {distance.shape}.")

    node_ids_path = args.node_ids
    if node_ids_path is None and (args.base_dataset / "meta.csv").exists():
        node_ids_path = args.base_dataset / "meta.csv"
    node_ids = load_node_ids(node_ids_path, distance.shape[0])

    sigma, sigma_info = (
        (float(args.sigma), {"mode": "explicit", "sigma_m": float(args.sigma)})
        if args.sigma
        else estimate_osrm_sigma(distance, conductance_adj)
    )
    sigma_info["sigma_m"] = sigma
    resistance, resistance_info = effective_resistance_matrix(conductance_adj)

    graphs = []
    for k in parse_csv_ints(args.k_list):
        edge_i, edge_j, selection_info = build_effres_topk(resistance, distance, k)
        adj = adjacency_from_edges(
            distance,
            resistance,
            edge_i,
            edge_j,
            sigma=sigma,
            weight_mode=args.weight_mode,
            include_self=args.include_self,
        )
        if args.min_weight > 0:
            adj[edge_i, edge_j] = np.maximum(adj[edge_i, edge_j], np.float32(args.min_weight))
        dataset_name = f"{args.dataset}_EFFRESTOPK_K{k:03d}"
        graph_path = args.output_graph_dir / f"{dataset_name}_adj_mx.pkl"
        dump_pickle(graph_path, basicts_payload(adj, node_ids), overwrite=args.overwrite)
        variant = {
            "source": "effective_resistance_topk",
            "k": int(k),
            "selection": {
                "metric": "effective_resistance",
                "conductance_adj": str(args.conductance_adj),
                **selection_info,
            },
            "weight": args.weight_mode,
            "min_weight": float(args.min_weight),
            "sigma": sigma_info,
            "resistance": resistance_info,
            "graph_path": str(graph_path),
            "stats": graph_stats(adj),
            "edge_quality": edge_quality(distance, resistance, edge_i, edge_j),
        }
        dataset_dir = stage_dataset(
            base_dataset=args.base_dataset,
            output_root=args.output_dataset_root,
            dataset_name=dataset_name,
            adj_path=graph_path,
            graph_variant=variant,
            copy_adj=args.copy_adj,
            overwrite=args.overwrite,
        )
        graphs.append(
            {
                "dataset_name": dataset_name,
                "dataset_dir": str(dataset_dir),
                "adj_mx": str((dataset_dir / "adj_mx.pkl").resolve()),
                "stats": variant["stats"],
                "edge_quality": variant["edge_quality"],
                "selection": variant["selection"],
            }
        )

    summary = {
        "dataset": args.dataset,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance": str(args.distance),
        "base_dataset": str(args.base_dataset),
        "conductance_adj": str(args.conductance_adj),
        "weight_mode": args.weight_mode,
        "graphs": graphs,
    }
    summary_path = args.output_graph_dir / f"{args.dataset}_effective_resistance_topk_graph_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists. Use --overwrite.")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build OSRM-nearest and GSP-smooth top-k prior graphs for BasicTS.

The two variants intentionally share the same edge weights.  OSRM top-k and GSP
smooth top-k differ only in edge selection:

* osrm_topk: per receiver node, keep the K smallest OSRM-distance neighbors.
* gsp_smooth_topk: per receiver node, first restrict to a physical OSRM
  candidate pool, then keep the K lowest training-signal roughness edges.

Only the train split is used for signal roughness.
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
    parser.add_argument("--output-graph-dir", required=True, type=Path)
    parser.add_argument("--output-dataset-root", required=True, type=Path, help="Usually BasicTS/datasets.")
    parser.add_argument("--reference-adj", type=Path, help="Reference adj_mx.pkl for Gaussian sigma estimation.")
    parser.add_argument("--node-ids", type=Path, help="Optional node_id CSV.")
    parser.add_argument("--k-list", default="32,64")
    parser.add_argument("--gsp-pool-k", type=int, default=128, help="Physical nearest-neighbor pool before GSP pruning.")
    parser.add_argument("--lambda-dist", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-hub", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--sigma", type=float, default=None)
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


def load_adjacency(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    obj = load_pickle(path)
    if isinstance(obj, (tuple, list)) and len(obj) >= 3:
        return np.asarray(obj[-1])
    return np.asarray(obj)


def load_node_ids(path: Path | None, n: int) -> list[str] | None:
    if path is None:
        return None
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"{path} must contain a node_id or ID column.")
        id_column = "node_id" if "node_id" in reader.fieldnames else "ID" if "ID" in reader.fieldnames else None
        if id_column is None:
            raise ValueError(f"{path} must contain a node_id or ID column.")
        node_ids = [str(row[id_column]) for row in reader]
    if len(node_ids) != n:
        raise ValueError(f"node ID count {len(node_ids)} != distance size {n}.")
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


def estimate_sigma(distance: np.ndarray, reference_adj: np.ndarray | None) -> tuple[float, dict[str, Any]]:
    if reference_adj is not None:
        ref_mask = (reference_adj > 0) & offdiag_mask(distance.shape[0])
        weights = reference_adj[ref_mask].astype(np.float64)
        dists = distance[ref_mask].astype(np.float64)
        denom = np.sqrt(-np.log(weights))
        valid = np.isfinite(dists) & (dists > 0.0) & np.isfinite(denom) & (denom > 1e-12)
        sigmas = dists[valid] / denom[valid]
        if sigmas.size:
            return float(np.median(sigmas)), {
                "mode": "reference-median",
                "valid_edge_count": int(sigmas.size),
                "sigma_quantiles_m": [float(item) for item in np.percentile(sigmas, [5, 25, 50, 75, 95])],
            }

    mask = offdiag_mask(distance.shape[0]) & np.isfinite(distance) & (distance > 0.0)
    values = distance[mask]
    if values.size == 0:
        raise ValueError("No finite positive distances for sigma estimation.")
    return float(np.std(values)), {"mode": "all-finite-std", "finite_positive_count": int(values.size)}


def load_train_signal(base_dataset: Path) -> tuple[np.ndarray, dict[str, Any]]:
    desc = json.loads((base_dataset / "desc.json").read_text(encoding="utf-8"))
    shape = tuple(desc["shape"])
    train_ratio = float(desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"][0])
    train_steps = int(shape[0] * train_ratio)
    data = np.memmap(base_dataset / "data.dat", dtype="float32", mode="r", shape=shape)
    signal = np.asarray(data[:train_steps, :, 0], dtype=np.float32)
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    signal = (signal - mean) / np.maximum(std, 1e-6)
    return signal, {"shape": list(shape), "train_steps": train_steps, "feature_index": 0}


def topk_indices(scores: np.ndarray, k: int, largest: bool) -> np.ndarray:
    if scores.size <= k:
        order = np.argsort(scores)
        return order[::-1] if largest else order
    if largest:
        idx = np.argpartition(scores, -k)[-k:]
        return idx[np.argsort(scores[idx])[::-1]]
    idx = np.argpartition(scores, k - 1)[:k]
    return idx[np.argsort(scores[idx])]


def build_osrm_topk(distance: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = distance.shape[0]
    edge_i: list[np.ndarray] = []
    edge_j: list[np.ndarray] = []
    for i in range(n):
        row = distance[i]
        cand = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if cand.size == 0:
            continue
        chosen = cand[topk_indices(row[cand], min(k, cand.size), largest=False)]
        edge_i.append(np.full(chosen.size, i, dtype=np.int64))
        edge_j.append(chosen.astype(np.int64))
    return np.concatenate(edge_i), np.concatenate(edge_j)


def build_gsp_topk(
    distance: np.ndarray,
    signal: np.ndarray,
    k: int,
    pool_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = distance.shape[0]
    pool_i: list[np.ndarray] = []
    pool_j: list[np.ndarray] = []
    for i in range(n):
        row = distance[i]
        cand = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if cand.size == 0:
            continue
        chosen = cand[topk_indices(row[cand], min(pool_k, cand.size), largest=False)]
        pool_i.append(np.full(chosen.size, i, dtype=np.int64))
        pool_j.append(chosen.astype(np.int64))

    cand_i = np.concatenate(pool_i)
    cand_j = np.concatenate(pool_j)
    cand_dist = distance[cand_i, cand_j].astype(np.float64)
    rough = np.square(signal[:, cand_i] - signal[:, cand_j]).mean(axis=0).astype(np.float64)
    dist_scale = max(float(np.median(cand_dist[np.isfinite(cand_dist)])), 1e-6)
    rough_scale = max(float(np.median(rough[np.isfinite(rough)])), 1e-6)

    keep = np.zeros(cand_i.size, dtype=bool)
    for i in range(n):
        idx = np.flatnonzero(cand_i == i)
        if idx.size == 0:
            continue
        chosen_local = topk_indices(rough[idx], min(k, idx.size), largest=False)
        keep[idx[chosen_local]] = True

    info = {
        "selection_metric": "train_signal_roughness",
        "candidate_pool_edges": int(cand_i.size),
        "candidate_pool_avg_degree": float(cand_i.size / n),
        "roughness_scale": rough_scale,
        "roughness_quantiles": [float(item) for item in np.percentile(rough, [0, 5, 25, 50, 75, 95, 100])],
        "distance_scale_m": dist_scale,
    }
    return cand_i[keep], cand_j[keep], info


def adjacency_from_edges(distance: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, sigma: float, include_self: bool) -> np.ndarray:
    n = distance.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    weights = np.exp(-np.square(distance[edge_i, edge_j].astype(np.float64) / sigma)).astype(np.float32)
    adj[edge_i, edge_j] = weights
    if include_self:
        np.fill_diagonal(adj, 1.0)
    return adj


def edge_quality(distance: np.ndarray, signal: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray) -> dict[str, Any]:
    d = distance[edge_i, edge_j].astype(np.float64)
    rough = np.square(signal[:, edge_i] - signal[:, edge_j]).mean(axis=0).astype(np.float64)
    return {
        "distance_m_mean": float(np.mean(d)),
        "distance_m_median": float(np.median(d)),
        "distance_m_p90": float(np.percentile(d, 90)),
        "roughness_mean": float(np.mean(rough)),
        "roughness_median": float(np.median(rough)),
        "roughness_p90": float(np.percentile(rough, 90)),
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
    if dataset_dir.exists():
        if not overwrite:
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
    n = distance.shape[0]
    reference_adj = load_adjacency(args.reference_adj)
    if reference_adj is not None and reference_adj.shape != distance.shape:
        raise ValueError(f"reference adjacency shape {reference_adj.shape} != distance shape {distance.shape}.")
    sigma, sigma_info = (float(args.sigma), {"mode": "explicit", "sigma_m": float(args.sigma)}) if args.sigma else estimate_sigma(distance, reference_adj)
    sigma_info["sigma_m"] = sigma
    node_ids = load_node_ids(args.node_ids, n)
    signal, signal_info = load_train_signal(args.base_dataset)

    graphs = []
    for k in parse_csv_ints(args.k_list):
        variants: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
        osrm_i, osrm_j = build_osrm_topk(distance, k)
        variants.append(("osrm_topk", osrm_i, osrm_j, {}))
        gsp_i, gsp_j, gsp_info = build_gsp_topk(
            distance=distance,
            signal=signal,
            k=k,
            pool_k=max(args.gsp_pool_k, k),
        )
        variants.append(("gsp_smooth_topk", gsp_i, gsp_j, gsp_info))

        for method, edge_i, edge_j, method_info in variants:
            adj = adjacency_from_edges(distance, edge_i, edge_j, sigma=sigma, include_self=args.include_self)
            dataset_name = f"{args.dataset}_{'OSRMTOPK' if method == 'osrm_topk' else 'GSPTOPK'}_K{k:03d}"
            graph_path = args.output_graph_dir / f"{dataset_name}_adj_mx.pkl"
            dump_pickle(graph_path, basicts_payload(adj, node_ids), overwrite=args.overwrite)
            variant = {
                "source": method,
                "k": int(k),
                "weight": "osrm_gaussian_distance",
                "sigma": sigma_info,
                "signal": signal_info,
                "graph_path": str(graph_path),
                "method": method_info,
                "stats": graph_stats(adj),
                "edge_quality": edge_quality(distance, signal, edge_i, edge_j),
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
                    "method": method,
                    "k": int(k),
                    "dataset_dir": str(dataset_dir),
                    "adj_mx": str((dataset_dir / "adj_mx.pkl").resolve()),
                    "stats": variant["stats"],
                    "edge_quality": variant["edge_quality"],
                }
            )

    summary = {
        "dataset": args.dataset,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance": str(args.distance),
        "base_dataset": str(args.base_dataset),
        "reference_adj": str(args.reference_adj) if args.reference_adj else None,
        "gsp_pool_k": args.gsp_pool_k,
        "gsp_selection_metric": "train_signal_roughness",
        "graphs": graphs,
    }
    summary_path = args.output_graph_dir / f"{args.dataset}_topk_prior_graph_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists. Use --overwrite.")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

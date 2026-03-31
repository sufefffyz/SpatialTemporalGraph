#!/usr/bin/env python3
"""Evaluate BasicTS learned adjacency snapshots against causalrivers label graphs.

This script bridges the BasicTS learned-graph outputs under `learned_graphs/*.npz`
with the causalrivers scoring logic. It loads a learned adjacency matrix, aligns it
to each label subgraph, and reuses `tools.scoring_tools.score()` to compute the same
metrics used by the benchmark.

Typical usage:

    python scripts/evaluate_learned_graphs_against_labels.py \
        --label-path datasets/traffic_city_traffic_m_volume__category__1_0_15min/random_3_paperlike/city_traffic_m_volume__category__1_0_15min.p \
        --learned-graph /path/to/BasicTS/checkpoints/AGCRN/TRAFFIC_VOLUME_15MIN_100_12_12/learned_graphs/final.npz \
        --adj-mx-pkl /path/to/BasicTS/datasets/TRAFFIC_VOLUME_15MIN/adj_mx.pkl \
        --output-dir results/learned_graph_eval/traffic15min_agcrn

    python scripts/evaluate_learned_graphs_against_labels.py \
        --label-path datasets/traffic_city_traffic_m_volume__category__1_0_15min \
        --learned-graph /path/to/BasicTS/checkpoints/AGCRN/TRAFFIC_VOLUME_15MIN_100_12_12/learned_graphs \
        --adj-mx-pkl /path/to/BasicTS/datasets/TRAFFIC_VOLUME_15MIN/adj_mx.pkl \
        --output-dir results/learned_graph_eval/traffic15min_agcrn_batch
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.scoring_tools import remove_diagonal, score_preprocessed
from tools.tools import graph_to_label_tensor, load_label_graphs


@dataclass(frozen=True)
class LearnedGraphArtifact:
    path: Path
    name: str
    adj: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedLabelSet:
    label_info: dict[str, Any]
    sample_nodes: list[list[Any]]
    label_tensors_preprocessed: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learned BasicTS adjacency snapshots against causalrivers label graphs."
    )
    parser.add_argument(
        "--label-path",
        action="append",
        required=True,
        default=[],
        dest="label_paths",
        type=Path,
        help=(
            "Path to a causalrivers label .p/.pkl file or a directory containing label pickles. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--learned-graph",
        action="append",
        required=True,
        default=[],
        dest="learned_graphs",
        type=Path,
        help="Path to a learned_graphs/*.npz snapshot or a directory containing such files. Can be repeated.",
    )
    parser.add_argument(
        "--adj-mx-pkl",
        type=Path,
        default=None,
        help="Optional BasicTS adj_mx.pkl for node-order mapping. Strongly recommended.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "learned_graph_eval" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Directory to store scoring outputs.",
    )
    parser.add_argument("--remove-autoregressive", action="store_true", default=True, help="Remove self-loops before scoring.")
    parser.add_argument("--keep-autoregressive", action="store_false", dest="remove_autoregressive", help="Keep self-loops during scoring.")
    parser.add_argument("--restrict-to", type=int, default=-1, help="Optional label-graph index to evaluate. Defaults to all samples.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel scorer jobs.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for scorer parallelism.")
    return parser.parse_args()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _load_node_order(adj_mx_pkl: Path | None) -> tuple[list[Any] | None, dict[str, int] | None]:
    if adj_mx_pkl is None:
        return None, None
    payload = _load_pickle(adj_mx_pkl)
    if isinstance(payload, tuple) and len(payload) == 3:
        node_ids, node_to_ind, _adj = payload
        normalized_node_ids = [node for node in node_ids]
        normalized_map = {_normalize_node_id(key): int(val) for key, val in node_to_ind.items()}
        return normalized_node_ids, normalized_map
    if isinstance(payload, np.ndarray):
        return list(range(payload.shape[0])), {str(idx): idx for idx in range(payload.shape[0])}
    raise ValueError(f"Unsupported adj_mx.pkl format in {adj_mx_pkl}.")


def _discover_learned_graph_files(inputs: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            learned_dir = path / "learned_graphs"
            if learned_dir.is_dir():
                discovered.extend(sorted(learned_dir.glob("*.npz")))
            else:
                discovered.extend(sorted(path.glob("*.npz")))
        elif path.is_file():
            discovered.append(path)
        else:
            raise FileNotFoundError(f"Learned graph path not found: {path}")
    unique = []
    seen = set()
    for path in discovered:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _discover_label_files(inputs: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.p")))
            discovered.extend(sorted(path.rglob("*.pkl")))
        elif path.is_file():
            discovered.append(path)
        else:
            raise FileNotFoundError(f"Label path not found: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_learned_graph(path: Path) -> LearnedGraphArtifact:
    with np.load(path, allow_pickle=True) as data:
        if "adj" not in data:
            raise KeyError(f"{path} does not contain an 'adj' array.")
        adj = np.asarray(data["adj"], dtype=np.float32)
        metadata = {
            key: (data[key].item() if np.asarray(data[key]).shape == () else np.asarray(data[key]))
            for key in data.files
            if key != "adj"
        }
    checkpoint_name = path.parent.parent.name if path.parent.name == "learned_graphs" else path.parent.name
    safe_checkpoint_name = checkpoint_name.replace(" ", "_")
    name = f"{safe_checkpoint_name}__{path.stem}"
    return LearnedGraphArtifact(path=path, name=name, adj=adj, metadata=metadata)


def _normalize_node_id(node: Any) -> str:
    if isinstance(node, (int, np.integer)):
        return str(int(node))
    text = str(node).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def _build_node_index_map(node_order: list[Any] | None, learned_adj_shape: tuple[int, int]) -> dict[str, int]:
    if node_order is None:
        size = int(learned_adj_shape[0])
        return {str(idx): idx for idx in range(size)}
    return {_normalize_node_id(node): idx for idx, node in enumerate(node_order)}


def _label_identifier(label_path: Path) -> str:
    parent_name = label_path.parent.name.replace(" ", "_")
    stem = label_path.stem.replace(" ", "_")
    return f"{parent_name}__{stem}"


def _normalize_metadata_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _parse_label_group(label_group: str) -> tuple[str | None, int | None, str | None]:
    parts = label_group.split("_")
    if len(parts) < 2:
        return None, None, None

    n_vars_idx = None
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].isdigit():
            n_vars_idx = idx
            break

    if n_vars_idx is None:
        return None, None, None

    strategy = "_".join(parts[:n_vars_idx]) or None
    n_vars = int(parts[n_vars_idx])
    label_tag = "_".join(parts[n_vars_idx + 1 :]) or None
    return strategy, n_vars, label_tag


def _resolve_node_index_map(
    artifact: LearnedGraphArtifact,
    adj_mx_pkl: Path | None,
    node_order: list[Any] | None,
    adj_node_map: dict[str, int] | None,
) -> dict[str, int]:
    if adj_mx_pkl is None:
        return _build_node_index_map(None, artifact.adj.shape)

    if adj_node_map is None or node_order is None:
        raise ValueError("adj_mx.pkl was provided, but its node mapping could not be loaded.")

    expected_size = len(node_order)
    actual_size = int(artifact.adj.shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f"Snapshot {artifact.path} has {actual_size} nodes, but {adj_mx_pkl} describes "
            f"{expected_size} nodes. Use a matching adj_mx.pkl for this learned graph."
        )
    return adj_node_map


def _score_single_artifact(
    artifact: LearnedGraphArtifact,
    prepared_labels: PreparedLabelSet,
    sample_positions: list[np.ndarray],
    remove_autoregressive: bool,
    n_jobs: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preds = np.stack(
        [
            np.asarray(artifact.adj[np.ix_(node_positions, node_positions)], dtype=np.float32)
            for node_positions in sample_positions
        ]
    )

    if remove_autoregressive:
        preds_preprocessed = remove_diagonal(preds)
    else:
        preds_preprocessed = preds

    start = time.perf_counter()
    scoring = score_preprocessed(
        preds_preprocessed,
        prepared_labels.label_tensors_preprocessed,
        name=artifact.name,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    runtime_seconds = time.perf_counter() - start
    scoring.index.name = "Metric"
    meta = {
        "snapshot_path": str(artifact.path),
        "snapshot_name": artifact.name,
        "num_label_graphs": len(prepared_labels.sample_nodes),
        "adj_shape": list(artifact.adj.shape),
        "runtime_seconds": runtime_seconds,
        "metadata": {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in artifact.metadata.items()},
    }
    return scoring, meta


def _build_sample_positions(sample_nodes: list[list[Any]], node_index_map: dict[str, int]) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for nodes in sample_nodes:
        node_positions = []
        for node in nodes:
            key = _normalize_node_id(node)
            if key not in node_index_map:
                raise KeyError(
                    f"Node {node!r} from label graph is missing from the learned-graph node map."
                )
            node_positions.append(node_index_map[key])
        positions.append(np.asarray(node_positions, dtype=np.int64))
    return positions


def _prepare_label_set(label_path: Path, restrict_to: int, remove_autoregressive: bool) -> PreparedLabelSet:
    label_graphs = load_label_graphs(
        SimpleNamespace(label_path=str(label_path), restrict_to=int(restrict_to))
    )
    if not label_graphs:
        raise ValueError(f"No label graphs loaded from {label_path}")

    sample_nodes = [sorted(sample_graph.nodes) for sample_graph in label_graphs]
    label_tensors = np.stack(
        [np.asarray(graph_to_label_tensor(sample_graph), dtype=np.float32) for sample_graph in label_graphs]
    )
    label_tensors_preprocessed = (
        remove_diagonal(label_tensors)
        if remove_autoregressive
        else label_tensors
    )

    label_id = _label_identifier(label_path)
    strategy, n_vars, label_tag = _parse_label_group(label_path.parent.name)
    label_info = {
        "label_path": str(label_path),
        "label_id": label_id,
        "label_group": label_path.parent.name,
        "label_name": label_path.stem,
        "label_dataset_name": label_path.stem,
        "strategy": strategy,
        "n_vars": n_vars,
        "label_tag": label_tag,
        "num_label_graphs": len(sample_nodes),
    }
    return PreparedLabelSet(
        label_info=label_info,
        sample_nodes=sample_nodes,
        label_tensors_preprocessed=label_tensors_preprocessed,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    label_paths = _discover_label_files(args.label_paths)
    if not label_paths:
        raise FileNotFoundError("No label pickle files found. Pass --label-path with a file or directory.")

    learned_graph_files = _discover_learned_graph_files(args.learned_graphs)
    if not learned_graph_files:
        raise FileNotFoundError("No learned graph snapshots found. Pass --learned-graph with a file or directory.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    node_order, adj_node_map = _load_node_order(args.adj_mx_pkl)

    scoring_tables = []
    runtime_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    label_manifests: list[dict[str, Any]] = []
    multi_label = len(label_paths) > 1

    for label_path in label_paths:
        prepared_labels = _prepare_label_set(
            label_path,
            restrict_to=args.restrict_to,
            remove_autoregressive=args.remove_autoregressive,
        )
        label_info = prepared_labels.label_info
        label_manifests.append(label_info)

        for learned_graph_path in learned_graph_files:
            artifact = _load_learned_graph(learned_graph_path)
            node_index_map = _resolve_node_index_map(
                artifact,
                args.adj_mx_pkl,
                node_order,
                adj_node_map,
            )
            sample_positions = _build_sample_positions(prepared_labels.sample_nodes, node_index_map)
            scoring, meta = _score_single_artifact(
                artifact,
                prepared_labels,
                sample_positions,
                args.remove_autoregressive,
                args.n_jobs,
                args.chunk_size,
            )
            score_name = artifact.name if not multi_label else f"{label_id}__{artifact.name}"
            scoring.columns = [score_name]
            scoring_tables.append(scoring)
            meta.update(label_info)
            meta["score_name"] = score_name
            manifest_entries.append(meta)
            runtime_rows.append(
                {
                    "label_id": label_info["label_id"],
                    "label_group": label_info["label_group"],
                    "label_name": label_info["label_name"],
                    "label_dataset_name": label_info["label_dataset_name"],
                    "label_path": label_info["label_path"],
                    "strategy": label_info["strategy"],
                    "n_vars": label_info["n_vars"],
                    "label_tag": label_info["label_tag"],
                    "snapshot": artifact.name,
                    "score_name": score_name,
                    "snapshot_path": str(artifact.path),
                    "model_name": _normalize_metadata_scalar(artifact.metadata.get("model_name")),
                    "learned_dataset_name": _normalize_metadata_scalar(artifact.metadata.get("dataset_name")),
                    "epoch": _normalize_metadata_scalar(artifact.metadata.get("epoch")),
                    "graph_semantics": _normalize_metadata_scalar(artifact.metadata.get("graph_semantics")),
                    "runtime_seconds": meta["runtime_seconds"],
                    "num_label_graphs": label_info["num_label_graphs"],
                }
            )
            for metric_name, row in scoring[[score_name]].iterrows():
                long_rows.append(
                    {
                        "label_id": label_info["label_id"],
                        "label_group": label_info["label_group"],
                        "label_name": label_info["label_name"],
                        "label_dataset_name": label_info["label_dataset_name"],
                        "label_path": label_info["label_path"],
                        "strategy": label_info["strategy"],
                        "n_vars": label_info["n_vars"],
                        "label_tag": label_info["label_tag"],
                        "snapshot": artifact.name,
                        "score_name": score_name,
                        "snapshot_path": str(artifact.path),
                        "model_name": _normalize_metadata_scalar(artifact.metadata.get("model_name")),
                        "learned_dataset_name": _normalize_metadata_scalar(artifact.metadata.get("dataset_name")),
                        "epoch": _normalize_metadata_scalar(artifact.metadata.get("epoch")),
                        "graph_semantics": _normalize_metadata_scalar(artifact.metadata.get("graph_semantics")),
                        "metric": metric_name,
                        "value": row.iloc[0],
                    }
                )

    if len(scoring_tables) == 1:
        merged = scoring_tables[0]
    else:
        merged = pd.concat(scoring_tables, axis=1)

    scoring_path = output_dir / "scoring.csv"
    scoring_long_path = output_dir / "scoring_long.csv"
    runtime_path = output_dir / "runtime.csv"
    manifest_path = output_dir / "manifest.json"

    merged.to_csv(scoring_path)
    _write_csv(scoring_long_path, long_rows)
    _write_csv(runtime_path, runtime_rows)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "label_path": str(label_paths[0]) if len(label_paths) == 1 else None,
        "label_paths": [str(path) for path in label_paths],
        "learned_graphs": [str(path) for path in learned_graph_files],
        "adj_mx_pkl": str(args.adj_mx_pkl) if args.adj_mx_pkl else None,
        "num_label_sets": len(label_paths),
        "num_label_graphs": label_manifests[0]["num_label_graphs"] if len(label_manifests) == 1 else None,
        "num_scored_pairs": len(scoring_tables),
        "output_dir": str(output_dir),
        "files": {
            "scoring": str(scoring_path),
            "scoring_long": str(scoring_long_path),
            "runtime": str(runtime_path),
        },
        "label_sets": label_manifests,
        "snapshots": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Evaluated {len(learned_graph_files)} learned graph snapshot(s) across "
        f"{len(label_paths)} label set(s); total scored pairs: {len(scoring_tables)}."
    )
    print(f"Outputs written to: {output_dir}")
    print(f"Scoring table: {scoring_path}")
    print(f"Long table: {scoring_long_path}")
    print(f"Runtime table: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

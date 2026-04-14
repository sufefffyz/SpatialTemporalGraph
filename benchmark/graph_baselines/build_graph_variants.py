#!/usr/bin/env python3

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_ROOT = REPO_ROOT / "graphs" / "SD"
DEFAULT_SOURCE_DATASET_DIR = REPO_ROOT / "datasets" / "SD_5min_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SD graph variants from a physical directed adjacency export."
    )
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT, help="Graph output directory")
    parser.add_argument(
        "--directed-adj",
        type=Path,
        help="Directed physical adjacency .pkl (defaults to graph-root/adj_mx_physical_directed.pkl)",
    )
    parser.add_argument(
        "--sensor-ids",
        type=Path,
        help="sensor_ids.npy path (defaults to graph-root/sensor_ids.npy)",
    )
    parser.add_argument(
        "--directed-edges",
        type=Path,
        help="directed_edges.csv path (defaults to graph-root/directed_edges.csv)",
    )
    parser.add_argument(
        "--sensor-index",
        type=Path,
        help="sensor_index.csv path (defaults to graph-root/sensor_index.csv)",
    )
    parser.add_argument(
        "--largest-original-adj",
        type=Path,
        default=DEFAULT_SOURCE_DATASET_DIR / "adj_mx_largeST_original.pkl",
        help="LargeST original adjacency source to copy into graph-root if missing",
    )
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as fp:
        obj = pickle.load(fp)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 3:
            obj = obj[2]
        elif len(obj) == 1:
            obj = obj[0]
    array = np.asarray(obj, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{path} is not a square adjacency matrix: shape={array.shape}")
    return array


def dump_pickle_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as fp:
        pickle.dump(np.asarray(array, dtype=np.float32), fp, protocol=4)


def normalize_sensor_type(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    mapping = {
        "MAINLINE": "ML",
        "ML": "ML",
        "ONRAMP": "OR",
        "OR": "OR",
        "OFFRAMP": "FR",
        "FR": "FR",
    }
    return mapping.get(text, text)


def infer_edge_type_columns(edge_df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("source_id", "target_id"),
        ("from_sensor_id", "to_sensor_id"),
    ]
    for src_col, dst_col in candidates:
        if src_col in edge_df.columns and dst_col in edge_df.columns:
            return src_col, dst_col
    raise ValueError("Could not find source/target sensor id columns in directed_edges.csv")


def infer_type_column(sensor_index_df: pd.DataFrame) -> str | None:
    for column in ("type", "sensor_type", "Type"):
        if column in sensor_index_df.columns:
            return column
    return None


def enrich_edge_types(edge_df: pd.DataFrame, sensor_index_df: pd.DataFrame) -> pd.DataFrame:
    if "edge_type" in edge_df.columns and edge_df["edge_type"].notna().any():
        edge_df = edge_df.copy()
        edge_df["edge_type"] = edge_df["edge_type"].map(normalize_sensor_type)
        return edge_df

    type_column = infer_type_column(sensor_index_df)
    if type_column is None:
        raise ValueError("sensor_index.csv is missing a type column needed to derive ML-only edges")

    sensor_index_df = sensor_index_df.copy()
    sensor_index_df["sensor_id"] = pd.to_numeric(sensor_index_df["sensor_id"], errors="raise").astype(np.int64)
    sensor_index_df[type_column] = sensor_index_df[type_column].map(normalize_sensor_type)
    type_lookup = dict(zip(sensor_index_df["sensor_id"], sensor_index_df[type_column]))

    src_col, dst_col = infer_edge_type_columns(edge_df)
    enriched = edge_df.copy()
    enriched[src_col] = pd.to_numeric(enriched[src_col], errors="raise").astype(np.int64)
    enriched[dst_col] = pd.to_numeric(enriched[dst_col], errors="raise").astype(np.int64)
    enriched["edge_type"] = [
        f"{type_lookup.get(src_id, 'UNK')}_{type_lookup.get(dst_id, 'UNK')}"
        for src_id, dst_id in zip(enriched[src_col], enriched[dst_col])
    ]
    return enriched


def build_ml_only_adjacency(
    directed_adj: np.ndarray,
    sensor_ids: np.ndarray,
    edge_df: pd.DataFrame,
) -> np.ndarray:
    src_col, dst_col = infer_edge_type_columns(edge_df)
    sensor_to_idx = {int(sensor_id): idx for idx, sensor_id in enumerate(sensor_ids.tolist())}
    ml_only = np.zeros_like(directed_adj, dtype=np.float32)
    ml_edges = edge_df[edge_df["edge_type"] == "ML_ML"]
    for _, row in ml_edges.iterrows():
        src_id = int(row[src_col])
        dst_id = int(row[dst_col])
        src_idx = sensor_to_idx.get(src_id)
        dst_idx = sensor_to_idx.get(dst_id)
        if src_idx is None or dst_idx is None:
            continue
        ml_only[src_idx, dst_idx] = directed_adj[src_idx, dst_idx]
    return ml_only


def main() -> None:
    args = parse_args()
    graph_root = args.graph_root.expanduser().resolve()
    graph_root.mkdir(parents=True, exist_ok=True)

    directed_adj_path = ensure_exists(
        args.directed_adj or (graph_root / "adj_mx_physical_directed.pkl"),
        "Directed physical adjacency",
    )
    sensor_ids_path = ensure_exists(
        args.sensor_ids or (graph_root / "sensor_ids.npy"),
        "sensor_ids.npy",
    )
    directed_edges_path = ensure_exists(
        args.directed_edges or (graph_root / "directed_edges.csv"),
        "directed_edges.csv",
    )
    sensor_index_path = ensure_exists(
        args.sensor_index or (graph_root / "sensor_index.csv"),
        "sensor_index.csv",
    )

    directed_adj = load_pickle_array(directed_adj_path)
    sensor_ids = np.load(sensor_ids_path, allow_pickle=True).astype(np.int64)
    if directed_adj.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError(
            f"Directed adjacency shape {directed_adj.shape} does not match sensor_ids length {len(sensor_ids)}"
        )

    edge_df = pd.read_csv(directed_edges_path)
    sensor_index_df = pd.read_csv(sensor_index_path)
    edge_df = enrich_edge_types(edge_df, sensor_index_df)
    edge_df.to_csv(graph_root / "directed_edges.csv", index=False)

    forward = directed_adj.astype(np.float32)
    reverse = directed_adj.T.astype(np.float32)
    bidir = np.maximum(directed_adj, directed_adj.T).astype(np.float32)
    ml_only = build_ml_only_adjacency(directed_adj, sensor_ids, edge_df)

    dump_pickle_array(graph_root / "adj_mx_physical_forward.pkl", forward)
    dump_pickle_array(graph_root / "adj_mx_physical_reverse.pkl", reverse)
    dump_pickle_array(graph_root / "adj_mx_physical_bidir.pkl", bidir)
    dump_pickle_array(graph_root / "adj_mx_physical_ML_only.pkl", ml_only)

    largest_original_target = graph_root / "adj_mx_largeST_original.pkl"
    if not largest_original_target.exists() and args.largest_original_adj:
        source_original = ensure_exists(args.largest_original_adj, "LargeST original adjacency source")
        shutil.copy2(source_original, largest_original_target)

    summary = {
        "graph_root": str(graph_root),
        "num_nodes": int(len(sensor_ids)),
        "num_directed_edges": int(np.count_nonzero(forward)),
        "num_reverse_edges": int(np.count_nonzero(reverse)),
        "num_bidir_edges": int(np.count_nonzero(bidir)),
        "num_ml_only_edges": int(np.count_nonzero(ml_only)),
        "files": {
            "directed": str(directed_adj_path),
            "forward": str(graph_root / "adj_mx_physical_forward.pkl"),
            "reverse": str(graph_root / "adj_mx_physical_reverse.pkl"),
            "bidir": str(graph_root / "adj_mx_physical_bidir.pkl"),
            "ml_only": str(graph_root / "adj_mx_physical_ML_only.pkl"),
            "directed_edges": str(graph_root / "directed_edges.csv"),
        },
    }
    (graph_root / "graph_variants_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

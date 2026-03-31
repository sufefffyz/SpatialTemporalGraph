import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from stgraph_ext.data_utils import (
    build_directed_adjacency_from_graph,
    build_multichannel_series,
    extract_networkx_graph,
    infer_frequency_minutes,
    maybe_int,
    normalize_node_labels,
    resolve_dataset_dir,
    save_basicts_adjacency,
    save_description,
    save_memmap_array,
    save_split_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RiversEastGermany CSV + graph pickle into a BasicTS-compatible dataset bundle."
    )
    parser.add_argument(
        "--input-csv",
        default="../causalrivers/product/rivers_ts_east_germany.csv",
        help="Path to the rivers East Germany CSV file.",
    )
    parser.add_argument(
        "--graph-pickle",
        default="../causalrivers/product/rivers_east_germany.p",
        help="Path to the rivers graph pickle.",
    )
    parser.add_argument(
        "--dataset-name",
        default="RIVERS_EAST_GERMANY_15MIN",
        help="Dataset name under BasicTS/datasets.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets",
        help="Relative or absolute output root directory.",
    )
    parser.add_argument("--input-len", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--output-len", type=int, default=12, help="Output sequence length.")
    return parser.parse_args()


def load_river_graph(graph_path: Path) -> nx.Graph:
    with open(graph_path, "rb") as fp:
        graph_obj = pickle.load(fp)
    graph = extract_networkx_graph(graph_obj)
    mapping = {node: maybe_int(node) for node in graph.nodes}
    return nx.relabel_nodes(graph, mapping, copy=True)


def load_targets(csv_path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.columns = normalize_node_labels(frame.columns.tolist())
    frame = frame.interpolate(limit_direction="both")
    frame = frame.fillna(0.0)
    return frame, pd.DatetimeIndex(frame.index)


def chronological_split(total_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len
    train_idx = np.arange(0, train_len, dtype=np.int64)
    val_idx = np.arange(train_len, train_len + val_len, dtype=np.int64)
    test_idx = np.arange(train_len + val_len, train_len + val_len + test_len, dtype=np.int64)
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    graph_path = Path(args.graph_pickle)
    if not csv_path.is_absolute():
        csv_path = (PROJECT_ROOT / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()
    if not graph_path.is_absolute():
        graph_path = (PROJECT_ROOT / graph_path).resolve()
    else:
        graph_path = graph_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph pickle file not found: {graph_path}")

    output_dir = resolve_dataset_dir(args.dataset_name, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, dt_index = load_targets(csv_path)
    graph = load_river_graph(graph_path)
    node_ids = normalize_node_labels(frame.columns.tolist())
    adjacency = build_directed_adjacency_from_graph(graph, node_ids)
    targets = frame.to_numpy(dtype=np.float32)
    data_with_features, feature_description = build_multichannel_series(targets, dt_index)
    train_idx, val_idx, test_idx = chronological_split(len(frame))
    unix_timestamps = (dt_index.asi8 // 1_000_000_000).astype(np.int64)

    description = {
        "name": args.dataset_name,
        "domain": "river discharge",
        "shape": list(data_with_features.shape),
        "num_time_steps": int(data_with_features.shape[0]),
        "num_nodes": int(data_with_features.shape[1]),
        "num_features": int(data_with_features.shape[2]),
        "feature_description": feature_description,
        "has_graph": True,
        "frequency (minutes)": infer_frequency_minutes(dt_index, fallback_minutes=15),
        "regular_settings": {
            "INPUT_LEN": int(args.input_len),
            "OUTPUT_LEN": int(args.output_len),
            "TRAIN_VAL_TEST_RATIO": [0.7, 0.1, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE", "WAPE"],
            "NULL_VAL": 0.0,
        },
    }

    save_memmap_array(output_dir / "data.dat", data_with_features)
    save_description(output_dir / "desc.json", description)
    save_split_indices(output_dir / "split_indices.npz", train_idx, val_idx, test_idx)
    save_basicts_adjacency(adjacency, output_dir / "adj_mx.pkl", node_ids=node_ids)
    np.save(output_dir / "node_ids.npy", np.asarray(node_ids, dtype=object))
    np.save(output_dir / "unix_timestamps.npy", unix_timestamps)

    print(f"Prepared BasicTS dataset at {output_dir}")
    print(
        f"shape={tuple(data_with_features.shape)} train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} edges={int(np.count_nonzero(adjacency))}"
    )


if __name__ == "__main__":
    main()

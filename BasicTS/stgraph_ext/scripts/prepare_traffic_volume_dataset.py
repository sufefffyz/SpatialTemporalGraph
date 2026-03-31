import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from stgraph_ext.data_utils import (
    build_multichannel_series,
    infer_frequency_minutes,
    resolve_dataset_dir,
    save_basicts_adjacency,
    save_description,
    save_memmap_array,
    save_split_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the UTB traffic volume NPZ file into a BasicTS-compatible dataset bundle."
    )
    parser.add_argument(
        "--input-npz",
        default="../data/city_traffic_m_volume__category__1_0.npz",
        help="Path to the UTB NPZ file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="TRAFFIC_VOLUME_5MIN",
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


def coerce_split_indices(values: np.ndarray, total_len: int, unix_timestamps: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if len(arr) == 0:
        return np.asarray([], dtype=np.int64)
    if np.issubdtype(arr.dtype, np.integer) and int(arr.min()) >= 0 and int(arr.max()) < total_len:
        return arr.astype(np.int64)
    timestamp_to_index = {int(ts): idx for idx, ts in enumerate(np.asarray(unix_timestamps, dtype=np.int64).tolist())}
    return np.asarray([timestamp_to_index[int(value)] for value in arr.tolist()], dtype=np.int64)


def impute_targets(targets: np.ndarray, dt_index: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.DataFrame(np.asarray(targets, dtype=np.float32), index=dt_index)
    frame = frame.interpolate(limit_direction="both")
    frame = frame.fillna(0.0)
    return frame.to_numpy(dtype=np.float32)


def build_adjacency(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in np.asarray(edges, dtype=np.int64):
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            adjacency[source, target] = 1.0
    return adjacency


def main():
    args = parse_args()
    input_path = Path(args.input_npz)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input NPZ file not found: {input_path}")

    output_dir = resolve_dataset_dir(args.dataset_name, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=True) as data:
        raw_targets = np.asarray(data["targets"], dtype=np.float32)
        unix_timestamps = np.asarray(data["unix_timestamps"], dtype=np.int64)
        num_nodes = int(data["num_nodes"]) if "num_nodes" in data else raw_targets.shape[1]
        edges = np.asarray(data["edges"], dtype=np.int64)
        node_ids = (
            np.asarray(data["node_ids_in_original_graph"])
            if "node_ids_in_original_graph" in data
            else np.arange(num_nodes, dtype=np.int64)
        )
        train_idx = coerce_split_indices(data["train_timestamps"], raw_targets.shape[0], unix_timestamps)
        val_idx = coerce_split_indices(data["val_timestamps"], raw_targets.shape[0], unix_timestamps)
        test_idx = coerce_split_indices(data["test_timestamps"], raw_targets.shape[0], unix_timestamps)

    if raw_targets.ndim != 2:
        raise ValueError(f"Expected traffic targets with shape [L, N], got {raw_targets.shape}.")

    dt_index = pd.to_datetime(unix_timestamps, unit="s", utc=True)
    targets = impute_targets(raw_targets, dt_index)
    data_with_features, feature_description = build_multichannel_series(targets, dt_index)
    adjacency = build_adjacency(num_nodes, edges)

    total_len = int(data_with_features.shape[0])
    regular_settings = {
        "INPUT_LEN": int(args.input_len),
        "OUTPUT_LEN": int(args.output_len),
        "TRAIN_VAL_TEST_RATIO": [
            float(len(train_idx) / total_len),
            float(len(val_idx) / total_len),
            float(len(test_idx) / total_len),
        ],
        "NORM_EACH_CHANNEL": False,
        "RESCALE": True,
        "METRICS": ["MAE", "RMSE", "MAPE", "WAPE"],
        "NULL_VAL": 0.0,
    }
    description = {
        "name": args.dataset_name,
        "domain": "urban traffic volume",
        "shape": list(data_with_features.shape),
        "num_time_steps": int(data_with_features.shape[0]),
        "num_nodes": int(data_with_features.shape[1]),
        "num_features": int(data_with_features.shape[2]),
        "feature_description": feature_description,
        "has_graph": True,
        "frequency (minutes)": infer_frequency_minutes(dt_index, fallback_minutes=5),
        "regular_settings": regular_settings,
    }

    save_memmap_array(output_dir / "data.dat", data_with_features)
    save_description(output_dir / "desc.json", description)
    save_split_indices(output_dir / "split_indices.npz", train_idx, val_idx, test_idx)
    save_basicts_adjacency(adjacency, output_dir / "adj_mx.pkl", node_ids=node_ids.tolist())
    np.save(output_dir / "node_ids.npy", np.asarray(node_ids))
    np.save(output_dir / "unix_timestamps.npy", unix_timestamps)

    print(f"Prepared BasicTS dataset at {output_dir}")
    print(
        f"shape={tuple(data_with_features.shape)} train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} edges={int(np.count_nonzero(adjacency))}"
    )


if __name__ == "__main__":
    main()

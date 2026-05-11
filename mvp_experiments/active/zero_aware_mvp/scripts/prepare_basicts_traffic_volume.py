#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
DEFAULT_FULL_VOLUME_NPZ = Path("/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a BasicTS traffic-volume dataset without importing torch.")
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_FULL_VOLUME_NPZ)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument(
        "--base-name",
        default="TRAFFIC_VOLUME_FULL_5MIN",
        help="Base dataset name used when --dataset-name is omitted.",
    )
    parser.add_argument("--output-root", type=Path, default=BASICTS_ROOT / "datasets")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--resolution", default="5min")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--adj-mode",
        choices=["none", "dense"],
        default="none",
        help="Use 'dense' only for small smoke subgraphs. Full city-M dense adjacency is very large.",
    )
    return parser.parse_args()


def parse_resolution_to_minutes(text: str | None) -> int | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    value = (
        value.replace("minutes", "min")
        .replace("minute", "min")
        .replace("mins", "min")
        .replace("hours", "h")
        .replace("hour", "h")
    )
    match = re.fullmatch(r"(\d+)\s*(min|h)", value)
    if not match:
        raise ValueError(f"Unsupported resolution: {text}")
    number = int(match.group(1))
    unit = match.group(2)
    return number * 60 if unit == "h" else number


def resolution_label(minutes: int) -> str:
    return f"{minutes // 60}H" if minutes % 60 == 0 else f"{minutes}MIN"


def dataset_name_with_resolution(base_name: str, minutes: int) -> str:
    label = resolution_label(minutes)
    if re.search(r"_(\d+MIN|\d+H)$", base_name):
        return re.sub(r"_(\d+MIN|\d+H)$", f"_{label}", base_name)
    return f"{base_name}_{label}"


def infer_frequency_minutes(unix_timestamps: np.ndarray, fallback: int = 5) -> int:
    diffs = np.diff(np.asarray(unix_timestamps, dtype=np.int64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    return max(1, int(round(float(np.median(diffs)) / 60.0)))


def coerce_split_indices(values: np.ndarray, total_len: int, unix_timestamps: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if len(arr) == 0:
        return np.asarray([], dtype=np.int64)
    if np.issubdtype(arr.dtype, np.integer) and int(arr.min()) >= 0 and int(arr.max()) < total_len:
        return arr.astype(np.int64)
    timestamp_to_index = {int(ts): idx for idx, ts in enumerate(np.asarray(unix_timestamps, dtype=np.int64).tolist())}
    return np.asarray([timestamp_to_index[int(value)] for value in arr.tolist()], dtype=np.int64)


def build_adjacency(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in np.asarray(edges, dtype=np.int64):
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            adjacency[source, target] = 1.0
    return adjacency


def aggregate_sum_with_splits(
    targets: np.ndarray,
    unix_timestamps: np.ndarray,
    splits: dict[str, np.ndarray],
    target_minutes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    bucket_seconds = target_minutes * 60
    bucket_starts = (unix_timestamps // bucket_seconds) * bucket_seconds
    split_codes = np.full(len(unix_timestamps), -1, dtype=np.int8)
    split_codes[splits["train"]] = 0
    split_codes[splits["val"]] = 1
    split_codes[splits["test"]] = 2

    aggregated_rows = []
    aggregated_timestamps = []
    labels = []
    dropped_mixed = 0
    dropped_unassigned = 0
    unique_buckets, first_indices = np.unique(bucket_starts, return_index=True)
    order = np.argsort(first_indices)
    for bucket in unique_buckets[order]:
        idx = np.flatnonzero(bucket_starts == bucket)
        unique_labels = np.unique(split_codes[idx])
        if len(unique_labels) != 1:
            dropped_mixed += 1
            continue
        label = int(unique_labels[0])
        if label < 0:
            dropped_unassigned += 1
            continue
        aggregated_rows.append(np.sum(targets[idx], axis=0, dtype=np.float32))
        aggregated_timestamps.append(int(bucket))
        labels.append(label)

    aggregated = np.stack(aggregated_rows, axis=0).astype(np.float32)
    labels_np = np.asarray(labels, dtype=np.int8)
    new_splits = {
        "train": np.where(labels_np == 0)[0].astype(np.int64),
        "val": np.where(labels_np == 1)[0].astype(np.int64),
        "test": np.where(labels_np == 2)[0].astype(np.int64),
    }
    info = {"dropped_mixed_buckets": int(dropped_mixed), "dropped_unassigned_buckets": int(dropped_unassigned)}
    return aggregated, np.asarray(aggregated_timestamps, dtype=np.int64), new_splits, info


def save_data_dat(path: Path, targets: np.ndarray, unix_timestamps: np.ndarray, chunk_size: int) -> None:
    total_len, num_nodes = targets.shape
    out = np.memmap(path, dtype="float32", mode="w+", shape=(total_len, num_nodes, 3))
    tod = ((unix_timestamps % 86400).astype(np.float32) / 86400.0)
    dow = (((unix_timestamps // 86400 + 3) % 7).astype(np.float32) / 7.0)
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        out[start:end, :, 0] = targets[start:end].astype(np.float32)
        out[start:end, :, 1] = tod[start:end, None]
        out[start:end, :, 2] = dow[start:end, None]
    out.flush()
    del out


def save_adj(path: Path, adjacency: np.ndarray, node_ids: np.ndarray) -> None:
    sensor_ids = [str(x) for x in node_ids.tolist()]
    sensor_id_to_ind = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    with path.open("wb") as fp:
        pickle.dump((sensor_ids, sensor_id_to_ind, adjacency.astype(np.float32)), fp)


def main() -> None:
    args = parse_args()
    input_npz = args.input_npz if args.input_npz.is_absolute() else (REPO_ROOT / args.input_npz)
    if not input_npz.exists():
        raise FileNotFoundError(
            f"Input NPZ not found: {input_npz}. Mount or copy the full Urban Traffic Benchmark file, "
            "or pass INPUT_NPZ=/path/to/city_traffic_m_volume.npz."
        )
    target_minutes = parse_resolution_to_minutes(args.resolution)
    if target_minutes is None:
        raise ValueError("--resolution is required")

    with np.load(input_npz, allow_pickle=True) as data:
        raw_targets = np.asarray(data["targets"], dtype=np.float32)
        unix_timestamps = np.asarray(data["unix_timestamps"], dtype=np.int64)
        num_nodes = int(data["num_nodes"]) if "num_nodes" in data else raw_targets.shape[1]
        edges = np.asarray(data["edges"], dtype=np.int64)
        node_ids = (
            np.asarray(data["node_ids_in_original_graph"])
            if "node_ids_in_original_graph" in data
            else np.arange(num_nodes, dtype=np.int64)
        )
        splits = {
            "train": coerce_split_indices(data["train_timestamps"], raw_targets.shape[0], unix_timestamps),
            "val": coerce_split_indices(data["val_timestamps"], raw_targets.shape[0], unix_timestamps),
            "test": coerce_split_indices(data["test_timestamps"], raw_targets.shape[0], unix_timestamps),
        }

    if np.isnan(raw_targets).any():
        raise ValueError("NaN targets found. Add imputation before preparing BasicTS data.")

    native_minutes = infer_frequency_minutes(unix_timestamps)
    if target_minutes < native_minutes or target_minutes % native_minutes:
        raise ValueError(f"Resolution {target_minutes}min must be a multiple of native {native_minutes}min.")

    if target_minutes == native_minutes:
        targets = raw_targets
        final_timestamps = unix_timestamps
        final_splits = splits
        aggregation_info = {"applied": False, "dropped_mixed_buckets": 0, "dropped_unassigned_buckets": 0}
    else:
        targets, final_timestamps, final_splits, aggregation_info = aggregate_sum_with_splits(
            raw_targets, unix_timestamps, splits, target_minutes
        )
        aggregation_info = {"applied": True, "resolution_minutes": target_minutes, **aggregation_info}

    dataset_name = args.dataset_name or dataset_name_with_resolution(args.base_name, target_minutes)
    output_root = args.output_root if args.output_root.is_absolute() else (REPO_ROOT / args.output_root)
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data_dat(output_dir / "data.dat", targets, final_timestamps, args.chunk_size)
    np.save(output_dir / "edges.npy", edges.astype(np.int64))
    adjacency_edges = int(len(edges))
    if args.adj_mode == "dense":
        adjacency = build_adjacency(num_nodes, edges)
        save_adj(output_dir / "adj_mx.pkl", adjacency, node_ids)
        adjacency_edges = int(np.count_nonzero(adjacency))
    np.save(output_dir / "node_ids.npy", node_ids)
    np.save(output_dir / "unix_timestamps.npy", final_timestamps)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=final_splits["train"],
        val_idx=final_splits["val"],
        test_idx=final_splits["test"],
    )

    total_len = int(targets.shape[0])
    desc = {
        "name": dataset_name,
        "domain": "urban observed traffic volume",
        "shape": [total_len, int(targets.shape[1]), 3],
        "num_time_steps": total_len,
        "num_nodes": int(targets.shape[1]),
        "num_features": 3,
        "feature_description": ["traffic volume", "time of day", "day of week"],
        "has_graph": True,
        "graph_storage": {
            "edges_npy": "edges.npy",
            "adj_mx_pkl": "adj_mx.pkl" if args.adj_mode == "dense" else None,
            "adj_mode": args.adj_mode,
        },
        "frequency (minutes)": target_minutes,
        "source_frequency (minutes)": native_minutes,
        "aggregation": aggregation_info,
        "regular_settings": {
            "INPUT_LEN": int(args.input_len),
            "OUTPUT_LEN": int(args.output_len),
            "TRAIN_VAL_TEST_RATIO": [
                float(len(final_splits["train"]) / total_len),
                float(len(final_splits["val"]) / total_len),
                float(len(final_splits["test"]) / total_len),
            ],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "WAPE"],
            "NULL_VAL": 0.0,
        },
    }
    (output_dir / "desc.json").write_text(json.dumps(desc, indent=4), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(output_dir),
                "shape": desc["shape"],
                "frequency_minutes": target_minutes,
                "train": int(len(final_splits["train"])),
                "val": int(len(final_splits["val"])),
                "test": int(len(final_splits["test"])),
                "edges": adjacency_edges,
                "aggregation": aggregation_info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

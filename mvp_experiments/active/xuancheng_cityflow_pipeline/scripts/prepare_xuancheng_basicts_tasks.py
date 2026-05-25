#!/usr/bin/env python3
"""Prepare Xuancheng 10s/5min flow and road-stock prediction tasks for BasicTS."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


TASKS = {
    "flow": {
        "description": "road-level through flow count in the interval",
        "source": "sum(movement_volume_lsr over L/S/R) plus movement_unknown when present",
        "aggregation": "sum",
    },
    "stock": {
        "description": "mean vehicles currently on the road during the interval",
        "source": "sum(active_volume_lsr over L/S/R) plus active_unknown when present",
        "aggregation": "mean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        required=True,
        help="Xuancheng DTIGNN-style NPZ from run_cityflow_dtignn_generation.py.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="BasicTS datasets directory, e.g. /path/to/BasicTS/datasets.",
    )
    parser.add_argument("--prefix", default="XCHENG")
    parser.add_argument(
        "--tasks",
        default="flow,stock",
        help="Comma-separated tasks from {flow,stock}.",
    )
    parser.add_argument(
        "--resolutions",
        default="10s,5min",
        help="Comma-separated resolutions. Supported: 10s and integer minutes like 5min.",
    )
    parser.add_argument("--split-ratios", default="0.6,0.2,0.2")
    parser.add_argument("--input-len-10s", type=int, default=30)
    parser.add_argument("--output-len-10s", type=int, default=1)
    parser.add_argument("--input-len-5min", type=int, default=12)
    parser.add_argument("--output-len-5min", type=int, default=12)
    parser.add_argument(
        "--norm-each-channel",
        action="store_true",
        help="Set BasicTS NORM_EACH_CHANNEL=true in desc.json.",
    )
    return parser.parse_args()


def parse_split_ratios(text: str) -> list[float]:
    ratios = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(ratios) != 3:
        raise ValueError(f"expected 3 split ratios, got {ratios}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {sum(ratios)}")
    return ratios


def parse_resolution(value: str, base_bucket_seconds: int) -> tuple[str, int, float]:
    text = value.strip().lower()
    if text.endswith("s"):
        seconds = int(text[:-1])
    elif text.endswith("min"):
        seconds = int(text[:-3]) * 60
    else:
        raise ValueError(f"unsupported resolution {value!r}; use 10s or 5min")
    if seconds <= 0 or seconds % base_bucket_seconds != 0:
        raise ValueError(
            f"resolution {value!r} must be a positive multiple of base bucket {base_bucket_seconds}s"
        )
    label = f"{seconds}s" if seconds < 60 else f"{seconds // 60}MIN"
    return label.upper(), seconds // base_bucket_seconds, seconds / 60.0


def make_adjacency(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    n_roads = len(archive["road_ids"])
    adj = np.zeros((n_roads, n_roads), dtype=np.float32)
    starts = np.asarray(archive["edge_start_road_idx"], dtype=np.int64)
    ends = np.asarray(archive["edge_end_road_idx"], dtype=np.int64)
    valid = (starts >= 0) & (starts < n_roads) & (ends >= 0) & (ends < n_roads)
    adj[starts[valid], ends[valid]] = 1.0
    return adj


def road_scalar_series(archive: np.lib.npyio.NpzFile, task: str) -> np.ndarray:
    if task == "flow":
        series = np.asarray(archive["movement_volume_lsr"], dtype=np.float32).sum(axis=2)
        if "movement_unknown" in archive:
            series = series + np.asarray(archive["movement_unknown"], dtype=np.float32)
        return series
    if task == "stock":
        series = np.asarray(archive["active_volume_lsr"], dtype=np.float32).sum(axis=2)
        if "active_unknown" in archive:
            series = series + np.asarray(archive["active_unknown"], dtype=np.float32)
        return series
    raise ValueError(f"unknown task: {task}")


def resample_series(series: np.ndarray, factor: int, task: str) -> np.ndarray:
    if factor == 1:
        return series[:, :, None].astype(np.float32, copy=False)
    usable = (series.shape[0] // factor) * factor
    grouped = series[:usable].reshape(usable // factor, factor, series.shape[1])
    if task == "flow":
        out = grouped.sum(axis=1)
    else:
        out = grouped.mean(axis=1)
    return out[:, :, None].astype(np.float32, copy=False)


def save_memmap(path: Path, data: np.ndarray) -> None:
    fp = np.memmap(path, dtype="float32", mode="w+", shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")


def write_meta_csv(path: Path, archive: np.lib.npyio.NpzFile) -> None:
    road_ids = np.asarray(archive["road_ids"]).astype(str)
    lane_count = np.asarray(
        archive["road_lane_count"] if "road_lane_count" in archive.files else np.zeros(len(road_ids)),
        dtype=np.float32,
    )
    length_m = np.asarray(
        archive["road_length_m"] if "road_length_m" in archive.files else np.zeros(len(road_ids)),
        dtype=np.float32,
    )
    speed_limit = np.asarray(
        archive["road_speed_limit_kmh"] if "road_speed_limit_kmh" in archive.files else np.zeros(len(road_ids)),
        dtype=np.float32,
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "road_id", "lane_count", "length_m", "speed_limit_kmh"])
        for idx, road_id in enumerate(road_ids):
            writer.writerow([idx, road_id, lane_count[idx], length_m[idx], speed_limit[idx]])


def dataset_name(prefix: str, resolution_label: str, task: str) -> str:
    return f"{prefix}_{resolution_label}_{task.upper()}"


def main() -> int:
    args = parse_args()
    input_npz = Path(args.input_npz).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    archive = np.load(input_npz, allow_pickle=True)
    base_bucket_seconds = int(np.asarray(archive["bucket_seconds"]).item())
    split_ratios = parse_split_ratios(args.split_ratios)
    tasks = [part.strip().lower() for part in args.tasks.split(",") if part.strip()]
    resolutions = [part.strip().lower() for part in args.resolutions.split(",") if part.strip()]
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unsupported task {task!r}; choose from {sorted(TASKS)}")

    road_ids = np.asarray(archive["road_ids"]).astype(str)
    adj = make_adjacency(archive)
    road_id_to_idx = {road_id: idx for idx, road_id in enumerate(road_ids)}

    all_summary: dict[str, dict] = {}
    for task in tasks:
        base_series = road_scalar_series(archive, task)
        for res in resolutions:
            resolution_label, factor, frequency_minutes = parse_resolution(res, base_bucket_seconds)
            data = resample_series(base_series, factor, task)
            name = dataset_name(args.prefix, resolution_label, task)
            out_dir = output_root / name
            out_dir.mkdir(parents=True, exist_ok=True)

            save_memmap(out_dir / "data.dat", data)
            with (out_dir / "adj_mx.pkl").open("wb") as f:
                # Store adjacency as nested Python lists for NumPy 1/2 pickle compatibility
                # across the CityFlow generation env and the STGraph training env.
                pickle.dump((road_ids.tolist(), road_id_to_idx, adj.tolist()), f, protocol=pickle.HIGHEST_PROTOCOL)
            write_meta_csv(out_dir / "meta.csv", archive)

            input_len = args.input_len_10s if resolution_label == "10S" else args.input_len_5min
            output_len = args.output_len_10s if resolution_label == "10S" else args.output_len_5min
            target = data[:, :, 0]
            positive = target[target > 0]
            desc = {
                "name": name,
                "domain": "xuancheng simulated road traffic",
                "shape": list(data.shape),
                "num_time_steps": int(data.shape[0]),
                "num_nodes": int(data.shape[1]),
                "num_features": int(data.shape[2]),
                "feature_description": [TASKS[task]["description"]],
                "has_graph": True,
                "frequency (minutes)": frequency_minutes,
                "source_bucket_seconds": base_bucket_seconds,
                "source_npz": str(input_npz),
                "target_task": task,
                "target_source": TASKS[task]["source"],
                "target_aggregation": TASKS[task]["aggregation"],
                "regular_settings": {
                    "INPUT_LEN": input_len,
                    "OUTPUT_LEN": output_len,
                    "TRAIN_VAL_TEST_RATIO": split_ratios,
                    "NORM_EACH_CHANNEL": bool(args.norm_each_channel),
                    "RESCALE": True,
                    "METRICS": ["MAE", "RMSE", "MAPE", "WAPE"],
                    "NULL_VAL": "nan",
                },
            }
            stats = {
                "dataset": name,
                "task": task,
                "resolution": resolution_label,
                "shape": list(data.shape),
                "input_len": input_len,
                "output_len": output_len,
                "zero_rate": float((target == 0).mean()),
                "mean": float(target.mean()),
                "std": float(target.std()),
                "positive_mean": float(positive.mean()) if positive.size else 0.0,
                "positive_count": int(positive.size),
                "total_sum": float(target.sum()),
                "adj_edges": int(adj.sum()),
                "files": {
                    "data_dat": str(out_dir / "data.dat"),
                    "desc_json": str(out_dir / "desc.json"),
                    "adj_mx_pkl": str(out_dir / "adj_mx.pkl"),
                    "meta_csv": str(out_dir / "meta.csv"),
                    "stats_json": str(out_dir / "stats.json"),
                },
            }
            write_json(out_dir / "desc.json", desc)
            write_json(out_dir / "stats.json", stats)
            all_summary[name] = stats
            print(
                f"[done] {name}: shape={data.shape} zero={stats['zero_rate']:.6f} "
                f"mean={stats['mean']:.6f} positive_mean={stats['positive_mean']:.6f}"
            )

    write_json(output_root / f"{args.prefix}_task_summary.json", all_summary)
    print(f"[done] wrote {output_root / f'{args.prefix}_task_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare 30-day Xuancheng road-aggregation 5min BasicTS datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


TASKS = {
    "flow": {
        "feature": "entered_veh",
        "description": "road-level vehicles entering the road link in the interval",
        "aggregation": "sum",
    },
    "stock": {
        "feature": "mean_active_veh",
        "description": "mean vehicles currently on the road link during the interval",
        "aggregation": "mean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory with daily road_agg_60s NPZ files.")
    parser.add_argument("--roadnet", required=True, help="CityFlow roadnet JSON for graph and road metadata.")
    parser.add_argument("--output-root", required=True, help="BasicTS datasets directory.")
    parser.add_argument("--prefix", default="XCHENG30D")
    parser.add_argument("--pattern", default="xuancheng_*_road_agg_60s.npz")
    parser.add_argument("--tasks", default="flow,stock")
    parser.add_argument("--bucket-seconds", type=int, default=60)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--split-ratios", default="0.6,0.2,0.2")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--norm-each-channel", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")


def parse_split_ratios(text: str) -> list[float]:
    ratios = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(ratios) != 3:
        raise ValueError(f"expected 3 split ratios, got {ratios}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {sum(ratios)}")
    return ratios


def infer_date(path: Path, archive: np.lib.npyio.NpzFile) -> str:
    if "date" in archive.files:
        value = archive["date"]
        if value.shape == ():
            return str(value.tolist())
    match = re.search(r"xuancheng_(\d{4}-\d{2}-\d{2})_", path.name)
    if not match:
        raise ValueError(f"cannot infer date from {path}")
    return match.group(1)


def road_length_m(points: list[dict[str, Any]]) -> float:
    total = 0.0
    for p0, p1 in zip(points, points[1:]):
        total += math.hypot(float(p0["x"]) - float(p1["x"]), float(p0["y"]) - float(p1["y"]))
    return total


def load_roadnet(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_adjacency(roadnet: dict[str, Any], road_ids: np.ndarray) -> np.ndarray:
    road_to_idx = {str(road_id): idx for idx, road_id in enumerate(road_ids.astype(str))}
    adj = np.zeros((len(road_ids), len(road_ids)), dtype=np.float32)
    for intersection in roadnet.get("intersections", []):
        for link in intersection.get("roadLinks", []):
            start = road_to_idx.get(str(link.get("startRoad")))
            end = road_to_idx.get(str(link.get("endRoad")))
            if start is not None and end is not None:
                adj[start, end] = 1.0
    return adj


def road_metadata(roadnet: dict[str, Any], road_ids: np.ndarray) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for road in roadnet.get("roads", []):
        road_id = str(road["id"])
        lanes = road.get("lanes", [])
        lane_count = len(lanes)
        length_m = road_length_m(road.get("points", []))
        speed_limit = 0.0
        if lane_count:
            speed_limit = sum(float(lane.get("maxSpeed", 0.0)) for lane in lanes) / lane_count * 3.6
        rows[road_id] = {
            "lane_count": float(lane_count),
            "length_m": float(length_m),
            "speed_limit_kmh": float(speed_limit),
        }
    return {
        str(road_id): rows.get(str(road_id), {"lane_count": 0.0, "length_m": 0.0, "speed_limit_kmh": 0.0})
        for road_id in road_ids.astype(str)
    }


def write_meta_csv(path: Path, road_ids: np.ndarray, meta: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "road_id", "lane_count", "length_m", "speed_limit_kmh"])
        for idx, road_id_value in enumerate(road_ids.astype(str)):
            item = meta[road_id_value]
            writer.writerow(
                [
                    idx,
                    road_id_value,
                    item["lane_count"],
                    item["length_m"],
                    item["speed_limit_kmh"],
                ]
            )


def validate_daily_file(
    path: Path,
    reference_road_ids: np.ndarray | None,
    reference_features: list[str] | None,
) -> tuple[np.ndarray, list[str], str]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"data", "road_ids", "feature_names", "bucket_start_s", "bucket_end_s"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
        road_ids = archive["road_ids"].astype(str)
        features = [str(value) for value in archive["feature_names"].tolist()]
        if reference_road_ids is not None and road_ids.tolist() != reference_road_ids.astype(str).tolist():
            raise ValueError(f"road_ids mismatch in {path}")
        if reference_features is not None and features != reference_features:
            raise ValueError(f"feature_names mismatch in {path}")
        return road_ids, features, infer_date(path, archive)


def resample_daily(path: Path, feature_idx: int, factor: int, aggregation: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        series = archive["data"][:, :, feature_idx].astype(np.float32, copy=False)
        usable = (series.shape[0] // factor) * factor
        grouped = series[:usable].reshape(usable // factor, factor, series.shape[1])
        if aggregation == "sum":
            return grouped.sum(axis=1, dtype=np.float32)
        if aggregation == "mean":
            return grouped.mean(axis=1, dtype=np.float32)
        raise ValueError(f"unsupported aggregation {aggregation}")


def save_memmap(path: Path, data: np.ndarray) -> None:
    fp = np.memmap(path, dtype="float32", mode="w+", shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp


def dataset_name(prefix: str, task: str, resolution_minutes: int) -> str:
    return f"{prefix}_{resolution_minutes}MIN_{task.upper()}"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    roadnet_path = Path(args.roadnet).expanduser().resolve()
    daily_paths = sorted(input_dir.glob(args.pattern))
    if not daily_paths:
        raise SystemExit(f"no files matched {input_dir / args.pattern}")
    factor = args.resolution_minutes * 60 // args.bucket_seconds
    if factor <= 0 or args.resolution_minutes * 60 % args.bucket_seconds != 0:
        raise SystemExit("--resolution-minutes must be a positive multiple of --bucket-seconds")

    tasks = [part.strip().lower() for part in args.tasks.split(",") if part.strip()]
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unsupported task {task}; choose from {sorted(TASKS)}")
    split_ratios = parse_split_ratios(args.split_ratios)

    reference_road_ids = None
    reference_features = None
    dates: list[str] = []
    for path in daily_paths:
        reference_road_ids, reference_features, date = validate_daily_file(path, reference_road_ids, reference_features)
        dates.append(date)
    assert reference_road_ids is not None and reference_features is not None

    roadnet = load_roadnet(roadnet_path)
    adj = make_adjacency(roadnet, reference_road_ids)
    meta = road_metadata(roadnet, reference_road_ids)
    road_id_to_idx = {str(road_id): int(idx) for idx, road_id in enumerate(reference_road_ids.astype(str))}

    summary: dict[str, Any] = {}
    for task in tasks:
        spec = TASKS[task]
        if spec["feature"] not in reference_features:
            raise ValueError(f"{spec['feature']} not in feature_names={reference_features}")
        feature_idx = reference_features.index(spec["feature"])
        chunks = [resample_daily(path, feature_idx, factor, spec["aggregation"]) for path in daily_paths]
        data2d = np.concatenate(chunks, axis=0)
        data = data2d[:, :, None].astype(np.float32, copy=False)

        name = dataset_name(args.prefix, task, args.resolution_minutes)
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_memmap(out_dir / "data.dat", data)
        with (out_dir / "adj_mx.pkl").open("wb") as f:
            pickle.dump((reference_road_ids.astype(str).tolist(), road_id_to_idx, adj.tolist()), f, protocol=pickle.HIGHEST_PROTOCOL)
        write_meta_csv(out_dir / "meta.csv", reference_road_ids, meta)

        target = data[:, :, 0]
        positive = target[target > 0]
        desc = {
            "name": name,
            "domain": "xuancheng simulated road traffic",
            "shape": list(data.shape),
            "num_time_steps": int(data.shape[0]),
            "num_nodes": int(data.shape[1]),
            "num_features": int(data.shape[2]),
            "feature_description": [spec["description"]],
            "has_graph": True,
            "frequency (minutes)": float(args.resolution_minutes),
            "source_bucket_seconds": args.bucket_seconds,
            "source_daily_npz_dir": str(input_dir),
            "source_pattern": args.pattern,
            "source_dates": dates,
            "target_task": task,
            "target_feature": spec["feature"],
            "target_aggregation": spec["aggregation"],
            "regular_settings": {
                "INPUT_LEN": args.input_len,
                "OUTPUT_LEN": args.output_len,
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
            "source_feature": spec["feature"],
            "resolution_minutes": args.resolution_minutes,
            "dates": dates,
            "shape": list(data.shape),
            "input_len": args.input_len,
            "output_len": args.output_len,
            "zero_rate": float((target == 0).mean()),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "positive_mean": float(positive.mean()) if positive.size else 0.0,
            "positive_count": int(positive.size),
            "total_sum": float(target.sum()),
            "adj_edges": int(adj.sum()),
        }
        write_json(out_dir / "desc.json", desc)
        write_json(out_dir / "stats.json", stats)
        summary[name] = stats
        print(
            f"[done] {name}: shape={data.shape} zero={stats['zero_rate']:.6f} "
            f"mean={stats['mean']:.6f} positive_mean={stats['positive_mean']:.6f}"
        )

    write_json(output_root / f"{args.prefix}_{args.resolution_minutes}MIN_task_summary.json", summary)
    print(f"[done] wrote {output_root / f'{args.prefix}_{args.resolution_minutes}MIN_task_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run CityFlow and aggregate per-lane simulation states to lane-level buckets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path


FEATURES = [
    "entered_veh",
    "exited_veh",
    "mean_active_veh",
    "mean_speed_kmh",
    "density_veh_per_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="CityFlow config JSON.")
    parser.add_argument("--date", required=True, help="Date label for outputs, e.g. 2023-04-03.")
    parser.add_argument("--output-dir", required=True, help="Directory for lane-level outputs.")
    parser.add_argument("--duration", type=int, default=86400, help="Simulation duration in seconds.")
    parser.add_argument("--bucket-seconds", type=int, default=60, help="Aggregation bucket width.")
    parser.add_argument("--thread-num", type=int, default=1, help="CityFlow thread_num.")
    parser.add_argument("--progress-interval", type=int, default=300, help="Progress print interval in seconds.")
    parser.add_argument("--write-csv", action="store_true", help="Also write dense lane CSV. Default: NPZ only.")
    parser.add_argument("--sparse-csv", action="store_true", help="With --write-csv, write only non-empty lane buckets.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_cityflow_path(config_path: Path, cfg: dict, key: str) -> Path:
    path = Path(cfg[key]).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(cfg.get("dir", ".")).expanduser()
    if not base.is_absolute():
        base = config_path.parent / base
    return (base / path).resolve()


def road_length_m(points: list[dict]) -> float:
    total = 0.0
    for p0, p1 in zip(points, points[1:]):
        dx = float(p0["x"]) - float(p1["x"])
        dy = float(p0["y"]) - float(p1["y"])
        total += math.hypot(dx, dy)
    return total


def load_lane_metadata(roadnet_file: Path) -> tuple[list[str], list[dict]]:
    with roadnet_file.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)

    lane_ids: list[str] = []
    lane_meta: list[dict] = []
    for road in roadnet["roads"]:
        road_id = str(road["id"])
        length_m = road_length_m(road.get("points", []))
        for lane_idx, lane in enumerate(road.get("lanes", [])):
            lane_id = f"{road_id}_{lane_idx}"
            lane_ids.append(lane_id)
            lane_meta.append(
                {
                    "lane_id": lane_id,
                    "road_id": road_id,
                    "lane_index": lane_idx,
                    "length_m": length_m,
                    "lane_km": max(length_m / 1000.0, 0.0),
                    "speed_limit_kmh": float(lane.get("maxSpeed", 0.0)) * 3.6,
                }
            )
    return lane_ids, lane_meta


def open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", newline="", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.bucket_seconds <= 0:
        raise SystemExit("--bucket-seconds must be positive")

    try:
        import cityflow  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("cityflow is required") from exc

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("numpy is required for lane-level NPZ output") from exc

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    interval = float(cfg.get("interval", 1.0))
    if interval <= 0:
        raise SystemExit(f"invalid CityFlow interval in config: {interval}")

    roadnet_path = resolve_cityflow_path(config_path, cfg, "roadnetFile")
    lane_ids, lane_meta = load_lane_metadata(roadnet_path)
    lane_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    n_lanes = len(lane_ids)
    expected_buckets = int(math.ceil(args.duration / args.bucket_seconds))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"xuancheng_{args.date}_lane_agg_{args.bucket_seconds}s.npz"
    csv_path = output_dir / f"xuancheng_{args.date}_lane_agg_{args.bucket_seconds}s.csv.gz"

    tensor = np.zeros((expected_buckets, n_lanes, len(FEATURES)), dtype=np.float32)
    bucket_start_s = np.zeros((expected_buckets,), dtype=np.float32)
    bucket_end_s = np.zeros((expected_buckets,), dtype=np.float32)

    print(f"config={config_path}")
    print(f"roadnet={roadnet_path}")
    print(f"lanes={n_lanes}")
    print(f"duration={args.duration}s, bucket={args.bucket_seconds}s, interval={interval}s")
    print(f"npz={npz_path}")
    if args.write_csv:
        print(f"csv={csv_path}")

    eng = cityflow.Engine(config_file=str(config_path), thread_num=args.thread_num)

    initial_lane_vehicles = eng.get_lane_vehicles()
    prev_lane_vehicles = {
        lane_id: set(initial_lane_vehicles.get(lane_id, [])) for lane_id in lane_ids
    }

    entered = np.zeros((n_lanes,), dtype=np.float64)
    exited = np.zeros((n_lanes,), dtype=np.float64)
    active_vehicle_seconds = np.zeros((n_lanes,), dtype=np.float64)
    speed_meter_seconds = np.zeros((n_lanes,), dtype=np.float64)
    speed_observation_seconds = np.zeros((n_lanes,), dtype=np.float64)

    csv_file = open_csv(csv_path) if args.write_csv else None
    writer = None
    if csv_file is not None:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "date",
                "bucket_index",
                "time_start_s",
                "time_end_s",
                "lane_id",
                "road_id",
                "lane_index",
                "entered_veh",
                "exited_veh",
                "mean_active_veh",
                "mean_speed_kmh",
                "density_veh_per_km",
                "vehicle_seconds",
                "lane_length_m",
                "speed_limit_kmh",
            ]
        )

    bucket_idx = 0
    bucket_start = 0.0
    bucket_elapsed = 0.0
    current_time = 0.0

    def flush_bucket(time_end: float) -> None:
        nonlocal bucket_idx, bucket_start, bucket_elapsed
        if bucket_idx >= expected_buckets:
            return
        elapsed = max(bucket_elapsed, interval)
        bucket_start_s[bucket_idx] = bucket_start
        bucket_end_s[bucket_idx] = time_end
        mean_active = active_vehicle_seconds / elapsed
        mean_speed_kmh = np.divide(
            speed_meter_seconds,
            speed_observation_seconds,
            out=np.zeros_like(speed_meter_seconds),
            where=speed_observation_seconds > 0,
        ) * 3.6
        lane_km = np.array([meta["lane_km"] for meta in lane_meta], dtype=np.float64)
        density = np.divide(mean_active, lane_km, out=np.zeros_like(mean_active), where=lane_km > 0)

        tensor[bucket_idx, :, 0] = entered
        tensor[bucket_idx, :, 1] = exited
        tensor[bucket_idx, :, 2] = mean_active
        tensor[bucket_idx, :, 3] = mean_speed_kmh
        tensor[bucket_idx, :, 4] = density

        if writer is not None:
            for lane_idx, meta in enumerate(lane_meta):
                if (
                    args.sparse_csv
                    and entered[lane_idx] == 0
                    and exited[lane_idx] == 0
                    and active_vehicle_seconds[lane_idx] == 0
                ):
                    continue
                writer.writerow(
                    [
                        args.date,
                        bucket_idx,
                        f"{bucket_start:.3f}",
                        f"{time_end:.3f}",
                        meta["lane_id"],
                        meta["road_id"],
                        meta["lane_index"],
                        int(entered[lane_idx]),
                        int(exited[lane_idx]),
                        f"{mean_active[lane_idx]:.6f}",
                        f"{mean_speed_kmh[lane_idx]:.6f}",
                        f"{density[lane_idx]:.6f}",
                        f"{active_vehicle_seconds[lane_idx]:.6f}",
                        f"{meta['length_m']:.6f}",
                        f"{meta['speed_limit_kmh']:.6f}",
                    ]
                )

        bucket_idx += 1
        bucket_start = time_end
        bucket_elapsed = 0.0
        entered.fill(0)
        exited.fill(0)
        active_vehicle_seconds.fill(0)
        speed_meter_seconds.fill(0)
        speed_observation_seconds.fill(0)

    while current_time < args.duration and bucket_idx < expected_buckets:
        eng.next_step()
        current_time = float(eng.get_current_time())
        lane_vehicles = eng.get_lane_vehicles()
        vehicle_speed = eng.get_vehicle_speed()

        current_lane_vehicles = {
            lane_id: set(lane_vehicles.get(lane_id, [])) for lane_id in lane_ids
        }
        for lane_id, lane_idx in lane_to_idx.items():
            current_set = current_lane_vehicles[lane_id]
            previous_set = prev_lane_vehicles.get(lane_id, set())
            entered[lane_idx] += len(current_set - previous_set)
            exited[lane_idx] += len(previous_set - current_set)

            active_count = len(current_set)
            active_vehicle_seconds[lane_idx] += active_count * interval
            if active_count:
                for vehicle_id in current_set:
                    speed = vehicle_speed.get(vehicle_id)
                    if speed is not None and speed >= 0:
                        speed_meter_seconds[lane_idx] += float(speed) * interval
                        speed_observation_seconds[lane_idx] += interval

        prev_lane_vehicles = current_lane_vehicles
        bucket_elapsed += interval

        if current_time >= bucket_start + args.bucket_seconds or current_time >= args.duration:
            flush_bucket(min(current_time, float(args.duration)))

        if args.progress_interval > 0 and int(current_time) % args.progress_interval == 0:
            print(f"[progress] t={current_time:.0f}s buckets={bucket_idx}/{expected_buckets}")

    if csv_file is not None:
        csv_file.close()

    np.savez_compressed(
        npz_path,
        data=tensor[:bucket_idx],
        lane_ids=np.array(lane_ids),
        road_ids=np.array([meta["road_id"] for meta in lane_meta]),
        lane_indices=np.array([meta["lane_index"] for meta in lane_meta], dtype=np.int16),
        lane_length_m=np.array([meta["length_m"] for meta in lane_meta], dtype=np.float32),
        speed_limit_kmh=np.array([meta["speed_limit_kmh"] for meta in lane_meta], dtype=np.float32),
        feature_names=np.array(FEATURES),
        bucket_start_s=bucket_start_s[:bucket_idx],
        bucket_end_s=bucket_end_s[:bucket_idx],
        date=np.array(args.date),
        source_config=np.array(str(config_path)),
    )

    if args.write_csv:
        print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {npz_path} with shape {tensor[:bucket_idx].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

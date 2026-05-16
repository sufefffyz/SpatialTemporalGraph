#!/usr/bin/env python3
"""Run CityFlow and aggregate per-lane simulation states to road-level buckets."""

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
    "density_veh_per_lane_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="CityFlow config JSON.")
    parser.add_argument("--date", required=True, help="Date label for outputs, e.g. 2023-04-03.")
    parser.add_argument("--output-dir", required=True, help="Directory for csv.gz and npz outputs.")
    parser.add_argument("--duration", type=int, default=86400, help="Simulation duration in seconds.")
    parser.add_argument("--bucket-seconds", type=int, default=60, help="Aggregation bucket width.")
    parser.add_argument("--thread-num", type=int, default=1, help="CityFlow thread_num.")
    parser.add_argument("--no-npz", action="store_true", help="Skip compressed dense NPZ tensor.")
    parser.add_argument("--sparse-csv", action="store_true", help="Write only non-empty road buckets.")
    parser.add_argument("--progress-interval", type=int, default=300, help="Progress print interval in seconds.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def road_length_m(points: list[dict]) -> float:
    total = 0.0
    for p0, p1 in zip(points, points[1:]):
        dx = float(p0["x"]) - float(p1["x"])
        dy = float(p0["y"]) - float(p1["y"])
        total += math.hypot(dx, dy)
    return total


def load_roadnet(roadnet_file: Path) -> tuple[list[str], dict[str, int], list[dict]]:
    with roadnet_file.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)

    road_ids: list[str] = []
    lane_to_road_idx: dict[str, int] = {}
    road_meta: list[dict] = []

    for road in roadnet["roads"]:
        road_id = str(road["id"])
        road_idx = len(road_ids)
        lanes = road.get("lanes", [])
        lane_count = len(lanes)
        length_m = road_length_m(road.get("points", []))
        if lanes:
            speed_limit_mps = sum(float(lane.get("maxSpeed", 0.0)) for lane in lanes) / lane_count
        else:
            speed_limit_mps = 0.0
        road_ids.append(road_id)
        road_meta.append(
            {
                "road_id": road_id,
                "lane_count": lane_count,
                "length_m": length_m,
                "lane_km": max(length_m * lane_count / 1000.0, 0.0),
                "speed_limit_kmh": speed_limit_mps * 3.6,
            }
        )
        for lane_idx in range(lane_count):
            lane_to_road_idx[f"{road_id}_{lane_idx}"] = road_idx

    return road_ids, lane_to_road_idx, road_meta


def zero_vector(n: int) -> list[float]:
    return [0.0] * n


def int_zero_vector(n: int) -> list[int]:
    return [0] * n


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
        raise SystemExit(
            "cityflow is required to run simulation. Use the official Docker image "
            "kingsleycl/cityflow_env:latest or a Python environment with cityflow installed."
        ) from exc

    np = None
    if not args.no_npz:
        try:
            import numpy as np_import  # type: ignore

            np = np_import
        except ModuleNotFoundError as exc:
            raise SystemExit("numpy is required for NPZ output; pass --no-npz to skip it") from exc

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    interval = float(cfg.get("interval", 1.0))
    if interval <= 0:
        raise SystemExit(f"invalid CityFlow interval in config: {interval}")

    roadnet_path = Path(cfg["roadnetFile"]).expanduser().resolve()
    road_ids, lane_to_road_idx, road_meta = load_roadnet(roadnet_path)
    n_roads = len(road_ids)
    expected_buckets = int(math.ceil(args.duration / args.bucket_seconds))

    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_path = output_dir / f"xuancheng_{args.date}_road_agg_{args.bucket_seconds}s.csv.gz"
    npz_path = output_dir / f"xuancheng_{args.date}_road_agg_{args.bucket_seconds}s.npz"

    tensor = None
    if np is not None:
        tensor = np.zeros((expected_buckets, n_roads, len(FEATURES)), dtype=np.float32)
        bucket_start_s = np.zeros((expected_buckets,), dtype=np.float32)
        bucket_end_s = np.zeros((expected_buckets,), dtype=np.float32)
    else:
        bucket_start_s = None
        bucket_end_s = None

    print(f"config={config_path}")
    print(f"roadnet={roadnet_path}")
    print(f"roads={n_roads}, lanes={len(lane_to_road_idx)}")
    print(f"duration={args.duration}s, bucket={args.bucket_seconds}s, interval={interval}s")
    print(f"csv={csv_path}")
    if tensor is not None:
        print(f"npz={npz_path}")

    eng = cityflow.Engine(config_file=str(config_path), thread_num=args.thread_num)

    initial_lane_vehicles = eng.get_lane_vehicles()
    prev_lane_vehicles = {
        lane_id: set(initial_lane_vehicles.get(lane_id, [])) for lane_id in lane_to_road_idx
    }

    entered = int_zero_vector(n_roads)
    exited = int_zero_vector(n_roads)
    active_vehicle_seconds = zero_vector(n_roads)
    speed_meter_seconds = zero_vector(n_roads)
    speed_observation_seconds = zero_vector(n_roads)

    bucket_idx = 0
    bucket_start = 0.0
    bucket_elapsed = 0.0
    current_time = 0.0

    with open_csv(csv_path) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "date",
                "bucket_index",
                "time_start_s",
                "time_end_s",
                "road_id",
                "entered_veh",
                "exited_veh",
                "mean_active_veh",
                "mean_speed_kmh",
                "density_veh_per_lane_km",
                "vehicle_seconds",
                "road_length_m",
                "lane_count",
                "speed_limit_kmh",
            ]
        )

        def flush_bucket(time_end: float) -> None:
            nonlocal bucket_idx, bucket_start, bucket_elapsed
            if bucket_idx >= expected_buckets:
                return
            elapsed = max(bucket_elapsed, interval)
            if bucket_start_s is not None and bucket_end_s is not None:
                bucket_start_s[bucket_idx] = bucket_start
                bucket_end_s[bucket_idx] = time_end
            for road_idx, road_id in enumerate(road_ids):
                meta = road_meta[road_idx]
                veh_seconds = active_vehicle_seconds[road_idx]
                mean_active = veh_seconds / elapsed
                if speed_observation_seconds[road_idx] > 0:
                    mean_speed_kmh = (
                        speed_meter_seconds[road_idx] / speed_observation_seconds[road_idx] * 3.6
                    )
                else:
                    mean_speed_kmh = 0.0
                lane_km = meta["lane_km"]
                density = mean_active / lane_km if lane_km > 0 else 0.0
                if tensor is not None:
                    tensor[bucket_idx, road_idx, 0] = entered[road_idx]
                    tensor[bucket_idx, road_idx, 1] = exited[road_idx]
                    tensor[bucket_idx, road_idx, 2] = mean_active
                    tensor[bucket_idx, road_idx, 3] = mean_speed_kmh
                    tensor[bucket_idx, road_idx, 4] = density
                if args.sparse_csv and entered[road_idx] == 0 and exited[road_idx] == 0 and veh_seconds == 0:
                    continue
                writer.writerow(
                    [
                        args.date,
                        bucket_idx,
                        f"{bucket_start:.3f}",
                        f"{time_end:.3f}",
                        road_id,
                        entered[road_idx],
                        exited[road_idx],
                        f"{mean_active:.6f}",
                        f"{mean_speed_kmh:.6f}",
                        f"{density:.6f}",
                        f"{veh_seconds:.6f}",
                        f"{meta['length_m']:.6f}",
                        meta["lane_count"],
                        f"{meta['speed_limit_kmh']:.6f}",
                    ]
                )
            bucket_idx += 1
            bucket_start = time_end
            bucket_elapsed = 0.0
            entered[:] = int_zero_vector(n_roads)
            exited[:] = int_zero_vector(n_roads)
            active_vehicle_seconds[:] = zero_vector(n_roads)
            speed_meter_seconds[:] = zero_vector(n_roads)
            speed_observation_seconds[:] = zero_vector(n_roads)

        while current_time < args.duration and bucket_idx < expected_buckets:
            eng.next_step()
            current_time = float(eng.get_current_time())
            lane_vehicles = eng.get_lane_vehicles()
            vehicle_speed = eng.get_vehicle_speed()

            current_lane_vehicles = {
                lane_id: set(lane_vehicles.get(lane_id, [])) for lane_id in lane_to_road_idx
            }
            for lane_id, road_idx in lane_to_road_idx.items():
                current_set = current_lane_vehicles[lane_id]
                previous_set = prev_lane_vehicles.get(lane_id, set())
                entered[road_idx] += len(current_set - previous_set)
                exited[road_idx] += len(previous_set - current_set)

                active_count = len(current_set)
                active_vehicle_seconds[road_idx] += active_count * interval
                if active_count:
                    for vehicle_id in current_set:
                        speed = vehicle_speed.get(vehicle_id)
                        if speed is not None and speed >= 0:
                            speed_meter_seconds[road_idx] += float(speed) * interval
                            speed_observation_seconds[road_idx] += interval

            prev_lane_vehicles = current_lane_vehicles

            bucket_elapsed += interval
            if current_time >= bucket_start + args.bucket_seconds or current_time >= args.duration:
                flush_bucket(min(current_time, float(args.duration)))

            if args.progress_interval > 0 and int(current_time) % args.progress_interval == 0:
                print(f"[progress] t={current_time:.0f}s buckets={bucket_idx}/{expected_buckets}")

    if tensor is not None:
        assert bucket_start_s is not None and bucket_end_s is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            data=tensor[:bucket_idx],
            road_ids=np.array(road_ids),
            feature_names=np.array(FEATURES),
            bucket_start_s=bucket_start_s[:bucket_idx],
            bucket_end_s=bucket_end_s[:bucket_idx],
            date=np.array(args.date),
            source_config=np.array(str(config_path)),
        )

    print(f"[done] wrote {csv_path}")
    if tensor is not None:
        print(f"[done] wrote {npz_path} with shape {tensor[:bucket_idx].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

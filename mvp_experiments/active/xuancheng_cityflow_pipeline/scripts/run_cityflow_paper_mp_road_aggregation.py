#!/usr/bin/env python3
"""Run the Xuancheng paper-style max-pressure signal setting and aggregate roads.

This implements the Scientific Data Xuancheng state-validation setting at the
simulation/policy level: CityFlow with RL traffic lights enabled, 10-second
max-pressure signal decisions, and 3-second transition phase handling.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


FEATURES = [
    "entered_veh",
    "exited_veh",
    "mean_active_veh",
    "mean_speed_kmh",
    "density_veh_per_lane_km",
]


@dataclass
class SignalIntersection:
    intersection_id: str
    phase_lane_links: list[list[tuple[str, str]]]
    valid_phases: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="CityFlow config JSON with rlTrafficLight=true.")
    parser.add_argument("--date", required=True, help="Date label, e.g. 2023-04-03.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--duration", type=int, default=3600, help="Aggregation duration in seconds.")
    parser.add_argument(
        "--start-second",
        type=int,
        default=17 * 3600,
        help="Simulation second where aggregation starts; paper state validation uses 17:00.",
    )
    parser.add_argument("--bucket-seconds", type=int, default=60, help="Aggregation bucket width.")
    parser.add_argument("--thread-num", type=int, default=1, help="CityFlow thread_num.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Max-pressure decision interval.")
    parser.add_argument("--all-red-time", type=int, default=3, help="Transition duration used by official config.")
    parser.add_argument(
        "--max-same-phase-intervals",
        type=int,
        default=5,
        help="Match official MPAgent guard: switch away after this many repeated decisions when possible.",
    )
    parser.add_argument("--progress-interval", type=int, default=300)
    parser.add_argument("--sparse-csv", action="store_true", help="Write only non-empty road buckets.")
    parser.add_argument("--no-npz", action="store_true", help="Skip compressed dense NPZ output.")
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
        total += math.hypot(float(p0["x"]) - float(p1["x"]), float(p0["y"]) - float(p1["y"]))
    return total


def load_roadnet(
    roadnet_file: Path,
) -> tuple[list[str], dict[str, int], list[dict], dict[str, SignalIntersection]]:
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
        speed_limit_mps = (
            sum(float(lane.get("maxSpeed", 0.0)) for lane in lanes) / lane_count if lanes else 0.0
        )
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

    signals: dict[str, SignalIntersection] = {}
    for inter in roadnet["intersections"]:
        if inter.get("virtual"):
            continue
        road_links = inter.get("roadLinks", [])
        phases = inter.get("trafficLight", {}).get("lightphases", [])
        if len(road_links) == 0 or len(phases) <= 1:
            continue

        road_link_lane_links: list[list[tuple[str, str]]] = []
        for road_link in road_links:
            start_road = str(road_link["startRoad"])
            end_road = str(road_link["endRoad"])
            lane_links = []
            for lane_link in road_link.get("laneLinks", []):
                lane_links.append(
                    (
                        f"{start_road}_{lane_link['startLaneIndex']}",
                        f"{end_road}_{lane_link['endLaneIndex']}",
                    )
                )
            road_link_lane_links.append(lane_links)

        phase_lane_links: list[list[tuple[str, str]]] = []
        valid_phases: list[int] = []
        for phase_idx, phase in enumerate(phases):
            links: list[tuple[str, str]] = []
            for road_link_idx in phase.get("availableRoadLinks", []):
                if 0 <= int(road_link_idx) < len(road_link_lane_links):
                    links.extend(road_link_lane_links[int(road_link_idx)])
            phase_lane_links.append(links)
            # The official CityFlowEnv maps "-1"/transition to phase 0 and
            # starts controllable phase mapping from phase[1:].
            if phase_idx > 0 and links:
                valid_phases.append(phase_idx)

        if valid_phases:
            signals[str(inter["id"])] = SignalIntersection(
                intersection_id=str(inter["id"]),
                phase_lane_links=phase_lane_links,
                valid_phases=valid_phases,
            )

    return road_ids, lane_to_road_idx, road_meta, signals


def choose_max_pressure_actions(
    signals: dict[str, SignalIntersection],
    lane_vehicle_counts: dict[str, int],
    current_phase: dict[str, int],
    current_phase_intervals: dict[str, int],
    max_same_phase_intervals: int,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for inter_id, signal in signals.items():
        phase_pressure: dict[int, float] = {}
        phase_inlane_veh_num: dict[int, float] = {}

        baseline = 0.0
        if signal.phase_lane_links:
            for in_lane, out_lane in signal.phase_lane_links[0]:
                baseline += lane_vehicle_counts.get(in_lane, 0) - lane_vehicle_counts.get(out_lane, 0)

        for phase_idx in signal.valid_phases:
            pressure = 0.0
            inlane_veh = 0.0
            for in_lane, out_lane in signal.phase_lane_links[phase_idx]:
                in_count = lane_vehicle_counts.get(in_lane, 0)
                out_count = lane_vehicle_counts.get(out_lane, 0)
                pressure += in_count - out_count
                inlane_veh += in_count
            phase_pressure[phase_idx] = pressure - baseline
            phase_inlane_veh_num[phase_idx] = inlane_veh

        cur = current_phase.get(inter_id, 0)
        ranked = sorted(
            signal.valid_phases,
            key=lambda p: (phase_pressure.get(p, -1e9), phase_inlane_veh_num.get(p, -1e9), int(p == cur)),
        )
        best = ranked[-1]
        if best == cur and current_phase_intervals.get(inter_id, 0) >= max_same_phase_intervals:
            for candidate in reversed(ranked):
                if candidate != cur:
                    best = candidate
                    break

        if best == cur:
            current_phase_intervals[inter_id] = current_phase_intervals.get(inter_id, 0) + 1
        else:
            current_phase_intervals[inter_id] = 1
        actions[inter_id] = int(best)
    return actions


def open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", newline="", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.bucket_seconds <= 0 or args.decision_interval <= 0:
        raise SystemExit("duration, bucket-seconds, and decision-interval must be positive")
    if args.start_second < 0:
        raise SystemExit("--start-second must be non-negative")

    try:
        import cityflow  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("cityflow is required") from exc

    np = None
    if not args.no_npz:
        try:
            import numpy as np_import  # type: ignore

            np = np_import
        except ModuleNotFoundError as exc:
            raise SystemExit("numpy is required for NPZ output; pass --no-npz to skip it") from exc

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    if not cfg.get("rlTrafficLight", False):
        raise SystemExit("paper max-pressure run requires rlTrafficLight=true in the CityFlow config")
    interval = float(cfg.get("interval", 1.0))
    if interval <= 0:
        raise SystemExit(f"invalid CityFlow interval in config: {interval}")

    roadnet_path = resolve_cityflow_path(config_path, cfg, "roadnetFile")
    road_ids, lane_to_road_idx, road_meta, signals = load_roadnet(roadnet_path)
    n_roads = len(road_ids)
    expected_buckets = int(math.ceil(args.duration / args.bucket_seconds))
    total_until = args.start_second + args.duration

    output_dir = Path(args.output_dir).expanduser().resolve()
    suffix = f"paper_mp_road_agg_{args.bucket_seconds}s_start{args.start_second}_dur{args.duration}"
    csv_path = output_dir / f"xuancheng_{args.date}_{suffix}.csv.gz"
    npz_path = output_dir / f"xuancheng_{args.date}_{suffix}.npz"
    summary_path = output_dir / f"xuancheng_{args.date}_{suffix}.summary.json"

    tensor = None
    if not args.no_npz:
        assert np is not None
        tensor = np.zeros((expected_buckets, n_roads, len(FEATURES)), dtype=np.float32)
        bucket_start_s = np.zeros((expected_buckets,), dtype=np.float32)
        bucket_end_s = np.zeros((expected_buckets,), dtype=np.float32)
    else:
        bucket_start_s = None
        bucket_end_s = None

    print(f"config={config_path}")
    print(f"roadnet={roadnet_path}")
    print(f"roads={n_roads}, lanes={len(lane_to_road_idx)}, signals={len(signals)}")
    print(
        "policy=paper_mp "
        f"start={args.start_second}s duration={args.duration}s bucket={args.bucket_seconds}s "
        f"decision={args.decision_interval}s all_red={args.all_red_time}s"
    )
    print(f"csv={csv_path}")
    if tensor is not None:
        print(f"npz={npz_path}")

    eng = cityflow.Engine(config_file=str(config_path), thread_num=args.thread_num)
    prev_lane_vehicles = {
        lane_id: set(eng.get_lane_vehicles().get(lane_id, [])) for lane_id in lane_to_road_idx
    }

    current_phase: dict[str, int] = {inter_id: 0 for inter_id in signals}
    current_phase_intervals: dict[str, int] = {inter_id: 0 for inter_id in signals}
    target_phase: dict[str, int] = {inter_id: 0 for inter_id in signals}
    transition_remaining: dict[str, int] = {inter_id: 0 for inter_id in signals}
    phase_change_count = 0
    decision_count = 0

    entered = [0] * n_roads
    exited = [0] * n_roads
    active_vehicle_seconds = [0.0] * n_roads
    speed_meter_seconds = [0.0] * n_roads
    speed_observation_seconds = [0.0] * n_roads
    bucket_idx = 0
    bucket_start = float(args.start_second)
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
                "signal_policy",
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
                    mean_speed_kmh = speed_meter_seconds[road_idx] / speed_observation_seconds[road_idx] * 3.6
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
                        "paper_mp",
                    ]
                )
            bucket_idx += 1
            bucket_start = time_end
            bucket_elapsed = 0.0
            entered[:] = [0] * n_roads
            exited[:] = [0] * n_roads
            active_vehicle_seconds[:] = [0.0] * n_roads
            speed_meter_seconds[:] = [0.0] * n_roads
            speed_observation_seconds[:] = [0.0] * n_roads

        while current_time < total_until and bucket_idx < expected_buckets:
            if int(current_time) % args.decision_interval == 0:
                lane_vehicle_counts = {
                    lane_id: len(vehicles) for lane_id, vehicles in eng.get_lane_vehicles().items()
                }
                actions = choose_max_pressure_actions(
                    signals,
                    lane_vehicle_counts,
                    current_phase,
                    current_phase_intervals,
                    args.max_same_phase_intervals,
                )
                decision_count += 1
                for inter_id, action in actions.items():
                    if decision_count == 1 or action == current_phase.get(inter_id, 0):
                        transition_remaining[inter_id] = 0
                    else:
                        transition_remaining[inter_id] = args.all_red_time
                        phase_change_count += 1
                    target_phase[inter_id] = action

            for inter_id in signals:
                if transition_remaining[inter_id] > 0:
                    eng.set_tl_phase(inter_id, 0)
                    transition_remaining[inter_id] -= 1
                    if transition_remaining[inter_id] == 0:
                        current_phase[inter_id] = target_phase[inter_id]
                else:
                    eng.set_tl_phase(inter_id, target_phase[inter_id])
                    current_phase[inter_id] = target_phase[inter_id]

            eng.next_step()
            current_time = float(eng.get_current_time())

            lane_vehicles = eng.get_lane_vehicles()
            current_lane_vehicles = {
                lane_id: set(lane_vehicles.get(lane_id, [])) for lane_id in lane_to_road_idx
            }

            if args.start_second < current_time <= total_until:
                vehicle_speed = eng.get_vehicle_speed()
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
                bucket_elapsed += interval
                if (
                    current_time >= bucket_start + args.bucket_seconds
                    or current_time >= total_until
                ):
                    flush_bucket(min(current_time, float(total_until)))

            prev_lane_vehicles = current_lane_vehicles

            if args.progress_interval > 0 and int(current_time) % args.progress_interval == 0:
                print(
                    f"[progress] sim_t={current_time:.0f}s "
                    f"agg_buckets={bucket_idx}/{expected_buckets} decisions={decision_count}"
                )

    if tensor is not None:
        assert np is not None
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
            signal_policy=np.array("paper_mp"),
            start_second=np.array(args.start_second),
            decision_interval=np.array(args.decision_interval),
            all_red_time=np.array(args.all_red_time),
        )

    summary = {
        "date": args.date,
        "policy": "paper_mp",
        "config": str(config_path),
        "roadnet": str(roadnet_path),
        "start_second": args.start_second,
        "duration": args.duration,
        "bucket_seconds": args.bucket_seconds,
        "roads": n_roads,
        "lanes": len(lane_to_road_idx),
        "signals": len(signals),
        "buckets_written": bucket_idx,
        "decision_count": decision_count,
        "phase_change_count": phase_change_count,
        "csv": str(csv_path),
        "npz": str(npz_path) if tensor is not None else None,
        "official_alignment_note": (
            "Paper setting: max-pressure signal control for the 17:00-18:00 state-validation window. "
            "This script implements max-pressure directly on CityFlow roadnet phases; it does not import "
            "the original repository's CityFlowEnv/MPAgent classes."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"[done] wrote {csv_path}")
    if tensor is not None:
        print(f"[done] wrote {npz_path} with shape {tensor[:bucket_idx].shape}")
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

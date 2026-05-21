#!/usr/bin/env python3
"""Generate DTIGNN-style Xuancheng traffic-flow transition tensors.

The primary target is 10-second road-segment directional movement volume:
for each start road, count vehicles that traverse a CityFlow roadLink during
the bucket, grouped as left / straight / right.  The script also stores a
state-like active-vehicle tensor grouped by each vehicle's next-turn intent,
plus signal phase traces needed to reconstruct phase-activated dynamic graphs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


TURN_TO_INDEX = {
    "turn_left": 0,
    "go_straight": 1,
    "turn_right": 2,
}
TURN_NAMES = ["turn_left", "go_straight", "turn_right"]


@dataclass
class SignalIntersection:
    intersection_id: str
    signal_index: int
    phase_lane_links: list[list[tuple[str, str]]]
    valid_phases: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="CityFlow config JSON with rlTrafficLight=true.")
    parser.add_argument("--date", required=True, help="Date label, e.g. 2023-04-03.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--duration", type=int, default=3600, help="Aggregation duration in seconds.")
    parser.add_argument("--start-second", type=int, default=0, help="Simulation second where aggregation starts.")
    parser.add_argument("--bucket-seconds", type=int, default=10, help="Bucket width; DTIGNN uses 10 seconds.")
    parser.add_argument("--thread-num", type=int, default=1, help="CityFlow thread_num.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Max-pressure decision interval.")
    parser.add_argument("--all-red-time", type=int, default=3, help="Transition/all-red phase duration.")
    parser.add_argument(
        "--max-same-phase-intervals",
        type=int,
        default=5,
        help="Match the official MPAgent guard: switch after this many repeated decisions when possible.",
    )
    parser.add_argument("--progress-interval", type=int, default=300)
    parser.add_argument(
        "--missing-ratios",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated road missing ratios for DTIGNN sparse-observation masks.",
    )
    parser.add_argument("--mask-seed", type=int, default=2026, help="Deterministic seed for observation masks.")
    return parser.parse_args()


def parse_missing_ratios(text: str) -> list[float]:
    ratios: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0 or value >= 1:
            raise argparse.ArgumentTypeError("missing ratios must be in [0, 1)")
        ratios.append(value)
    if not ratios:
        raise argparse.ArgumentTypeError("at least one missing ratio is required")
    return ratios


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


def make_lane_links(road_link: dict) -> list[tuple[str, str]]:
    start_road = str(road_link["startRoad"])
    end_road = str(road_link["endRoad"])
    links: list[tuple[str, str]] = []
    for lane_link in road_link.get("laneLinks", []):
        links.append(
            (
                f"{start_road}_{lane_link['startLaneIndex']}",
                f"{end_road}_{lane_link['endLaneIndex']}",
            )
        )
    return links


def load_roadnet(roadnet_file: Path) -> dict:
    with roadnet_file.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)

    road_ids: list[str] = []
    road_meta: list[dict] = []
    lane_to_road_idx: dict[str, int] = {}
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
                "speed_limit_kmh": speed_limit_mps * 3.6,
            }
        )
        for lane_idx in range(lane_count):
            lane_to_road_idx[f"{road_id}_{lane_idx}"] = road_idx

    road_to_idx = {road_id: idx for idx, road_id in enumerate(road_ids)}
    intersection_ids: list[str] = []
    intersection_to_idx: dict[str, int] = {}
    signal_ids: list[str] = []
    signals: dict[str, SignalIntersection] = {}
    max_phase_count = 1

    edge_start: list[int] = []
    edge_end: list[int] = []
    edge_turn: list[int] = []
    edge_intersection: list[int] = []
    edge_road_link_index: list[int] = []
    edge_signal: list[int] = []
    edge_always_active: list[int] = []
    movement_to_turn: dict[tuple[str, str], int] = {}

    phase_edge_signal: list[int] = []
    phase_edge_phase: list[int] = []
    phase_edge_index: list[int] = []

    for inter in roadnet["intersections"]:
        if inter.get("virtual"):
            continue
        inter_id = str(inter["id"])
        intersection_to_idx[inter_id] = len(intersection_ids)
        intersection_ids.append(inter_id)

    for inter in roadnet["intersections"]:
        if inter.get("virtual"):
            continue
        inter_id = str(inter["id"])
        road_links = inter.get("roadLinks", [])
        phases = inter.get("trafficLight", {}).get("lightphases", [])
        max_phase_count = max(max_phase_count, len(phases))

        road_link_edges: list[int] = []
        road_link_lane_links: list[list[tuple[str, str]]] = []
        valid_phases: list[int] = []
        has_controllable_phase = any(
            phase_idx > 0 and phase.get("availableRoadLinks")
            for phase_idx, phase in enumerate(phases)
        )
        signal_idx = -1
        if has_controllable_phase:
            signal_idx = len(signal_ids)
            signal_ids.append(inter_id)

        for road_link_idx, road_link in enumerate(road_links):
            start_road = str(road_link["startRoad"])
            end_road = str(road_link["endRoad"])
            turn_name = str(road_link.get("type", ""))
            if start_road not in road_to_idx or end_road not in road_to_idx:
                road_link_edges.append(-1)
                road_link_lane_links.append([])
                continue
            turn_idx = TURN_TO_INDEX.get(turn_name)
            if turn_idx is None:
                road_link_edges.append(-1)
                road_link_lane_links.append([])
                continue

            edge_idx = len(edge_start)
            road_link_edges.append(edge_idx)
            road_link_lane_links.append(make_lane_links(road_link))
            edge_start.append(road_to_idx[start_road])
            edge_end.append(road_to_idx[end_road])
            edge_turn.append(turn_idx)
            edge_intersection.append(intersection_to_idx[inter_id])
            edge_road_link_index.append(road_link_idx)
            edge_signal.append(signal_idx)
            edge_always_active.append(0 if has_controllable_phase else 1)
            movement_to_turn[(start_road, end_road)] = turn_idx

        phase_lane_links: list[list[tuple[str, str]]] = []
        for phase_idx, phase in enumerate(phases):
            links: list[tuple[str, str]] = []
            for road_link_idx in phase.get("availableRoadLinks", []):
                road_link_idx = int(road_link_idx)
                if 0 <= road_link_idx < len(road_link_lane_links):
                    links.extend(road_link_lane_links[road_link_idx])
                if has_controllable_phase and 0 <= road_link_idx < len(road_link_edges):
                    edge_idx = road_link_edges[road_link_idx]
                    if edge_idx >= 0:
                        phase_edge_signal.append(signal_idx)
                        phase_edge_phase.append(phase_idx)
                        phase_edge_index.append(edge_idx)
            phase_lane_links.append(links)
            if phase_idx > 0 and phase.get("availableRoadLinks"):
                valid_phases.append(phase_idx)

        if has_controllable_phase:
            signals[inter_id] = SignalIntersection(
                intersection_id=inter_id,
                signal_index=signal_idx,
                phase_lane_links=phase_lane_links,
                valid_phases=valid_phases,
            )

    return {
        "road_ids": road_ids,
        "road_to_idx": road_to_idx,
        "road_meta": road_meta,
        "lane_to_road_idx": lane_to_road_idx,
        "intersection_ids": intersection_ids,
        "signal_ids": signal_ids,
        "signals": signals,
        "movement_to_turn": movement_to_turn,
        "max_phase_count": max_phase_count,
        "edge_start": edge_start,
        "edge_end": edge_end,
        "edge_turn": edge_turn,
        "edge_intersection": edge_intersection,
        "edge_road_link_index": edge_road_link_index,
        "edge_signal": edge_signal,
        "edge_always_active": edge_always_active,
        "phase_edge_signal": phase_edge_signal,
        "phase_edge_phase": phase_edge_phase,
        "phase_edge_index": phase_edge_index,
    }


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


def parse_vehicle_road(info: dict) -> str | None:
    road = info.get("road")
    if road:
        return str(road)
    drivable = info.get("drivable", "")
    if "_" in drivable:
        return str(drivable).rsplit("_", 1)[0]
    return None


def next_road_from_route(info: dict, current_road: str) -> str | None:
    route = str(info.get("route", "")).split()
    if not route:
        return None
    for idx, road_id in enumerate(route):
        if road_id == current_road:
            if idx + 1 < len(route):
                return route[idx + 1]
            return None
    return None


def make_road_masks(np, n_roads: int, missing_ratios: list[float], seed: int):
    rng = np.random.default_rng(seed)
    masks = np.zeros((len(missing_ratios), n_roads), dtype=np.uint8)
    all_indices = np.arange(n_roads)
    for ratio_idx, missing_ratio in enumerate(missing_ratios):
        observed_count = max(1, int(round(n_roads * (1.0 - missing_ratio))))
        observed = rng.choice(all_indices, size=observed_count, replace=False)
        masks[ratio_idx, observed] = 1
    return masks


def main() -> int:
    args = parse_args()
    missing_ratios = parse_missing_ratios(args.missing_ratios)
    if args.duration <= 0 or args.bucket_seconds <= 0 or args.decision_interval <= 0:
        raise SystemExit("duration, bucket-seconds, and decision-interval must be positive")
    if args.start_second < 0:
        raise SystemExit("--start-second must be non-negative")

    try:
        import cityflow  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("cityflow is required") from exc

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("numpy is required") from exc

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    if not cfg.get("rlTrafficLight", False):
        raise SystemExit("DTIGNN-style max-pressure generation requires rlTrafficLight=true")
    interval = float(cfg.get("interval", 1.0))
    if interval <= 0:
        raise SystemExit(f"invalid CityFlow interval in config: {interval}")

    roadnet_path = resolve_cityflow_path(config_path, cfg, "roadnetFile")
    roadnet = load_roadnet(roadnet_path)
    road_ids = roadnet["road_ids"]
    road_to_idx = roadnet["road_to_idx"]
    signals = roadnet["signals"]
    movement_to_turn = roadnet["movement_to_turn"]
    signal_ids = roadnet["signal_ids"]
    n_roads = len(road_ids)
    n_signals = len(signal_ids)
    n_buckets = int(math.ceil(args.duration / args.bucket_seconds))
    total_until = args.start_second + args.duration

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"dtignn_turn_{args.bucket_seconds}s_start{args.start_second}_dur{args.duration}"
    npz_path = output_dir / f"xuancheng_{args.date}_{suffix}.npz"
    summary_path = output_dir / f"xuancheng_{args.date}_{suffix}.summary.json"

    movement_volume_lsr = np.zeros((n_buckets, n_roads, 3), dtype=np.uint16)
    active_volume_lsr = np.zeros((n_buckets, n_roads, 3), dtype=np.float32)
    movement_unknown = np.zeros((n_buckets, n_roads), dtype=np.uint16)
    active_unknown = np.zeros((n_buckets, n_roads), dtype=np.float32)
    phase_id_end = np.zeros((n_buckets, n_signals), dtype=np.int16)
    phase_time_fraction = np.zeros(
        (n_buckets, n_signals, int(roadnet["max_phase_count"])),
        dtype=np.float32,
    )
    bucket_start_s = np.zeros((n_buckets,), dtype=np.float32)
    bucket_end_s = np.zeros((n_buckets,), dtype=np.float32)

    bucket_movement = np.zeros((n_roads, 3), dtype=np.uint32)
    bucket_active_seconds = np.zeros((n_roads, 3), dtype=np.float64)
    bucket_movement_unknown = np.zeros((n_roads,), dtype=np.uint32)
    bucket_active_unknown_seconds = np.zeros((n_roads,), dtype=np.float64)
    bucket_phase_seconds = np.zeros((n_signals, int(roadnet["max_phase_count"])), dtype=np.float64)

    print(f"config={config_path}")
    print(f"roadnet={roadnet_path}")
    print(f"roads={n_roads}, signals={n_signals}, edges={len(roadnet['edge_start'])}")
    print(
        "policy=paper_mp "
        f"start={args.start_second}s duration={args.duration}s bucket={args.bucket_seconds}s "
        f"decision={args.decision_interval}s all_red={args.all_red_time}s"
    )
    print(f"npz={npz_path}")

    eng = cityflow.Engine(config_file=str(config_path), thread_num=args.thread_num)
    current_phase: dict[str, int] = {inter_id: 0 for inter_id in signals}
    current_phase_intervals: dict[str, int] = {inter_id: 0 for inter_id in signals}
    target_phase: dict[str, int] = {inter_id: 0 for inter_id in signals}
    transition_remaining: dict[str, int] = {inter_id: 0 for inter_id in signals}
    applied_phase: dict[str, int] = {inter_id: 0 for inter_id in signals}

    prev_vehicle_road: dict[str, str] = {}
    decision_count = 0
    phase_change_count = 0
    bucket_idx = 0
    bucket_start = float(args.start_second)
    bucket_elapsed = 0.0
    current_time = 0.0
    skipped_no_road = 0
    skipped_unknown_road = 0
    skipped_no_next = 0
    skipped_unknown_turn = 0

    def flush_bucket(time_end: float) -> None:
        nonlocal bucket_idx, bucket_start, bucket_elapsed
        if bucket_idx >= n_buckets:
            return
        elapsed = max(bucket_elapsed, interval)
        bucket_start_s[bucket_idx] = bucket_start
        bucket_end_s[bucket_idx] = time_end
        movement_volume_lsr[bucket_idx] = np.minimum(bucket_movement, np.iinfo(np.uint16).max)
        active_volume_lsr[bucket_idx] = bucket_active_seconds / elapsed
        movement_unknown[bucket_idx] = np.minimum(bucket_movement_unknown, np.iinfo(np.uint16).max)
        active_unknown[bucket_idx] = bucket_active_unknown_seconds / elapsed
        phase_id_end[bucket_idx] = np.array(
            [applied_phase.get(signal_id, 0) for signal_id in signal_ids],
            dtype=np.int16,
        )
        if n_signals:
            phase_time_fraction[bucket_idx] = bucket_phase_seconds / elapsed

        bucket_idx += 1
        bucket_start = time_end
        bucket_elapsed = 0.0
        bucket_movement.fill(0)
        bucket_active_seconds.fill(0)
        bucket_movement_unknown.fill(0)
        bucket_active_unknown_seconds.fill(0)
        bucket_phase_seconds.fill(0)

    while current_time < total_until and bucket_idx < n_buckets:
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
                applied_phase[inter_id] = 0
                transition_remaining[inter_id] -= 1
                if transition_remaining[inter_id] == 0:
                    current_phase[inter_id] = target_phase[inter_id]
            else:
                eng.set_tl_phase(inter_id, target_phase[inter_id])
                applied_phase[inter_id] = target_phase[inter_id]
                current_phase[inter_id] = target_phase[inter_id]

        eng.next_step()
        current_time = float(eng.get_current_time())
        vehicles = eng.get_vehicles()
        current_vehicle_road: dict[str, str] = {}
        vehicle_info: dict[str, dict] = {}

        for vehicle_id in vehicles:
            info = eng.get_vehicle_info(vehicle_id)
            road = parse_vehicle_road(info)
            if road is None:
                skipped_no_road += 1
                continue
            current_vehicle_road[vehicle_id] = road
            vehicle_info[vehicle_id] = info

        if args.start_second < current_time <= total_until:
            for vehicle_id, road in current_vehicle_road.items():
                prev_road = prev_vehicle_road.get(vehicle_id)
                if prev_road and prev_road != road:
                    prev_road_idx = road_to_idx.get(prev_road)
                    if prev_road_idx is None:
                        skipped_unknown_road += 1
                    else:
                        turn_idx = movement_to_turn.get((prev_road, road))
                        if turn_idx is None:
                            bucket_movement_unknown[prev_road_idx] += 1
                            skipped_unknown_turn += 1
                        else:
                            bucket_movement[prev_road_idx, turn_idx] += 1

                road_idx = road_to_idx.get(road)
                if road_idx is None:
                    skipped_unknown_road += 1
                    continue
                next_road = next_road_from_route(vehicle_info[vehicle_id], road)
                if next_road is None:
                    bucket_active_unknown_seconds[road_idx] += interval
                    skipped_no_next += 1
                    continue
                turn_idx = movement_to_turn.get((road, next_road))
                if turn_idx is None:
                    bucket_active_unknown_seconds[road_idx] += interval
                    skipped_unknown_turn += 1
                else:
                    bucket_active_seconds[road_idx, turn_idx] += interval

            for signal in signals.values():
                phase = applied_phase.get(signal.intersection_id, 0)
                if 0 <= phase < bucket_phase_seconds.shape[1]:
                    bucket_phase_seconds[signal.signal_index, phase] += interval

            bucket_elapsed += interval
            if current_time >= bucket_start + args.bucket_seconds or current_time >= total_until:
                flush_bucket(min(current_time, float(total_until)))

        prev_vehicle_road = current_vehicle_road

        if args.progress_interval > 0 and int(current_time) % args.progress_interval == 0:
            print(
                f"[progress] sim_t={current_time:.0f}s "
                f"buckets={bucket_idx}/{n_buckets} decisions={decision_count}"
            )

    observed_road_masks = make_road_masks(np, n_roads, missing_ratios, args.mask_seed)
    observed_lsr_masks = np.repeat(observed_road_masks[:, :, None], 3, axis=2).astype(np.uint8)

    movement_total = int(movement_volume_lsr[:bucket_idx].sum())
    active_total = float(active_volume_lsr[:bucket_idx].sum())
    movement_unknown_total = int(movement_unknown[:bucket_idx].sum())
    active_unknown_total = float(active_unknown[:bucket_idx].sum())

    np.savez_compressed(
        npz_path,
        movement_volume_lsr=movement_volume_lsr[:bucket_idx],
        active_volume_lsr=active_volume_lsr[:bucket_idx],
        movement_unknown=movement_unknown[:bucket_idx],
        active_unknown=active_unknown[:bucket_idx],
        phase_id_end=phase_id_end[:bucket_idx],
        phase_time_fraction=phase_time_fraction[:bucket_idx],
        bucket_start_s=bucket_start_s[:bucket_idx],
        bucket_end_s=bucket_end_s[:bucket_idx],
        road_ids=np.array(road_ids),
        turn_names=np.array(TURN_NAMES),
        signal_ids=np.array(signal_ids),
        intersection_ids=np.array(roadnet["intersection_ids"]),
        road_lane_count=np.array([m["lane_count"] for m in roadnet["road_meta"]], dtype=np.int16),
        road_length_m=np.array([m["length_m"] for m in roadnet["road_meta"]], dtype=np.float32),
        road_speed_limit_kmh=np.array(
            [m["speed_limit_kmh"] for m in roadnet["road_meta"]],
            dtype=np.float32,
        ),
        edge_start_road_idx=np.array(roadnet["edge_start"], dtype=np.int32),
        edge_end_road_idx=np.array(roadnet["edge_end"], dtype=np.int32),
        edge_turn_type_idx=np.array(roadnet["edge_turn"], dtype=np.int8),
        edge_intersection_idx=np.array(roadnet["edge_intersection"], dtype=np.int32),
        edge_road_link_idx=np.array(roadnet["edge_road_link_index"], dtype=np.int16),
        edge_signal_idx=np.array(roadnet["edge_signal"], dtype=np.int16),
        edge_always_active=np.array(roadnet["edge_always_active"], dtype=np.uint8),
        phase_edge_signal_idx=np.array(roadnet["phase_edge_signal"], dtype=np.int16),
        phase_edge_phase_id=np.array(roadnet["phase_edge_phase"], dtype=np.int16),
        phase_edge_idx=np.array(roadnet["phase_edge_index"], dtype=np.int32),
        missing_ratios=np.array(missing_ratios, dtype=np.float32),
        observed_road_masks=observed_road_masks,
        observed_lsr_masks=observed_lsr_masks,
        date=np.array(args.date),
        source_config=np.array(str(config_path)),
        source_roadnet=np.array(str(roadnet_path)),
        signal_policy=np.array("paper_mp"),
        start_second=np.array(args.start_second),
        duration=np.array(args.duration),
        bucket_seconds=np.array(args.bucket_seconds),
    )

    summary = {
        "date": args.date,
        "config": str(config_path),
        "roadnet": str(roadnet_path),
        "output_npz": str(npz_path),
        "signal_policy": "paper_mp",
        "start_second": args.start_second,
        "duration": args.duration,
        "bucket_seconds": args.bucket_seconds,
        "buckets_written": bucket_idx,
        "roads": n_roads,
        "signals": n_signals,
        "road_link_edges": len(roadnet["edge_start"]),
        "phase_edge_rows": len(roadnet["phase_edge_index"]),
        "movement_total": movement_total,
        "movement_unknown_total": movement_unknown_total,
        "movement_unknown_rate": movement_unknown_total / max(movement_total + movement_unknown_total, 1),
        "active_lsr_total": active_total,
        "active_unknown_total": active_unknown_total,
        "active_unknown_rate": active_unknown_total / max(active_total + active_unknown_total, 1e-9),
        "decision_count": decision_count,
        "phase_change_count": phase_change_count,
        "missing_ratios": missing_ratios,
        "skipped_no_road": skipped_no_road,
        "skipped_unknown_road": skipped_unknown_road,
        "skipped_no_next": skipped_no_next,
        "skipped_unknown_turn": skipped_unknown_turn,
        "dtignn_alignment_note": (
            "Primary target is road-segment left/straight/right movement volume at 10-second buckets. "
            "Signal dynamic graph is represented by phase_id_end plus phase_edge_* metadata; non-signal "
            "intersections are encoded as always-active edges."
        ),
        "reported_deviation_note": (
            "The generator uses the released raw Xuancheng flow and a direct max-pressure controller implemented "
            "against CityFlow roadnet phases. This follows the paper-level MP setting, but it is not the official "
            "repository's as-is test.py entrypoint because the public release is missing required files/imports."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"[done] wrote {npz_path} with movement shape {movement_volume_lsr[:bucket_idx].shape}")
    print(f"[done] wrote {summary_path}")
    print(
        "[summary] "
        f"movement_total={movement_total} movement_unknown_rate={summary['movement_unknown_rate']:.6f} "
        f"active_unknown_rate={summary['active_unknown_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

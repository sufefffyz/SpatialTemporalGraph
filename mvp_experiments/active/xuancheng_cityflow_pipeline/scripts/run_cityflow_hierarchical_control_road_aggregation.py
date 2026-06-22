#!/usr/bin/env python3
"""Run Xuancheng hierarchical-control MVP and aggregate road states.

This extends the paper-style max-pressure signal-control runner with an
MFD-inspired perimeter-control layer. The public Xuancheng repository exposes
network metrics and ``set_vehicle_route`` support, but it does not ship a
directly runnable full hierarchical controller. This script therefore keeps the
verified max-pressure lower layer and adds a clearly reported heuristic upper
layer:

* protected core roads are selected either from road geometry or from a prior
  stock NPZ;
* boundary phases that move vehicles from outside the core into the core are
  metered when core accumulation exceeds a target density;
* an optional route-guidance hook can reroute active vehicles around the core.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from run_cityflow_paper_mp_road_aggregation import (
    FEATURES,
    SignalIntersection,
    load_config,
    load_roadnet,
    open_csv,
    resolve_cityflow_path,
)


@dataclass
class RoadGeometry:
    road_id: str
    center_x: float
    center_y: float
    lane_count: int
    lane_km: float


@dataclass
class BoundaryControl:
    boundary_signals: set[str]
    inbound_phases: dict[str, set[int]]
    safe_phases: dict[str, list[int]]


@dataclass
class PerimeterState:
    active: bool = False
    activation_count: int = 0
    override_count: int = 0
    all_red_count: int = 0
    open_release_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="CityFlow config JSON with rlTrafficLight=true.")
    parser.add_argument("--date", required=True, help="Date label, e.g. 2023-04-03.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--start-second", type=int, default=0)
    parser.add_argument("--bucket-seconds", type=int, default=60)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--all-red-time", type=int, default=3)
    parser.add_argument("--max-same-phase-intervals", type=int, default=5)
    parser.add_argument("--progress-interval", type=int, default=300)
    parser.add_argument("--sparse-csv", action="store_true")
    parser.add_argument("--no-npz", action="store_true")

    parser.add_argument("--perimeter-control", action="store_true", help="Enable MFD-style perimeter gating.")
    parser.add_argument(
        "--core-mode",
        choices=["bbox", "score_npz"],
        default="score_npz",
        help="How to choose protected core roads.",
    )
    parser.add_argument(
        "--core-bbox-quantiles",
        default="0.25,0.75,0.25,0.75",
        help="x_low,x_high,y_low,y_high quantiles for --core-mode bbox.",
    )
    parser.add_argument(
        "--core-score-npz",
        default="",
        help="Prior road-aggregation NPZ used to choose high-stock core roads.",
    )
    parser.add_argument("--core-score-start-second", type=int, default=7 * 3600)
    parser.add_argument("--core-score-duration", type=int, default=3 * 3600)
    parser.add_argument("--core-score-topk", type=int, default=250)
    parser.add_argument(
        "--perimeter-target-density",
        type=float,
        default=55.0,
        help="Activate perimeter when core stock density exceeds this veh/lane-km.",
    )
    parser.add_argument(
        "--perimeter-release-density",
        type=float,
        default=40.0,
        help="Release perimeter when core stock density falls below this veh/lane-km.",
    )
    parser.add_argument(
        "--perimeter-open-every-n",
        type=int,
        default=0,
        help="If >0, allow the original inbound phase once every N active perimeter decisions.",
    )

    parser.add_argument("--route-guidance", action="store_true", help="Enable optional route-guidance hook.")
    parser.add_argument(
        "--route-guidance-max-vehicles-per-decision",
        type=int,
        default=50,
        help="Cap reroute attempts per decision when route guidance is enabled.",
    )
    parser.add_argument(
        "--route-guidance-cooldown",
        type=int,
        default=300,
        help="Minimum seconds between reroute attempts for the same vehicle.",
    )
    return parser.parse_args()


def road_center(points: list[dict]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return (
        sum(float(p["x"]) for p in points) / len(points),
        sum(float(p["y"]) for p in points) / len(points),
    )


def road_length_m(points: list[dict]) -> float:
    total = 0.0
    for p0, p1 in zip(points, points[1:]):
        total += math.hypot(float(p0["x"]) - float(p1["x"]), float(p0["y"]) - float(p1["y"]))
    return total


def load_roadnet_extra(roadnet_file: Path) -> tuple[dict[str, RoadGeometry], dict[str, list[str]]]:
    with roadnet_file.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)

    road_geom: dict[str, RoadGeometry] = {}
    adjacency: dict[str, list[str]] = {}
    for road in roadnet["roads"]:
        road_id = str(road["id"])
        points = road.get("points", [])
        cx, cy = road_center(points)
        lanes = road.get("lanes", [])
        lane_count = len(lanes)
        length_m = road_length_m(points)
        road_geom[road_id] = RoadGeometry(
            road_id=road_id,
            center_x=cx,
            center_y=cy,
            lane_count=lane_count,
            lane_km=max(length_m * lane_count / 1000.0, 0.0),
        )
        adjacency[road_id] = []

    for inter in roadnet["intersections"]:
        for link in inter.get("roadLinks", []):
            start = str(link["startRoad"])
            end = str(link["endRoad"])
            if start in adjacency and end in road_geom:
                adjacency[start].append(end)
    return road_geom, adjacency


def lane_to_road_id(lane_id: str) -> str:
    left, sep, right = lane_id.rpartition("_")
    if sep and right.isdigit():
        return left
    return lane_id


def choose_bbox_core(road_geom: dict[str, RoadGeometry], spec: str) -> set[str]:
    vals = [float(x.strip()) for x in spec.split(",")]
    if len(vals) != 4:
        raise SystemExit("--core-bbox-quantiles must have four comma-separated values")
    qx0, qx1, qy0, qy1 = vals
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("numpy is required for bbox core selection") from exc
    xs = np.array([g.center_x for g in road_geom.values()], dtype=np.float64)
    ys = np.array([g.center_y for g in road_geom.values()], dtype=np.float64)
    x0, x1 = np.quantile(xs, [qx0, qx1])
    y0, y1 = np.quantile(ys, [qy0, qy1])
    return {
        road_id
        for road_id, geom in road_geom.items()
        if x0 <= geom.center_x <= x1 and y0 <= geom.center_y <= y1
    }


def choose_score_core(args: argparse.Namespace, road_ids: list[str]) -> set[str]:
    if not args.core_score_npz:
        raise SystemExit("--core-score-npz is required for --core-mode score_npz")
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("numpy is required for score_npz core selection") from exc
    z = np.load(args.core_score_npz, allow_pickle=True)
    data = z["data"]
    features = [str(x) for x in z["feature_names"]]
    source_roads = [str(x) for x in z["road_ids"]]
    if "mean_active_veh" not in features:
        raise SystemExit("core-score NPZ must contain mean_active_veh")
    fi = features.index("mean_active_veh")
    bucket_start = z["bucket_start_s"] if "bucket_start_s" in z else np.arange(data.shape[0]) * 60
    bucket_end = z["bucket_end_s"] if "bucket_end_s" in z else bucket_start + 60
    start = args.core_score_start_second
    end = start + args.core_score_duration
    mask = (bucket_end > start) & (bucket_start < end)
    if not mask.any():
        raise SystemExit("core-score window has no overlapping buckets")
    scores = data[mask, :, fi].mean(axis=0)
    order = np.argsort(scores)[::-1]
    available = set(road_ids)
    chosen: list[str] = []
    for idx in order:
        road_id = source_roads[int(idx)]
        if road_id in available:
            chosen.append(road_id)
        if len(chosen) >= args.core_score_topk:
            break
    return set(chosen)


def compute_boundary_control(signals: dict[str, SignalIntersection], core_roads: set[str]) -> BoundaryControl:
    boundary_signals: set[str] = set()
    inbound_phases: dict[str, set[int]] = {}
    safe_phases: dict[str, list[int]] = {}
    for inter_id, signal in signals.items():
        inbound: set[int] = set()
        safe: list[int] = []
        for phase_idx in signal.valid_phases:
            has_inbound = False
            for in_lane, out_lane in signal.phase_lane_links[phase_idx]:
                in_road = lane_to_road_id(in_lane)
                out_road = lane_to_road_id(out_lane)
                if in_road not in core_roads and out_road in core_roads:
                    has_inbound = True
                    break
            if has_inbound:
                inbound.add(phase_idx)
            else:
                safe.append(phase_idx)
        if inbound:
            boundary_signals.add(inter_id)
            inbound_phases[inter_id] = inbound
            safe_phases[inter_id] = safe
    return BoundaryControl(boundary_signals, inbound_phases, safe_phases)


def phase_scores(
    signal: SignalIntersection,
    lane_vehicle_counts: dict[str, int],
    phases: list[int],
) -> dict[int, tuple[float, float]]:
    baseline = 0.0
    if signal.phase_lane_links:
        for in_lane, out_lane in signal.phase_lane_links[0]:
            baseline += lane_vehicle_counts.get(in_lane, 0) - lane_vehicle_counts.get(out_lane, 0)
    scores: dict[int, tuple[float, float]] = {}
    for phase_idx in phases:
        pressure = 0.0
        inlane_veh = 0.0
        for in_lane, out_lane in signal.phase_lane_links[phase_idx]:
            in_count = lane_vehicle_counts.get(in_lane, 0)
            out_count = lane_vehicle_counts.get(out_lane, 0)
            pressure += in_count - out_count
            inlane_veh += in_count
        scores[phase_idx] = (pressure - baseline, inlane_veh)
    return scores


def choose_max_pressure_actions(
    signals: dict[str, SignalIntersection],
    lane_vehicle_counts: dict[str, int],
    current_phase: dict[str, int],
    current_phase_intervals: dict[str, int],
    max_same_phase_intervals: int,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for inter_id, signal in signals.items():
        scores = phase_scores(signal, lane_vehicle_counts, signal.valid_phases)
        cur = current_phase.get(inter_id, 0)
        ranked = sorted(
            signal.valid_phases,
            key=lambda p: (scores.get(p, (-1e9, -1e9))[0], scores.get(p, (-1e9, -1e9))[1], int(p == cur)),
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


def apply_perimeter_overrides(
    actions: dict[str, int],
    signals: dict[str, SignalIntersection],
    lane_vehicle_counts: dict[str, int],
    boundary: BoundaryControl,
    pstate: PerimeterState,
    decision_count: int,
    open_every_n: int,
) -> dict[str, int]:
    if not pstate.active:
        return actions
    out = dict(actions)
    for inter_id in boundary.boundary_signals:
        action = out.get(inter_id, 0)
        inbound = boundary.inbound_phases.get(inter_id, set())
        if action not in inbound:
            continue
        if open_every_n > 0 and decision_count % open_every_n == 0:
            pstate.open_release_count += 1
            continue
        safe = boundary.safe_phases.get(inter_id, [])
        if not safe:
            out[inter_id] = 0
            pstate.all_red_count += 1
            pstate.override_count += 1
            continue
        scores = phase_scores(signals[inter_id], lane_vehicle_counts, safe)
        out[inter_id] = max(safe, key=lambda p: (scores[p][0], scores[p][1], int(p == action)))
        pstate.override_count += 1
    return out


def parse_vehicle_current_road(info: dict) -> str | None:
    road = info.get("road")
    if road:
        return str(road)
    drivable = str(info.get("drivable", ""))
    if "_TO_" in drivable:
        left, _ = drivable.split("_TO_", 1)
        return lane_to_road_id(left)
    return None


def shortest_path_avoid_core(
    adjacency: dict[str, list[str]],
    start: str,
    goal: str,
    core_roads: set[str],
) -> list[str] | None:
    if start == goal:
        return [start]
    queue: deque[str] = deque([start])
    prev: dict[str, str | None] = {start: None}
    while queue:
        cur = queue.popleft()
        for nxt in adjacency.get(cur, []):
            if nxt in prev:
                continue
            if nxt in core_roads and nxt != goal:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [goal]
                back = cur
                while back is not None:
                    path.append(back)
                    back = prev[back]
                return list(reversed(path))
            queue.append(nxt)
    return None


def maybe_apply_route_guidance(
    eng,
    adjacency: dict[str, list[str]],
    core_roads: set[str],
    current_time: float,
    last_reroute: dict[str, float],
    max_vehicles: int,
    cooldown: int,
) -> tuple[int, int]:
    attempts = 0
    changed = 0
    # Prioritize slow vehicles near core-bound routes.
    candidates: list[tuple[float, str, dict]] = []
    try:
        vehicle_ids = list(eng.get_vehicles())
    except Exception:
        lane_vehicles = eng.get_lane_vehicles()
        vehicle_ids = sorted({vid for vids in lane_vehicles.values() for vid in vids})
    for veh_id in vehicle_ids:
        if current_time - last_reroute.get(veh_id, -1e12) < cooldown:
            continue
        try:
            info = eng.get_vehicle_info(veh_id)
        except Exception:
            continue
        route = [str(x) for x in str(info.get("route", "")).split() if x]
        if not route or not any(r in core_roads for r in route):
            continue
        cur = parse_vehicle_current_road(info)
        if cur is None or cur in core_roads:
            continue
        speed = float(info.get("speed", 0.0) or 0.0)
        heapq.heappush(candidates, (speed, veh_id, info))
    while candidates and attempts < max_vehicles:
        _, veh_id, info = heapq.heappop(candidates)
        route = [str(x) for x in str(info.get("route", "")).split() if x]
        cur = parse_vehicle_current_road(info)
        if cur is None or not route:
            continue
        goal = route[-1]
        new_route = shortest_path_avoid_core(adjacency, cur, goal, core_roads)
        attempts += 1
        last_reroute[veh_id] = current_time
        if new_route is None or len(new_route) <= 1 or new_route == route:
            continue
        try:
            ok = eng.set_vehicle_route(veh_id, new_route)
        except Exception:
            ok = False
        if ok is not False:
            changed += 1
    return attempts, changed


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.bucket_seconds <= 0 or args.decision_interval <= 0:
        raise SystemExit("duration, bucket-seconds, and decision-interval must be positive")
    if args.start_second < 0:
        raise SystemExit("--start-second must be non-negative")
    if args.perimeter_release_density > args.perimeter_target_density:
        raise SystemExit("--perimeter-release-density should be <= --perimeter-target-density")

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
        raise SystemExit("hierarchical control requires rlTrafficLight=true in the CityFlow config")
    interval = float(cfg.get("interval", 1.0))
    if interval <= 0:
        raise SystemExit(f"invalid CityFlow interval in config: {interval}")

    roadnet_path = resolve_cityflow_path(config_path, cfg, "roadnetFile")
    road_ids, lane_to_road_idx, road_meta, signals = load_roadnet(roadnet_path)
    road_geom, adjacency = load_roadnet_extra(roadnet_path)
    n_roads = len(road_ids)

    if args.perimeter_control:
        if args.core_mode == "score_npz":
            core_roads = choose_score_core(args, road_ids)
        else:
            core_roads = choose_bbox_core(road_geom, args.core_bbox_quantiles)
    else:
        core_roads = set()
    core_lane_ids = [
        lane_id for lane_id in lane_to_road_idx if lane_to_road_id(lane_id) in core_roads
    ]
    core_lane_km = sum(road_geom[r].lane_km for r in core_roads if r in road_geom)
    boundary = compute_boundary_control(signals, core_roads) if args.perimeter_control else BoundaryControl(set(), {}, {})
    target_stock = args.perimeter_target_density * core_lane_km
    release_stock = args.perimeter_release_density * core_lane_km
    pstate = PerimeterState()

    expected_buckets = int(math.ceil(args.duration / args.bucket_seconds))
    total_until = args.start_second + args.duration
    output_dir = Path(args.output_dir).expanduser().resolve()
    suffix = f"hier_mp_road_agg_{args.bucket_seconds}s_start{args.start_second}_dur{args.duration}"
    csv_path = output_dir / f"xuancheng_{args.date}_{suffix}.csv.gz"
    npz_path = output_dir / f"xuancheng_{args.date}_{suffix}.npz"
    summary_path = output_dir / f"xuancheng_{args.date}_{suffix}.summary.json"

    tensor = None
    if not args.no_npz:
        assert np is not None
        tensor = np.zeros((expected_buckets, n_roads, len(FEATURES)), dtype=np.float32)
        bucket_start_s = np.zeros((expected_buckets,), dtype=np.float32)
        bucket_end_s = np.zeros((expected_buckets,), dtype=np.float32)
        bucket_core_mean_active = np.zeros((expected_buckets,), dtype=np.float32)
        bucket_perimeter_active_fraction = np.zeros((expected_buckets,), dtype=np.float32)
    else:
        bucket_start_s = bucket_end_s = None
        bucket_core_mean_active = bucket_perimeter_active_fraction = None

    policy = "hierarchical_mp_perimeter" if args.perimeter_control else "paper_mp"
    print(f"config={config_path}")
    print(f"roadnet={roadnet_path}")
    print(f"roads={n_roads}, lanes={len(lane_to_road_idx)}, signals={len(signals)}")
    print(
        f"policy={policy} start={args.start_second}s duration={args.duration}s "
        f"bucket={args.bucket_seconds}s decision={args.decision_interval}s all_red={args.all_red_time}s"
    )
    print(
        "perimeter="
        f"{args.perimeter_control} core_roads={len(core_roads)} core_lane_km={core_lane_km:.3f} "
        f"boundary_signals={len(boundary.boundary_signals)} target_stock={target_stock:.2f} "
        f"release_stock={release_stock:.2f}"
    )
    print(f"route_guidance={args.route_guidance}")
    print(f"csv={csv_path}")
    if tensor is not None:
        print(f"npz={npz_path}")

    eng = cityflow.Engine(config_file=str(config_path), thread_num=args.thread_num)
    prev_lane_vehicles = {
        lane_id: set(eng.get_lane_vehicles().get(lane_id, [])) for lane_id in lane_to_road_idx
    }
    current_phase = {inter_id: 0 for inter_id in signals}
    current_phase_intervals = {inter_id: 0 for inter_id in signals}
    target_phase = {inter_id: 0 for inter_id in signals}
    transition_remaining = {inter_id: 0 for inter_id in signals}
    phase_change_count = 0
    decision_count = 0
    route_attempt_count = 0
    route_change_count = 0
    last_reroute: dict[str, float] = {}

    entered = [0] * n_roads
    exited = [0] * n_roads
    active_vehicle_seconds = [0.0] * n_roads
    speed_meter_seconds = [0.0] * n_roads
    speed_observation_seconds = [0.0] * n_roads
    bucket_core_vehicle_seconds = 0.0
    bucket_perimeter_active_seconds = 0.0
    bucket_idx = 0
    bucket_start = float(args.start_second)
    bucket_elapsed = 0.0
    current_time = 0.0

    with open_csv(csv_path) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "date", "bucket_index", "time_start_s", "time_end_s", "road_id",
                "entered_veh", "exited_veh", "mean_active_veh", "mean_speed_kmh",
                "density_veh_per_lane_km", "vehicle_seconds", "road_length_m",
                "lane_count", "speed_limit_kmh", "signal_policy",
            ]
        )

        def flush_bucket(time_end: float) -> None:
            nonlocal bucket_idx, bucket_start, bucket_elapsed
            nonlocal bucket_core_vehicle_seconds, bucket_perimeter_active_seconds
            if bucket_idx >= expected_buckets:
                return
            elapsed = max(bucket_elapsed, interval)
            if bucket_start_s is not None and bucket_end_s is not None:
                bucket_start_s[bucket_idx] = bucket_start
                bucket_end_s[bucket_idx] = time_end
            if bucket_core_mean_active is not None and bucket_perimeter_active_fraction is not None:
                bucket_core_mean_active[bucket_idx] = bucket_core_vehicle_seconds / elapsed
                bucket_perimeter_active_fraction[bucket_idx] = bucket_perimeter_active_seconds / elapsed
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
                        args.date, bucket_idx, f"{bucket_start:.3f}", f"{time_end:.3f}",
                        road_id, entered[road_idx], exited[road_idx], f"{mean_active:.6f}",
                        f"{mean_speed_kmh:.6f}", f"{density:.6f}", f"{veh_seconds:.6f}",
                        f"{meta['length_m']:.6f}", meta["lane_count"],
                        f"{meta['speed_limit_kmh']:.6f}", policy,
                    ]
                )
            bucket_idx += 1
            bucket_start = time_end
            bucket_elapsed = 0.0
            bucket_core_vehicle_seconds = 0.0
            bucket_perimeter_active_seconds = 0.0
            entered[:] = [0] * n_roads
            exited[:] = [0] * n_roads
            active_vehicle_seconds[:] = [0.0] * n_roads
            speed_meter_seconds[:] = [0.0] * n_roads
            speed_observation_seconds[:] = [0.0] * n_roads

        while current_time < total_until and bucket_idx < expected_buckets:
            lane_vehicle_snapshot = None
            if int(current_time) % args.decision_interval == 0:
                lane_vehicle_snapshot = eng.get_lane_vehicles()
                lane_vehicle_counts = {lane_id: len(vehicles) for lane_id, vehicles in lane_vehicle_snapshot.items()}
                actions = choose_max_pressure_actions(
                    signals, lane_vehicle_counts, current_phase,
                    current_phase_intervals, args.max_same_phase_intervals,
                )
                core_stock = sum(lane_vehicle_counts.get(lane_id, 0) for lane_id in core_lane_ids)
                if args.perimeter_control:
                    if pstate.active:
                        if core_stock <= release_stock:
                            pstate.active = False
                    elif core_stock >= target_stock:
                        pstate.active = True
                        pstate.activation_count += 1
                    actions = apply_perimeter_overrides(
                        actions, signals, lane_vehicle_counts, boundary, pstate,
                        decision_count + 1, args.perimeter_open_every_n,
                    )
                if args.route_guidance and args.perimeter_control and pstate.active:
                    attempts, changed = maybe_apply_route_guidance(
                        eng, adjacency, core_roads, current_time, last_reroute,
                        args.route_guidance_max_vehicles_per_decision,
                        args.route_guidance_cooldown,
                    )
                    route_attempt_count += attempts
                    route_change_count += changed
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
                core_stock_now = 0
                for lane_id, road_idx in lane_to_road_idx.items():
                    current_set = current_lane_vehicles[lane_id]
                    previous_set = prev_lane_vehicles.get(lane_id, set())
                    entered[road_idx] += len(current_set - previous_set)
                    exited[road_idx] += len(previous_set - current_set)
                    active_count = len(current_set)
                    active_vehicle_seconds[road_idx] += active_count * interval
                    if lane_id in core_lane_ids:
                        core_stock_now += active_count
                    if active_count:
                        for vehicle_id in current_set:
                            speed = vehicle_speed.get(vehicle_id)
                            if speed is not None and speed >= 0:
                                speed_meter_seconds[road_idx] += float(speed) * interval
                                speed_observation_seconds[road_idx] += interval
                bucket_core_vehicle_seconds += core_stock_now * interval
                if pstate.active:
                    bucket_perimeter_active_seconds += interval
                bucket_elapsed += interval
                if current_time >= bucket_start + args.bucket_seconds or current_time >= total_until:
                    flush_bucket(min(current_time, float(total_until)))
            prev_lane_vehicles = current_lane_vehicles

            if args.progress_interval > 0 and int(current_time) % args.progress_interval == 0:
                print(
                    f"[progress] sim_t={current_time:.0f}s agg_buckets={bucket_idx}/{expected_buckets} "
                    f"decisions={decision_count} perimeter_active={int(pstate.active)} "
                    f"overrides={pstate.override_count} reroutes={route_change_count}"
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
            bucket_core_mean_active=bucket_core_mean_active[:bucket_idx],
            bucket_perimeter_active_fraction=bucket_perimeter_active_fraction[:bucket_idx],
            date=np.array(args.date),
            source_config=np.array(str(config_path)),
            signal_policy=np.array(policy),
            start_second=np.array(args.start_second),
            decision_interval=np.array(args.decision_interval),
            all_red_time=np.array(args.all_red_time),
            perimeter_target_density=np.array(args.perimeter_target_density),
            perimeter_release_density=np.array(args.perimeter_release_density),
        )

    summary = {
        "date": args.date,
        "policy": policy,
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
        "perimeter_control": bool(args.perimeter_control),
        "core_mode": args.core_mode,
        "core_roads": len(core_roads),
        "core_lane_km": core_lane_km,
        "boundary_signals": len(boundary.boundary_signals),
        "perimeter_target_density": args.perimeter_target_density,
        "perimeter_release_density": args.perimeter_release_density,
        "perimeter_open_every_n": args.perimeter_open_every_n,
        "perimeter_activation_count": pstate.activation_count,
        "perimeter_override_count": pstate.override_count,
        "perimeter_all_red_count": pstate.all_red_count,
        "perimeter_open_release_count": pstate.open_release_count,
        "core_score_npz": args.core_score_npz if args.core_mode == "score_npz" else None,
        "core_score_start_second": args.core_score_start_second if args.core_mode == "score_npz" else None,
        "core_score_duration": args.core_score_duration if args.core_mode == "score_npz" else None,
        "core_score_topk": args.core_score_topk if args.core_mode == "score_npz" else None,
        "route_guidance": bool(args.route_guidance),
        "route_attempt_count": route_attempt_count,
        "route_change_count": route_change_count,
        "csv": str(csv_path),
        "npz": str(npz_path) if tensor is not None else None,
        "official_alignment_note": (
            "MVP implementation aligned with the paper's hierarchical-control concept: "
            "max-pressure signal control plus network-level perimeter gating. The public "
            "repository does not ship a full runnable perimeter/route-guidance agent, so "
            "the upper-layer controller is a reported heuristic, not an official as-is reproduction."
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

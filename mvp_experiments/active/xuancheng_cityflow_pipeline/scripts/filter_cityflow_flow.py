#!/usr/bin/env python3
"""Filter and optionally expand CityFlow flow routes against the roadnet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path


def daily_name(date_text: str) -> str:
    return f"data_{date_text.replace('-', '_')}_type_filtered.json"


def default_output_name(date_text: str) -> str:
    return f"data_{date_text.replace('-', '_')}_type_filtered.valid.json"


def load_roadnet(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    with path.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)
    road_ids = {str(road["id"]) for road in roadnet["roads"]}
    adjacency = {road_id: set() for road_id in road_ids}
    for inter in roadnet["intersections"]:
        for link in inter.get("roadLinks", []):
            start = str(link["startRoad"])
            end = str(link["endRoad"])
            if start in adjacency:
                adjacency[start].add(end)
    return road_ids, adjacency


def find_path(
    start: str,
    end: str,
    adjacency: dict[str, set[str]],
    path_cache: dict[tuple[str, str], list[str] | None],
) -> list[str] | None:
    key = (start, end)
    if key in path_cache:
        return path_cache[key]
    if start == end:
        path_cache[key] = [start]
        return path_cache[key]

    queue = deque([start])
    visited = {start}
    parent: dict[str, str] = {}
    while queue:
        road_id = queue.popleft()
        for nxt in adjacency.get(road_id, ()):
            if nxt == end:
                parent[nxt] = road_id
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                path_cache[key] = path
                return path_cache[key]
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = road_id
                queue.append(nxt)
    path_cache[key] = None
    return None


def validate_and_expand_route(
    route: object,
    road_ids: set[str],
    adjacency: dict[str, set[str]],
    path_cache: dict[tuple[str, str], list[str] | None],
    expand_routes: bool,
) -> tuple[str | None, list[str] | None, bool]:
    if not isinstance(route, list) or not route:
        return "empty_or_nonlist_route", None, False
    route_ids = [str(road_id) for road_id in route]
    for road_id in route_ids:
        if road_id not in road_ids:
            return "unknown_road", None, False
    expanded = [route_ids[0]]
    changed = False
    for start, end in zip(route_ids, route_ids[1:]):
        path = find_path(start, end, adjacency, path_cache)
        if path is None:
            return "unreachable_anchor", None, False
        if len(path) > 2:
            changed = True
        if len(path) == 1:
            continue
        expanded.extend(path[1:])
    if not expand_routes:
        return None, route_ids, changed
    return None, expanded, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root with raw/ downloaded files.")
    parser.add_argument("--date", required=True, help="Date such as 2023-04-03.")
    parser.add_argument("--roadnet", help="Roadnet JSON path. Default: <data-root>/raw/roadnet_xuancheng250319.json.")
    parser.add_argument("--flow", help="Input flow JSON. Default: <data-root>/raw/data_YYYY_MM_DD_type_filtered.json.")
    parser.add_argument("--output", help="Filtered flow JSON. Default: <data-root>/raw/data_YYYY_MM_DD_type_filtered.valid.json.")
    parser.add_argument("--summary", help="Summary JSON. Default: <output>.summary.json.")
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Only filter unreachable anchor routes; do not expand anchors to full shortest paths.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing filtered flow.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    roadnet_path = Path(args.roadnet).expanduser().resolve() if args.roadnet else raw_dir / "roadnet_xuancheng250319.json"
    flow_path = Path(args.flow).expanduser().resolve() if args.flow else raw_dir / daily_name(args.date)
    output_path = Path(args.output).expanduser().resolve() if args.output else raw_dir / default_output_name(args.date)
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")

    if output_path.exists() and summary_path.exists() and not args.force:
        print(f"[skip] filtered flow already exists: {output_path}")
        print(f"[skip] summary: {summary_path}")
        return 0

    road_ids, adjacency = load_roadnet(roadnet_path)
    with flow_path.open("r", encoding="utf-8") as f:
        flows = json.load(f)

    kept = []
    reasons: Counter[str] = Counter()
    path_cache: dict[tuple[str, str], list[str] | None] = {}
    expanded_records = 0
    original_route_lengths = []
    output_route_lengths = []
    for flow in flows:
        reason, route, changed = validate_and_expand_route(
            flow.get("route") if isinstance(flow, dict) else None,
            road_ids,
            adjacency,
            path_cache,
            expand_routes=not args.no_expand,
        )
        if reason is None:
            if not isinstance(flow, dict):
                reasons["non_dict_flow"] += 1
                continue
            flow_out = dict(flow)
            assert route is not None
            original_route_lengths.append(len(flow["route"]))
            output_route_lengths.append(len(route))
            if changed:
                expanded_records += 1
            flow_out["route"] = route
            kept.append(flow_out)
        else:
            reasons[reason] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, separators=(",", ":"))
    tmp_path.replace(output_path)

    summary = {
        "date": args.date,
        "roadnet": str(roadnet_path),
        "input_flow": str(flow_path),
        "output_flow": str(output_path),
        "input_records": len(flows),
        "kept_records": len(kept),
        "dropped_records": len(flows) - len(kept),
        "drop_reasons": dict(sorted(reasons.items())),
        "expand_routes": not args.no_expand,
        "expanded_records": expanded_records,
        "unique_anchor_pairs_checked": len(path_cache),
        "mean_input_route_length": sum(original_route_lengths) / len(original_route_lengths)
        if original_route_lengths
        else 0,
        "mean_output_route_length": sum(output_route_lengths) / len(output_route_lengths)
        if output_route_lengths
        else 0,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

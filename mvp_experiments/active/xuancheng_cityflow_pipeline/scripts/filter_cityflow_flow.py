#!/usr/bin/env python3
"""Filter CityFlow flow records whose routes are not connected in the roadnet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def daily_name(date_text: str) -> str:
    return f"data_{date_text.replace('-', '_')}_type_filtered.json"


def default_output_name(date_text: str) -> str:
    return f"data_{date_text.replace('-', '_')}_type_filtered.valid.json"


def load_roadnet(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    with path.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)
    road_ids = {str(road["id"]) for road in roadnet["roads"]}
    transitions: set[tuple[str, str]] = set()
    for inter in roadnet["intersections"]:
        for link in inter.get("roadLinks", []):
            transitions.add((str(link["startRoad"]), str(link["endRoad"])))
    return road_ids, transitions


def invalid_reason(route: object, road_ids: set[str], transitions: set[tuple[str, str]]) -> str | None:
    if not isinstance(route, list) or not route:
        return "empty_or_nonlist_route"
    route_ids = [str(road_id) for road_id in route]
    for road_id in route_ids:
        if road_id not in road_ids:
            return "unknown_road"
    for start, end in zip(route_ids, route_ids[1:]):
        if (start, end) not in transitions:
            return "disconnected_turn"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root with raw/ downloaded files.")
    parser.add_argument("--date", required=True, help="Date such as 2023-04-03.")
    parser.add_argument("--roadnet", help="Roadnet JSON path. Default: <data-root>/raw/roadnet_xuancheng250319.json.")
    parser.add_argument("--flow", help="Input flow JSON. Default: <data-root>/raw/data_YYYY_MM_DD_type_filtered.json.")
    parser.add_argument("--output", help="Filtered flow JSON. Default: <data-root>/raw/data_YYYY_MM_DD_type_filtered.valid.json.")
    parser.add_argument("--summary", help="Summary JSON. Default: <output>.summary.json.")
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

    road_ids, transitions = load_roadnet(roadnet_path)
    with flow_path.open("r", encoding="utf-8") as f:
        flows = json.load(f)

    kept = []
    reasons: Counter[str] = Counter()
    for flow in flows:
        reason = invalid_reason(flow.get("route") if isinstance(flow, dict) else None, road_ids, transitions)
        if reason is None:
            kept.append(flow)
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
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

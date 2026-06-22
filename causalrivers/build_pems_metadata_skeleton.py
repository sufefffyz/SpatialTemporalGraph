#!/usr/bin/env python3
"""Build a metadata-only skeleton graph for PeMS sensors.

This script intentionally uses only the Python standard library so it can run in
lightweight environments. It does not claim to recover the true road graph.
Instead, it produces a directional backbone from PeMS metadata that can later be
reviewed against SHN / OSM / OSRM outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a metadata-only PeMS graph skeleton.")
    parser.add_argument(
        "--meta-dir",
        default="data/PEMSD3/station_meta",
        help="Directory containing d03_text_meta_YYYY_MM_DD.txt snapshots.",
    )
    parser.add_argument(
        "--reference-date",
        default="2025-06-01",
        help="Choose the metadata snapshot closest to and not after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--sensor-types",
        nargs="+",
        default=["ML"],
        help="Sensor types to retain as graph nodes. Defaults to ML only.",
    )
    parser.add_argument(
        "--max-pm-gap",
        type=float,
        default=2.0,
        help="Flag edges whose adjacent absolute postmile gap is larger than this threshold.",
    )
    parser.add_argument(
        "--max-geo-gap-m",
        type=float,
        default=8000.0,
        help="Flag edges whose geodesic distance exceeds this threshold in meters.",
    )
    parser.add_argument(
        "--output-dir",
        default="SpatialTemporalGraph/causalrivers/pems_metadata_skeleton",
        help="Directory where nodes.csv / edges.csv / manifest.json will be written.",
    )
    return parser.parse_args()


def parse_snapshot_date(path: Path) -> datetime | None:
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})", path.name)
    if match is None:
        return None
    return datetime(*map(int, match.groups()))


def pick_metadata_snapshot(meta_dir: Path, reference_date: datetime) -> tuple[Path, datetime]:
    candidates = []
    for path in sorted(meta_dir.glob("d03_text_meta_*.txt")):
        snapshot_date = parse_snapshot_date(path)
        if snapshot_date is not None:
            candidates.append((snapshot_date, path))
    if not candidates:
        raise FileNotFoundError(f"No d03_text_meta_*.txt files found under {meta_dir}")

    eligible = [item for item in candidates if item[0] <= reference_date]
    if eligible:
        chosen_date, chosen_path = max(eligible, key=lambda item: item[0])
    else:
        chosen_date, chosen_path = min(candidates, key=lambda item: item[0])
    return chosen_path, chosen_date


def maybe_float(value: str) -> float | None:
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def load_rows(path: Path, allowed_types: set[str]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sensor_type = str(row.get("Type", "")).strip().upper()
            if sensor_type not in allowed_types:
                continue

            lat = maybe_float(row.get("Latitude", ""))
            lon = maybe_float(row.get("Longitude", ""))
            if lat is None or lon is None:
                continue

            abs_pm = maybe_float(row.get("Abs_PM", ""))
            direction = str(row.get("Dir", "")).strip().upper()
            fwy = str(row.get("Fwy", "")).strip()
            sensor_id = str(row.get("ID", "")).strip()
            if not sensor_id or not fwy or direction not in {"N", "S", "E", "W"}:
                continue

            rows.append(
                {
                    "ID": sensor_id,
                    "Fwy": fwy,
                    "Dir": direction,
                    "District": str(row.get("District", "")).strip(),
                    "County": str(row.get("County", "")).strip(),
                    "City": str(row.get("City", "")).strip(),
                    "State_PM": str(row.get("State_PM", "")).strip(),
                    "Abs_PM": abs_pm,
                    "Latitude": lat,
                    "Longitude": lon,
                    "Length": maybe_float(row.get("Length", "")),
                    "Type": sensor_type,
                    "Lanes": str(row.get("Lanes", "")).strip(),
                    "Name": str(row.get("Name", "")).strip(),
                }
            )
    return rows


def travel_sort_key(direction: str, abs_pm: float | None):
    if abs_pm is None:
        return (1, 0.0)
    if direction in {"N", "E"}:
        return (0, abs_pm)
    return (0, -abs_pm)


def quality_marker(value_present: bool) -> int:
    return 0 if value_present else -1


def build_edges(rows: list[dict], max_pm_gap: float, max_geo_gap_m: float) -> tuple[list[dict], list[dict]]:
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["Fwy"], row["Dir"])].append(row)

    edges = []
    suspicious = []
    for (fwy, direction), group_rows in sorted(by_group.items()):
        ordered = sorted(group_rows, key=lambda row: travel_sort_key(direction, row["Abs_PM"]))
        for source, target in zip(ordered[:-1], ordered[1:]):
            pm_gap = None
            if source["Abs_PM"] is not None and target["Abs_PM"] is not None:
                pm_gap = abs(target["Abs_PM"] - source["Abs_PM"])

            geo_gap_m = haversine_m(
                source["Latitude"],
                source["Longitude"],
                target["Latitude"],
                target["Longitude"],
            )

            needs_external_validation = (
                pm_gap is None
                or pm_gap > max_pm_gap
                or geo_gap_m > max_geo_gap_m
            )

            edge = {
                "source_id": source["ID"],
                "target_id": target["ID"],
                "fwy": fwy,
                "dir": direction,
                "source_abs_pm": "" if source["Abs_PM"] is None else f"{source['Abs_PM']:.6f}",
                "target_abs_pm": "" if target["Abs_PM"] is None else f"{target['Abs_PM']:.6f}",
                "abs_pm_gap": "" if pm_gap is None else f"{pm_gap:.6f}",
                "geo_distance_m": f"{geo_gap_m:.3f}",
                "source_name": source["Name"],
                "target_name": target["Name"],
                "source_type": source["Type"],
                "target_type": target["Type"],
                "edge_origin": "metadata_abs_pm_chain",
                "quality_pm": quality_marker(pm_gap is not None),
                "quality_geo": quality_marker(True),
                "quality_external": -1,
                "needs_external_validation": int(needs_external_validation),
            }
            edges.append(edge)
            if needs_external_validation:
                suspicious.append(edge)

    return edges, suspicious


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d")
    sensor_types = {sensor_type.strip().upper() for sensor_type in args.sensor_types if sensor_type.strip()}

    meta_path, snapshot_date = pick_metadata_snapshot(meta_dir, reference_date)
    rows = load_rows(meta_path, sensor_types)
    if not rows:
        raise ValueError(f"No valid rows found in {meta_path} for sensor types {sorted(sensor_types)}")

    edges, suspicious_edges = build_edges(rows, max_pm_gap=args.max_pm_gap, max_geo_gap_m=args.max_geo_gap_m)

    rows_sorted = sorted(rows, key=lambda row: (row["Fwy"], row["Dir"], travel_sort_key(row["Dir"], row["Abs_PM"]), row["ID"]))
    node_fieldnames = [
        "ID",
        "Fwy",
        "Dir",
        "District",
        "County",
        "City",
        "State_PM",
        "Abs_PM",
        "Latitude",
        "Longitude",
        "Length",
        "Type",
        "Lanes",
        "Name",
    ]
    edge_fieldnames = [
        "source_id",
        "target_id",
        "fwy",
        "dir",
        "source_abs_pm",
        "target_abs_pm",
        "abs_pm_gap",
        "geo_distance_m",
        "source_name",
        "target_name",
        "source_type",
        "target_type",
        "edge_origin",
        "quality_pm",
        "quality_geo",
        "quality_external",
        "needs_external_validation",
    ]

    write_csv(output_dir / "nodes.csv", rows_sorted, node_fieldnames)
    write_csv(output_dir / "edges.csv", edges, edge_fieldnames)
    write_csv(output_dir / "suspicious_edges.csv", suspicious_edges, edge_fieldnames)

    by_group = Counter((row["Fwy"], row["Dir"]) for row in rows_sorted)
    manifest = {
        "reference_date": args.reference_date,
        "selected_snapshot": meta_path.name,
        "selected_snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "sensor_types": sorted(sensor_types),
        "num_nodes": len(rows_sorted),
        "num_edges": len(edges),
        "num_suspicious_edges": len(suspicious_edges),
        "num_freeway_direction_groups": len(by_group),
        "max_pm_gap": args.max_pm_gap,
        "max_geo_gap_m": args.max_geo_gap_m,
        "warning": (
            "This is a metadata-only skeleton graph. It is useful for candidate generation and review, "
            "but it is not a verified road graph or a causal graph."
        ),
        "next_step": (
            "Validate suspicious edges and freeway-direction chains against SHN / OSM / OSRM, "
            "then export the reviewed graph into the causalrivers product format."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Snapshot: {meta_path.name}")
    print(f"Sensor types: {sorted(sensor_types)}")
    print(f"Nodes: {len(rows_sorted)}")
    print(f"Edges: {len(edges)}")
    print(f"Suspicious edges needing external validation: {len(suspicious_edges)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

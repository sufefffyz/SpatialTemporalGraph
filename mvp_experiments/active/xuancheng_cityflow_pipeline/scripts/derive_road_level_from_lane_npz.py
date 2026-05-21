#!/usr/bin/env python3
"""Derive road-level Xuancheng tensors and metadata from lane-level NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROAD_AGGREGATION_NOTES = {
    "entered_veh": "sum over lanes belonging to the same road",
    "exited_veh": "sum over lanes belonging to the same road",
    "mean_active_veh": "sum over lane mean active vehicles",
    "mean_speed_kmh": "mean_active_veh weighted average over lanes",
    "density_veh_per_km": "recomputed as road mean_active_veh divided by summed lane km",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Xuancheng lane-level daily NPZ files into road-level daily NPZ "
            "files and write lane/road metadata sidecars."
        )
    )
    parser.add_argument("--lane-dir", required=True, help="Directory with lane-level daily NPZ files.")
    parser.add_argument("--road-output-dir", required=True, help="Directory for derived road-level NPZ files.")
    parser.add_argument(
        "--metadata-dir",
        required=True,
        help="Directory for lane_metadata.csv, road_metadata.csv, and feature_metadata.csv.",
    )
    parser.add_argument("--manifest", required=True, help="Path to write dataset view manifest JSON.")
    parser.add_argument(
        "--pattern",
        default="xuancheng_*_lane_agg_60s.npz",
        help="Glob pattern for lane-level NPZ files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing derived road-level NPZ files.",
    )
    return parser.parse_args()


def scalar_to_str(value: np.ndarray) -> str:
    if value.shape == ():
        return str(value.tolist())
    raise ValueError(f"expected scalar array, got shape {value.shape}")


def infer_date(path: Path, archive: np.lib.npyio.NpzFile) -> str:
    if "date" in archive:
        return scalar_to_str(archive["date"])
    match = re.search(r"xuancheng_(\d{4}-\d{2}-\d{2})_", path.name)
    if match:
        return match.group(1)
    raise ValueError(f"cannot infer date from {path}")


def build_index(lane_road_ids: np.ndarray) -> tuple[list[str], np.ndarray]:
    road_ids: list[str] = []
    road_to_idx: dict[str, int] = {}
    inverse = np.empty(len(lane_road_ids), dtype=np.int32)
    for lane_idx, road_id_value in enumerate(lane_road_ids):
        road_id = str(road_id_value)
        road_idx = road_to_idx.get(road_id)
        if road_idx is None:
            road_idx = len(road_ids)
            road_to_idx[road_id] = road_idx
            road_ids.append(road_id)
        inverse[lane_idx] = road_idx
    return road_ids, inverse


def road_metadata_from_lane(
    lane_ids: np.ndarray,
    lane_road_ids: np.ndarray,
    lane_indices: np.ndarray,
    lane_length_m: np.ndarray,
    speed_limit_kmh: np.ndarray,
    road_ids: list[str],
    inverse: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lane_rows: list[dict[str, Any]] = []
    road_rows: list[dict[str, Any]] = []
    road_lane_count = np.bincount(inverse, minlength=len(road_ids)).astype(np.int32)
    road_lane_km = np.zeros(len(road_ids), dtype=np.float64)
    road_speed_sum = np.zeros(len(road_ids), dtype=np.float64)
    road_length_m = np.zeros(len(road_ids), dtype=np.float64)

    for lane_idx, road_idx in enumerate(inverse):
        length_m = float(lane_length_m[lane_idx])
        lane_rows.append(
            {
                "lane_idx": lane_idx,
                "lane_id": str(lane_ids[lane_idx]),
                "road_id": str(lane_road_ids[lane_idx]),
                "lane_index": int(lane_indices[lane_idx]),
                "lane_length_m": length_m,
                "lane_km": length_m / 1000.0,
                "speed_limit_kmh": float(speed_limit_kmh[lane_idx]),
            }
        )
        road_lane_km[road_idx] += length_m / 1000.0
        road_speed_sum[road_idx] += float(speed_limit_kmh[lane_idx])
        if road_length_m[road_idx] == 0.0:
            road_length_m[road_idx] = length_m

    road_speed_limit = np.divide(
        road_speed_sum,
        road_lane_count,
        out=np.zeros_like(road_speed_sum),
        where=road_lane_count > 0,
    )
    for road_idx, road_id in enumerate(road_ids):
        road_rows.append(
            {
                "road_idx": road_idx,
                "road_id": road_id,
                "lane_count": int(road_lane_count[road_idx]),
                "road_length_m": float(road_length_m[road_idx]),
                "lane_km": float(road_lane_km[road_idx]),
                "speed_limit_kmh": float(road_speed_limit[road_idx]),
            }
        )
    return lane_rows, road_rows


def aggregate_lane_to_road(
    lane_data: np.ndarray,
    feature_names: list[str],
    inverse: np.ndarray,
    road_lane_km: np.ndarray,
) -> np.ndarray:
    required = set(ROAD_AGGREGATION_NOTES)
    missing = required - set(feature_names)
    if missing:
        raise ValueError(f"missing required lane features: {sorted(missing)}")

    n_time, _, n_features = lane_data.shape
    n_roads = len(road_lane_km)
    road_data = np.zeros((n_time, n_roads, n_features), dtype=np.float64)
    speed_num = np.zeros((n_time, n_roads), dtype=np.float64)
    speed_den = np.zeros((n_time, n_roads), dtype=np.float64)

    feature_idx = {name: idx for idx, name in enumerate(feature_names)}
    entered_idx = feature_idx["entered_veh"]
    exited_idx = feature_idx["exited_veh"]
    active_idx = feature_idx["mean_active_veh"]
    speed_idx = feature_idx["mean_speed_kmh"]
    density_idx = feature_idx["density_veh_per_km"]

    for lane_idx, road_idx in enumerate(inverse):
        lane_slice = lane_data[:, lane_idx, :].astype(np.float64, copy=False)
        road_data[:, road_idx, entered_idx] += lane_slice[:, entered_idx]
        road_data[:, road_idx, exited_idx] += lane_slice[:, exited_idx]
        road_data[:, road_idx, active_idx] += lane_slice[:, active_idx]
        weights = lane_slice[:, active_idx]
        speed_num[:, road_idx] += lane_slice[:, speed_idx] * weights
        speed_den[:, road_idx] += weights

    road_data[:, :, speed_idx] = np.divide(
        speed_num,
        speed_den,
        out=np.zeros_like(speed_num),
        where=speed_den > 0,
    )
    road_data[:, :, density_idx] = np.divide(
        road_data[:, :, active_idx],
        road_lane_km[None, :],
        out=np.zeros_like(road_data[:, :, active_idx]),
        where=road_lane_km[None, :] > 0,
    )
    return road_data.astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_metadata(path: Path, feature_names: list[str]) -> None:
    rows = [
        {
            "feature_idx": feature_idx,
            "feature_name": feature_name,
            "road_from_lane_aggregation": ROAD_AGGREGATION_NOTES.get(feature_name, "copied/unspecified"),
        }
        for feature_idx, feature_name in enumerate(feature_names)
    ]
    write_csv(path, rows)


def process_file(
    lane_path: Path,
    road_output_dir: Path,
    overwrite: bool,
    reference_signature: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(lane_path, allow_pickle=False) as archive:
        required_keys = {
            "data",
            "lane_ids",
            "road_ids",
            "lane_indices",
            "lane_length_m",
            "speed_limit_kmh",
            "feature_names",
            "bucket_start_s",
            "bucket_end_s",
        }
        missing = required_keys - set(archive.files)
        if missing:
            raise ValueError(f"{lane_path} is missing required keys: {sorted(missing)}")

        lane_data = archive["data"]
        lane_ids = archive["lane_ids"]
        lane_road_ids = archive["road_ids"]
        lane_indices = archive["lane_indices"]
        lane_length_m = archive["lane_length_m"].astype(np.float64)
        speed_limit_kmh = archive["speed_limit_kmh"].astype(np.float64)
        feature_names = [str(value) for value in archive["feature_names"].tolist()]
        bucket_start_s = archive["bucket_start_s"]
        bucket_end_s = archive["bucket_end_s"]
        date = infer_date(lane_path, archive)

        road_ids, inverse = build_index(lane_road_ids)
        lane_rows, road_rows = road_metadata_from_lane(
            lane_ids,
            lane_road_ids,
            lane_indices,
            lane_length_m,
            speed_limit_kmh,
            road_ids,
            inverse,
        )
        road_lane_km = np.array([row["lane_km"] for row in road_rows], dtype=np.float64)

        signature = {
            "lane_ids": lane_ids.tolist(),
            "lane_road_ids": lane_road_ids.tolist(),
            "lane_indices": lane_indices.tolist(),
            "lane_length_m": lane_length_m.tolist(),
            "speed_limit_kmh": speed_limit_kmh.tolist(),
            "feature_names": feature_names,
            "road_ids": road_ids,
        }
        if reference_signature is not None and signature != reference_signature:
            raise ValueError(f"{lane_path} metadata order differs from the first file")

        road_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = road_output_dir / lane_path.name.replace("_lane_agg_", "_road_agg_from_lane_")
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} already exists; pass --overwrite")

        road_data = aggregate_lane_to_road(lane_data, feature_names, inverse, road_lane_km)
        np.savez_compressed(
            out_path,
            data=road_data,
            road_ids=np.array(road_ids),
            lane_count=np.array([row["lane_count"] for row in road_rows], dtype=np.int16),
            road_length_m=np.array([row["road_length_m"] for row in road_rows], dtype=np.float32),
            lane_km=np.array([row["lane_km"] for row in road_rows], dtype=np.float32),
            speed_limit_kmh=np.array([row["speed_limit_kmh"] for row in road_rows], dtype=np.float32),
            feature_names=np.array(feature_names),
            bucket_start_s=bucket_start_s,
            bucket_end_s=bucket_end_s,
            date=np.array(date),
            source_lane_npz=np.array(str(lane_path)),
            aggregation=np.array("lane_to_road_v1_sum_counts_weighted_speed"),
        )

    file_summary = {
        "date": date,
        "lane_npz": str(lane_path),
        "road_npz": str(out_path),
        "lane_shape": list(lane_data.shape),
        "road_shape": list(road_data.shape),
    }
    sidecar_payload = {
        "signature": signature,
        "lane_rows": lane_rows,
        "road_rows": road_rows,
        "feature_names": feature_names,
    }
    return file_summary, sidecar_payload


def main() -> int:
    args = parse_args()
    lane_dir = Path(args.lane_dir).expanduser().resolve()
    road_output_dir = Path(args.road_output_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    lane_files = sorted(lane_dir.glob(args.pattern))
    if not lane_files:
        raise SystemExit(f"no files match {lane_dir / args.pattern}")

    file_summaries: list[dict[str, Any]] = []
    reference_signature: dict[str, Any] | None = None
    sidecar_payload: dict[str, Any] | None = None

    for lane_path in lane_files:
        summary, payload = process_file(
            lane_path,
            road_output_dir,
            args.overwrite,
            reference_signature,
        )
        if reference_signature is None:
            reference_signature = payload["signature"]
            sidecar_payload = payload
        file_summaries.append(summary)
        print(f"[done] {summary['date']} {summary['lane_shape']} -> {summary['road_shape']}")

    assert sidecar_payload is not None
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metadata_dir / "lane_metadata.csv", sidecar_payload["lane_rows"])
    write_csv(metadata_dir / "road_metadata.csv", sidecar_payload["road_rows"])
    write_feature_metadata(metadata_dir / "feature_metadata.csv", sidecar_payload["feature_names"])

    manifest = {
        "schema_version": "xuancheng_cityflow_dataset_views_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "CityFlow official raw flow, lane-level aggregation generated by this MVP pipeline",
        "note": (
            "The road-level view in road_level.data_dir is derived offline from the lane-level NPZ "
            "files, not an independent CityFlow rerun."
        ),
        "lane_level": {
            "data_dir": str(lane_dir),
            "file_pattern": args.pattern,
            "metadata_csv": str(metadata_dir / "lane_metadata.csv"),
            "num_lanes": len(sidecar_payload["lane_rows"]),
        },
        "road_level": {
            "data_dir": str(road_output_dir),
            "file_pattern": "xuancheng_*_road_agg_from_lane_60s.npz",
            "metadata_csv": str(metadata_dir / "road_metadata.csv"),
            "num_roads": len(sidecar_payload["road_rows"]),
            "aggregation": ROAD_AGGREGATION_NOTES,
        },
        "features_csv": str(metadata_dir / "feature_metadata.csv"),
        "dates": [summary["date"] for summary in file_summaries],
        "files": file_summaries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] metadata_dir={metadata_dir}")
    print(f"[done] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

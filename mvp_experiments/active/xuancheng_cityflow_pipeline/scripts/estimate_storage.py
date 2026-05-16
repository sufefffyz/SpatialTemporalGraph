#!/usr/bin/env python3
"""Estimate Xuancheng release and road-tensor storage requirements."""

from __future__ import annotations

import argparse
import math


RAW_DAILY_BYTES = {
    "2023-04-01": 200210220,
    "2023-04-02": 198598785,
    "2023-04-03": 213008752,
    "2023-04-04": 213003967,
    "2023-04-05": 157667232,
    "2023-04-06": 207191063,
    "2023-04-07": 200835137,
    "2023-04-08": 183346329,
    "2023-04-09": 183346329,
    "2023-04-10": 185896032,
    "2023-04-11": 182439202,
    "2023-04-12": 182492671,
    "2023-04-13": 187626778,
    "2023-04-14": 206410636,
    "2023-04-15": 183803420,
    "2023-04-16": 182276005,
    "2023-04-17": 182419815,
    "2023-04-18": 183357370,
    "2023-04-19": 181466139,
    "2023-04-20": 177821055,
    "2023-04-21": 189829723,
    "2023-04-22": 185105699,
    "2023-04-23": 203079882,
    "2023-04-24": 200078843,
    "2023-04-25": 192088174,
    "2023-04-26": 181749771,
    "2023-04-27": 181080564,
    "2023-04-28": 235129894,
    "2023-04-29": 237073449,
    "2023-04-30": 205586359,
}


def human_size(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roads", type=int, default=1744)
    parser.add_argument("--lanes", type=int, default=3546)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--bucket-seconds", type=int, default=60)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--csv-bytes-per-row", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    buckets_per_day = math.ceil(86400 / args.bucket_seconds)
    raw_total = sum(RAW_DAILY_BYTES.values())
    road_tensor_bytes = args.days * buckets_per_day * args.roads * args.features * 4
    lane_tensor_bytes = args.days * buckets_per_day * args.lanes * args.features * 4
    road_csv_bytes = args.days * buckets_per_day * args.roads * args.csv_bytes_per_row
    lane_csv_bytes = args.days * buckets_per_day * args.lanes * args.csv_bytes_per_row

    print("Xuancheng official release coverage")
    print(f"  dates: {min(RAW_DAILY_BYTES)}..{max(RAW_DAILY_BYTES)}")
    print(f"  daily files: {len(RAW_DAILY_BYTES)}")
    print(f"  raw daily flow JSON total: {human_size(raw_total)}")
    print()
    print("Dense float32 tensor estimates")
    print(f"  bucket_seconds: {args.bucket_seconds}")
    print(f"  buckets_per_day: {buckets_per_day}")
    print(f"  road tensor [{args.days * buckets_per_day}, {args.roads}, {args.features}]: {human_size(road_tensor_bytes)}")
    print(f"  lane tensor [{args.days * buckets_per_day}, {args.lanes}, {args.features}]: {human_size(lane_tensor_bytes)}")
    print()
    print("Long CSV rough uncompressed estimates")
    print(f"  assumed bytes per row: {args.csv_bytes_per_row}")
    print(f"  road CSV: {human_size(road_csv_bytes)}")
    print(f"  lane CSV: {human_size(lane_csv_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

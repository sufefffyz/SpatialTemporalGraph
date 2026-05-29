#!/usr/bin/env python3
"""Create a BasicTS dataset with time-of-day and day-of-week covariates."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dataset", required=True, help="Input BasicTS dataset directory.")
    parser.add_argument("--output-dataset", required=True, help="Output BasicTS dataset directory.")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD start date.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")


def infer_start_date(desc: dict[str, Any], override: str | None) -> datetime:
    if override:
        return datetime.fromisoformat(override)
    dates = desc.get("source_dates") or []
    if dates:
        return datetime.fromisoformat(str(dates[0]))
    raise ValueError("--start-date is required when desc.json has no source_dates")


def main() -> int:
    args = parse_args()
    in_dir = Path(args.input_dataset).expanduser().resolve()
    out_dir = Path(args.output_dataset).expanduser().resolve()
    desc = read_json(in_dir / "desc.json")
    shape = tuple(int(x) for x in desc["shape"])
    if len(shape) != 3 or shape[2] != 1:
        raise ValueError(f"expected one-channel [T,N,1] input, got shape={shape}")

    frequency_minutes = float(desc["frequency (minutes)"])
    steps_per_day = int(round(24 * 60 / frequency_minutes))
    start = infer_start_date(desc, args.start_date)

    source = np.memmap(in_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    out_shape = (shape[0], shape[1], 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = np.memmap(out_dir / "data.dat", dtype="float32", mode="w+", shape=out_shape)
    target[:, :, 0] = source[:, :, 0]

    tod = np.empty((shape[0],), dtype=np.float32)
    dow = np.empty((shape[0],), dtype=np.float32)
    for idx in range(shape[0]):
        stamp = start + timedelta(minutes=frequency_minutes * idx)
        tod[idx] = ((stamp.hour * 60) + stamp.minute) / (24 * 60)
        dow[idx] = stamp.weekday() / 7
    target[:, :, 1] = tod[:, None]
    target[:, :, 2] = dow[:, None]
    target.flush()
    del target

    for name in ("adj_mx.pkl", "meta.csv"):
        shutil.copy2(in_dir / name, out_dir / name)

    out_desc = dict(desc)
    out_desc["name"] = out_dir.name
    out_desc["shape"] = list(out_shape)
    out_desc["num_features"] = 3
    descriptions = list(out_desc.get("feature_description", ["target"]))
    out_desc["feature_description"] = descriptions[:1] + [
        "time of day normalized to [0,1)",
        "day of week normalized to [0,1)",
    ]
    out_desc["base_dataset"] = in_dir.name
    out_desc["time_feature_start_date"] = start.date().isoformat()
    out_desc["time_feature_steps_per_day"] = steps_per_day
    write_json(out_dir / "desc.json", out_desc)

    value = np.asarray(source[:, :, 0])
    stats = {
        "dataset": out_dir.name,
        "base_dataset": in_dir.name,
        "shape": list(out_shape),
        "frequency_minutes": frequency_minutes,
        "steps_per_day": steps_per_day,
        "start_date": start.date().isoformat(),
        "target_zero_rate": float((value == 0).mean()),
        "target_mean": float(value.mean()),
        "target_std": float(value.std()),
    }
    write_json(out_dir / "stats.json", stats)
    print(f"[done] wrote {out_dir}: shape={out_shape}, steps_per_day={steps_per_day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

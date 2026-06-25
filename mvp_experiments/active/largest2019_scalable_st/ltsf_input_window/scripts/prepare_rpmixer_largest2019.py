#!/usr/bin/env python3
"""Prepare FaST/RPMixer LargeST-2019 inputs from BasicTS datasets.

The official RPMixer runner expects:

- ``process_data/{lower}_his_2019_agg.npz`` with data/tod/dow/lat/lng/direction.
- ``data/96_{horizon}/idx_{train,val,test}.npy`` under the RPMixer cwd.

This script creates only those adapter artifacts and leaves BasicTS datasets
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASET_TO_RPMIXER = {
    "SD": "sd_his_2019_agg",
    "GBA": "gba_his_2019_agg",
    "GLA": "gla_his_2019_agg",
}

DIRECTION_MAP = {
    "N": 0,
    "E": 1,
    "S": 2,
    "W": 3,
}


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_meta(meta_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat: list[float] = []
    lng: list[float] = []
    direction: list[int] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat.append(float(row["Lat"]))
            lng.append(float(row["Lng"]))
            raw_direction = (row.get("Direction") or "N").strip().upper()
            direction.append(DIRECTION_MAP.get(raw_direction[:1], 0))
    return (
        np.asarray(lat, dtype=np.float32),
        np.asarray(lng, dtype=np.float32),
        np.asarray(direction, dtype=np.int64),
    )


def load_basicts_array(dataset_dir: Path) -> np.ndarray:
    with open(dataset_dir / "desc.json", "r", encoding="utf-8") as f:
        desc = json.load(f)
    shape = tuple(desc["shape"])
    his_path = dataset_dir / "his.npz"
    if his_path.exists():
        return np.load(his_path)["data"].astype(np.float32, copy=False)
    return np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)


def prepare_npz(dataset: str, basicts_root: Path, rpmixer_root: Path, force: bool) -> Path:
    dataset_dir = basicts_root / dataset
    target_name = DATASET_TO_RPMIXER[dataset]
    process_dir = rpmixer_root / "process_data"
    process_dir.mkdir(parents=True, exist_ok=True)
    out_path = process_dir / f"{target_name}.npz"
    if out_path.exists() and not force:
        return out_path

    raw = load_basicts_array(dataset_dir)
    lat, lng, direction = read_meta(dataset_dir / "meta.csv")
    np.savez(
        out_path,
        data=np.asarray(raw[:, :, 0], dtype=np.float32),
        tod=np.asarray(raw[:, 0, 1], dtype=np.float32),
        dow=np.asarray(raw[:, 0, 2], dtype=np.float32),
        lat=lat,
        lng=lng,
        direction=direction,
    )
    return out_path


def prepare_indices(rpmixer_root: Path, num_steps: int, input_len: int, horizon: int, force: bool) -> Path:
    index_dir = rpmixer_root / "data" / f"{input_len}_{horizon}"
    train_path = index_dir / "idx_train.npy"
    if train_path.exists() and not force:
        return index_dir

    total_steps = input_len + horizon
    indices = np.arange(total_steps - 1, num_steps)
    train_end = int(0.6 * len(indices))
    val_end = train_end + int(0.2 * len(indices))
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "idx_train.npy", indices[:train_end])
    np.save(index_dir / "idx_val.npy", indices[train_end:val_end])
    np.save(index_dir / "idx_test.npy", indices[val_end:])
    return index_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="SD,GBA,GLA")
    parser.add_argument("--horizons", default="48,96,192,672")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--basicts-root", default="BasicTS/datasets")
    parser.add_argument(
        "--rpmixer-root",
        default="mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/third_party/RPMixer",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    basicts_root = Path(args.basicts_root)
    rpmixer_root = Path(args.rpmixer_root)
    datasets = csv_list(args.datasets)
    horizons = [int(item) for item in csv_list(args.horizons)]

    num_steps_by_dataset: dict[str, int] = {}
    for dataset in datasets:
        if dataset not in DATASET_TO_RPMIXER:
            raise KeyError(f"Unsupported dataset for RPMixer: {dataset}")
        with open(basicts_root / dataset / "desc.json", "r", encoding="utf-8") as f:
            desc = json.load(f)
        num_steps_by_dataset[dataset] = int(desc["num_time_steps"])
        out_path = prepare_npz(dataset, basicts_root, rpmixer_root, args.force)
        print(f"prepared data {dataset}: {out_path}")

    num_steps_values = set(num_steps_by_dataset.values())
    if len(num_steps_values) != 1:
        raise ValueError(f"RPMixer index generation assumes aligned lengths, got {num_steps_by_dataset}")
    num_steps = num_steps_values.pop()
    for horizon in horizons:
        index_dir = prepare_indices(rpmixer_root, num_steps, args.input_len, horizon, args.force)
        print(f"prepared indices L{args.input_len} H{horizon}: {index_dir}")


if __name__ == "__main__":
    main()

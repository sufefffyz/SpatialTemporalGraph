#!/usr/bin/env python3
"""Prepare non-overwriting LargeST LTSF caches for the LargeST/BiST loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


def dataset_dir_name(dataset: str) -> str:
    return dataset.lower()


def load_years(data_root: Path, dataset: str, years: list[str]) -> pd.DataFrame:
    ds_dir = data_root / dataset_dir_name(dataset)
    frames = []
    for year in years:
        h5_path = ds_dir / f"{dataset_dir_name(dataset)}_his_{year}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing LargeST history file: {h5_path}")
        frames.append(pd.read_hdf(h5_path))
    return pd.concat(frames)


def generate_data_and_idx(
    df: pd.DataFrame,
    input_len: int,
    horizon: int,
    add_time_of_day: bool,
    add_day_of_week: bool,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values.astype(np.float32), axis=-1)

    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = (dow_tiled / 7).astype(np.float32)
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)
    x_offsets = np.arange(-(input_len - 1), 1, 1)
    y_offsets = np.arange(1, horizon + 1, 1)
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def write_cache(
    data_root: Path,
    dataset: str,
    output_tag: str,
    data: np.ndarray,
    idx: np.ndarray,
    input_len: int,
) -> dict:
    out_dir = data_root / dataset_dir_name(dataset) / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)

    idx_train = idx[:num_train]
    idx_val = idx[num_train : num_train + num_val]
    idx_test = idx[num_train + num_val :]

    x_train = data[: idx_val[0] - input_len, :, 0]
    scaler = StandardScaler(mean=float(x_train.mean()), std=float(x_train.std()))
    data = data.copy()
    data[..., 0] = scaler.transform(data[..., 0])

    np.savez_compressed(out_dir / "his.npz", data=data, mean=scaler.mean, std=scaler.std)
    np.save(out_dir / "idx_train", idx_train)
    np.save(out_dir / "idx_val", idx_val)
    np.save(out_dir / "idx_test", idx_test)

    return {
        "out_dir": str(out_dir),
        "data_shape": list(data.shape),
        "idx_train": int(len(idx_train)),
        "idx_val": int(len(idx_val)),
        "idx_test": int(len(idx_test)),
        "mean": scaler.mean,
        "std": scaler.std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--largest-data-root", default="LargeST/data")
    parser.add_argument("--dataset", required=True, choices=["SD", "GBA", "GLA", "CA"])
    parser.add_argument("--years", default="2019")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=672)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--tod", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.largest_data_root)
    years = [x for x in args.years.split("_") if x]
    output_tag = args.output_tag or f"{args.years}_L{args.input_len}_H{args.horizon}"
    out_dir = data_root / dataset_dir_name(args.dataset) / output_tag
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"{out_dir} already exists. Pass --force to overwrite its files.")

    df = load_years(data_root, args.dataset, years)
    data, idx = generate_data_and_idx(df, args.input_len, args.horizon, args.tod, args.dow)
    summary = write_cache(data_root, args.dataset, output_tag, data, idx, args.input_len)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CA_H5 = REPO_ROOT / "LargeST" / "data" / "ca" / "ca_his_raw_2019.h5"
DEFAULT_SD_META = REPO_ROOT / "LargeST" / "data" / "sd" / "sd_meta.csv"
DEFAULT_SD_ADJ = REPO_ROOT / "LargeST" / "data" / "sd" / "sd_rn_adj.npy"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "SD_5min_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 5-minute SD raw bundle from LargeST CA raw data."
    )
    parser.add_argument("--ca-h5", type=Path, default=DEFAULT_CA_H5, help="LargeST CA raw 2019 HDF path")
    parser.add_argument("--sd-meta", type=Path, default=DEFAULT_SD_META, help="LargeST SD metadata CSV path")
    parser.add_argument("--sd-adj", type=Path, default=DEFAULT_SD_ADJ, help="LargeST SD road-network adjacency .npy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--output-h5-name",
        default="sd_his_5min_2019.h5",
        help="Output HDF filename inside output-dir",
    )
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def normalize_sensor_id(value) -> str:
    return str(int(value))


def main() -> None:
    args = parse_args()
    ca_h5 = ensure_exists(args.ca_h5, "CA raw HDF")
    sd_meta = ensure_exists(args.sd_meta, "SD metadata CSV")
    sd_adj = ensure_exists(args.sd_adj, "SD adjacency NPY")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(sd_meta)
    required_cols = {"ID"}
    missing_cols = required_cols - set(meta_df.columns)
    if missing_cols:
        raise ValueError(f"{sd_meta} missing required columns: {sorted(missing_cols)}")

    meta_df["ID"] = pd.to_numeric(meta_df["ID"], errors="raise").astype(np.int64)
    sensor_ids = meta_df["ID"].to_numpy(dtype=np.int64)
    if len(sensor_ids) != len(np.unique(sensor_ids)):
        raise ValueError("SD metadata contains duplicate sensor IDs.")

    adj = np.load(sd_adj)
    if adj.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError(
            f"Adjacency shape {adj.shape} does not match sensor count {len(sensor_ids)} from metadata."
        )

    ca_df = pd.read_hdf(ca_h5)
    column_lookup = {str(column): column for column in ca_df.columns}
    ordered_columns = []
    missing_sensor_ids = []
    for sensor_id in sensor_ids.tolist():
        sensor_key = normalize_sensor_id(sensor_id)
        original_column = column_lookup.get(sensor_key)
        if original_column is None:
            missing_sensor_ids.append(sensor_id)
        else:
            ordered_columns.append(original_column)

    if missing_sensor_ids:
        raise ValueError(
            f"CA raw HDF is missing {len(missing_sensor_ids)} SD sensors, sample={missing_sensor_ids[:10]}"
        )

    sd_df = ca_df.loc[:, ordered_columns].copy()
    sd_df.columns = [normalize_sensor_id(sensor_id) for sensor_id in sensor_ids.tolist()]

    output_h5 = output_dir / args.output_h5_name
    sd_df.to_hdf(output_h5, key="t", mode="w")
    meta_df.to_csv(output_dir / "sd_meta.csv", index=False)
    np.save(output_dir / "sd_rn_adj.npy", adj.astype(np.float32))
    np.save(output_dir / "sensor_ids.npy", sensor_ids.astype(np.int64))
    (output_dir / "sensor_ids.txt").write_text(
        "\n".join(str(int(sensor_id)) for sensor_id in sensor_ids.tolist()) + "\n",
        encoding="utf-8",
    )

    summary = {
        "ca_h5": str(ca_h5),
        "sd_meta": str(sd_meta),
        "sd_adj": str(sd_adj),
        "output_h5": str(output_h5),
        "num_time_steps": int(sd_df.shape[0]),
        "num_nodes": int(sd_df.shape[1]),
        "adj_shape": list(adj.shape),
        "start_timestamp": str(sd_df.index.min()),
        "end_timestamp": str(sd_df.index.max()),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

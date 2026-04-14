#!/usr/bin/env python3

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "datasets" / "SD_5min_raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "SD_5min_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the SD 5-minute raw bundle into a BasicTS-style memmap dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input raw bundle directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output dataset directory")
    parser.add_argument(
        "--input-h5-name",
        default="sd_his_5min_2019.h5",
        help="Input HDF filename inside input-dir",
    )
    parser.add_argument("--dataset-name", default="SD_5min_full", help="Dataset name written into desc.json")
    parser.add_argument("--input-len", type=int, default=12, help="Sequence input length")
    parser.add_argument("--output-len", type=int, default=12, help="Sequence output length")
    parser.add_argument("--steps-per-day", type=int, default=288, help="5-minute steps per day")
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_sensor_ids(input_dir: Path, meta_path: Path, expected_num_nodes: int) -> np.ndarray:
    sensor_ids_path = input_dir / "sensor_ids.npy"
    if sensor_ids_path.exists():
        sensor_ids = np.load(sensor_ids_path, allow_pickle=True).astype(np.int64)
        if len(sensor_ids) != expected_num_nodes:
            raise ValueError(
                f"sensor_ids.npy length {len(sensor_ids)} does not match HDF columns {expected_num_nodes}."
            )
        return sensor_ids

    meta_df = pd.read_csv(meta_path)
    if "ID" not in meta_df.columns:
        raise ValueError(f"{meta_path} is missing required column 'ID'.")
    sensor_ids = pd.to_numeric(meta_df["ID"], errors="raise").astype(np.int64).to_numpy()
    if len(sensor_ids) != expected_num_nodes:
        raise ValueError(
            f"Recovered {len(sensor_ids)} sensor IDs from {meta_path}, expected {expected_num_nodes}."
        )
    return sensor_ids


def build_features(flow_df: pd.DataFrame, steps_per_day: int) -> np.ndarray:
    flow = np.expand_dims(flow_df.to_numpy(dtype=np.float32), axis=-1)
    time_of_day = (
        (flow_df.index.values - flow_df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    ).astype(np.float32)
    time_of_day = np.tile(time_of_day[:, None, None], (1, flow.shape[1], 1))

    day_of_week = (flow_df.index.dayofweek.to_numpy(dtype=np.float32) / 7.0).astype(np.float32)
    day_of_week = np.tile(day_of_week[:, None, None], (1, flow.shape[1], 1))

    data = np.concatenate([flow, time_of_day, day_of_week], axis=-1).astype(np.float32)
    return data


def dump_memmap(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp


def main() -> None:
    args = parse_args()
    input_dir = ensure_exists(args.input_dir, "Input raw bundle directory")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_h5 = ensure_exists(input_dir / args.input_h5_name, "SD raw HDF")
    meta_path = ensure_exists(input_dir / "sd_meta.csv", "sd_meta.csv")
    adj_path = ensure_exists(input_dir / "sd_rn_adj.npy", "sd_rn_adj.npy")

    flow_df = pd.read_hdf(input_h5)
    sensor_ids = load_sensor_ids(input_dir, meta_path, flow_df.shape[1])
    if flow_df.shape[1] != len(sensor_ids):
        raise ValueError(
            f"HDF columns {flow_df.shape[1]} do not match sensor_ids length {len(sensor_ids)}."
        )

    data = build_features(flow_df, args.steps_per_day)
    dump_memmap(output_dir / "data.dat", data)
    np.save(output_dir / "shape.npy", np.asarray(data.shape, dtype=np.int64))
    np.save(output_dir / "sensor_ids.npy", sensor_ids.astype(np.int64))
    (output_dir / "sensor_ids.txt").write_text(
        "\n".join(str(int(sensor_id)) for sensor_id in sensor_ids.tolist()) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(meta_path, output_dir / "meta.csv")

    adj = np.load(adj_path).astype(np.float32)
    with (output_dir / "adj_mx_largeST_original.pkl").open("wb") as fp:
        pickle.dump(adj, fp, protocol=4)

    desc = {
        "name": args.dataset_name,
        "domain": "traffic flow",
        "shape": list(data.shape),
        "num_time_steps": int(data.shape[0]),
        "num_nodes": int(data.shape[1]),
        "num_features": int(data.shape[2]),
        "feature_description": ["flow", "time of day", "day of week"],
        "has_graph": True,
        "frequency (minutes)": 5,
        "regular_settings": {
            "INPUT_LEN": args.input_len,
            "OUTPUT_LEN": args.output_len,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0,
        },
    }
    (output_dir / "desc.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")

    summary = {
        "input_h5": str(input_h5),
        "output_dir": str(output_dir),
        "shape": list(data.shape),
        "sensor_ids_path": str(output_dir / "sensor_ids.npy"),
        "meta_path": str(output_dir / "meta.csv"),
        "adj_path": str(output_dir / "adj_mx_largeST_original.pkl"),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_5min_full"
DEFAULT_EXISTING_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_EXISTING_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "graphs" / "SD"


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
    parser.add_argument("--days-1m", type=int, default=31, help="Train days kept for the 1m split")
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=DEFAULT_GRAPH_ROOT,
        help="Optional graph export directory for benchmark graph variants.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional explicit metadata CSV path. Defaults to input-dir/sd_meta.csv or BasicTS/datasets/SD/meta.csv",
    )
    parser.add_argument(
        "--adj-path",
        type=Path,
        default=None,
        help="Optional explicit adjacency path (.npy or .pkl). Defaults to input-dir/sd_rn_adj.npy or BasicTS/datasets/SD/adj_mx.pkl",
    )
    parser.add_argument(
        "--physical-adj-path",
        type=Path,
        default=None,
        help="Optional physical graph adjacency path. Defaults to BasicTS/datasets/SD_phys/adj_mx.pkl if present.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def resolve_optional_input_path(primary: Path | None, candidates: list[Path], desc: str) -> Path:
    if primary is not None:
        return ensure_exists(primary, desc)
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{desc} not found in any of: " + ", ".join(str(candidate) for candidate in candidates)
    )


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


def load_adjacency_matrix(adj_path: Path) -> np.ndarray:
    if adj_path.suffix.lower() == ".npy":
        return np.load(adj_path).astype(np.float32)

    with adj_path.open("rb") as fp:
        payload = pickle.load(fp)
    if isinstance(payload, tuple) and len(payload) == 3:
        payload = payload[2]
    return np.asarray(payload, dtype=np.float32)


def validate_physical_alignment(
    physical_adj: np.ndarray,
    physical_meta_path: Path | None,
    sensor_ids: np.ndarray,
) -> None:
    if physical_adj.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError(
            f"Physical adjacency shape {physical_adj.shape} does not match sensor count {len(sensor_ids)}."
        )

    if physical_meta_path is None or not physical_meta_path.exists():
        return

    physical_meta_df = pd.read_csv(physical_meta_path)
    if "ID" not in physical_meta_df.columns:
        return

    physical_sensor_ids = pd.to_numeric(physical_meta_df["ID"], errors="raise").astype(np.int64).to_numpy()
    if len(physical_sensor_ids) != len(sensor_ids):
        raise ValueError(
            f"Physical metadata contains {len(physical_sensor_ids)} IDs, expected {len(sensor_ids)}."
        )
    if not np.array_equal(physical_sensor_ids, sensor_ids):
        raise ValueError("SD_phys metadata sensor order does not match the 5-minute SD sensor order.")


def export_graph_bundle(
    graph_root: Path,
    sensor_ids: np.ndarray,
    meta_path: Path,
    large_st_adj: np.ndarray,
    physical_adj: np.ndarray | None,
) -> dict[str, str]:
    graph_root.mkdir(parents=True, exist_ok=True)

    with (graph_root / "adj_mx_largeST_original.pkl").open("wb") as fp:
        pickle.dump(large_st_adj.astype(np.float32), fp, protocol=4)

    np.save(graph_root / "sensor_ids.npy", sensor_ids.astype(np.int64))
    (graph_root / "sensor_ids.txt").write_text(
        "\n".join(str(int(sensor_id)) for sensor_id in sensor_ids.tolist()) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(meta_path, graph_root / "meta.csv")

    exports = {
        "graph_root": str(graph_root),
        "largeST_original": str(graph_root / "adj_mx_largeST_original.pkl"),
        "sensor_ids": str(graph_root / "sensor_ids.npy"),
        "meta": str(graph_root / "meta.csv"),
    }

    if physical_adj is not None:
        with (graph_root / "adj_mx_physical_directed.pkl").open("wb") as fp:
            pickle.dump(physical_adj.astype(np.float32), fp, protocol=4)
        exports["physical_directed"] = str(graph_root / "adj_mx_physical_directed.pkl")

    return exports


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


def build_split_indices(total_len: int, days_1m: int = 31, steps_per_day: int = 288) -> dict[str, dict[str, np.ndarray]]:
    valid_len = int(total_len * 0.2)
    test_len = int(total_len * 0.2)
    full_train_len = total_len - valid_len - test_len
    month_train_len = min(days_1m * steps_per_day, full_train_len)

    full = {
        "train_idx": np.arange(0, full_train_len, dtype=np.int64),
        "val_idx": np.arange(full_train_len, full_train_len + valid_len, dtype=np.int64),
        "test_idx": np.arange(full_train_len + valid_len, total_len, dtype=np.int64),
    }
    onemonth = {
        "train_idx": np.arange(full_train_len - month_train_len, full_train_len, dtype=np.int64),
        "val_idx": full["val_idx"].copy(),
        "test_idx": full["test_idx"].copy(),
    }
    return {"full": full, "1m": onemonth}


def main() -> None:
    args = parse_args()
    input_dir = ensure_exists(args.input_dir, "Input raw bundle directory")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_root = args.graph_root.expanduser().resolve()

    input_h5 = ensure_exists(input_dir / args.input_h5_name, "SD raw HDF")
    meta_path = resolve_optional_input_path(
        args.meta_path,
        [input_dir / "sd_meta.csv", DEFAULT_EXISTING_SD_DIR / "meta.csv"],
        "SD metadata CSV",
    )
    adj_path = resolve_optional_input_path(
        args.adj_path,
        [input_dir / "sd_rn_adj.npy", DEFAULT_EXISTING_SD_DIR / "adj_mx.pkl"],
        "SD adjacency",
    )
    physical_adj_path = None
    if args.physical_adj_path is not None:
        physical_adj_path = ensure_exists(args.physical_adj_path, "Physical adjacency")
    elif (DEFAULT_EXISTING_SD_PHYS_DIR / "adj_mx.pkl").exists():
        physical_adj_path = (DEFAULT_EXISTING_SD_PHYS_DIR / "adj_mx.pkl").resolve()

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

    adj = load_adjacency_matrix(adj_path)
    if adj.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError(
            f"Adjacency shape {adj.shape} does not match recovered sensor count {len(sensor_ids)}."
        )
    with (output_dir / "adj_mx_largeST_original.pkl").open("wb") as fp:
        pickle.dump(adj, fp, protocol=4)

    physical_adj = None
    if physical_adj_path is not None:
        physical_adj = load_adjacency_matrix(physical_adj_path)
        validate_physical_alignment(
            physical_adj=physical_adj,
            physical_meta_path=(physical_adj_path.parent / "meta.csv"),
            sensor_ids=sensor_ids,
        )
        with (output_dir / "adj_mx_physical_directed.pkl").open("wb") as fp:
            pickle.dump(physical_adj, fp, protocol=4)

    split_variants = build_split_indices(
        total_len=data.shape[0],
        days_1m=args.days_1m,
        steps_per_day=args.steps_per_day,
    )
    for split_name, split_payload in split_variants.items():
        np.savez_compressed(output_dir / f"split_indices_{split_name}.npz", **split_payload)

    graph_exports = export_graph_bundle(
        graph_root=graph_root,
        sensor_ids=sensor_ids,
        meta_path=meta_path,
        large_st_adj=adj,
        physical_adj=physical_adj,
    )

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
        "physical_adj_path": str(output_dir / "adj_mx_physical_directed.pkl") if physical_adj is not None else None,
        "split_files": {
            split_name: str(output_dir / f"split_indices_{split_name}.npz")
            for split_name in split_variants
        },
        "graph_exports": graph_exports,
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

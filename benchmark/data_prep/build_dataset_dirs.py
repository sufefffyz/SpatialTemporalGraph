#!/usr/bin/env python3

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DATASET_DIR = REPO_ROOT / "datasets" / "SD_5min_full"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "graphs" / "SD"
DEFAULT_BASICTS_DATASETS_DIR = REPO_ROOT / "BasicTS" / "datasets"
DEFAULT_WINDOWS = ("full", "1m")
DEFAULT_GRAPHS = (
    "largeST_original",
    "physical_forward",
    "physical_reverse",
    "physical_bidir",
    "physical_ML_only",
    "adaptive_only",
    "adaptive_plus_phys",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BasicTS dataset directories for SD 5-minute graph benchmarks."
    )
    parser.add_argument("--source-dataset-dir", type=Path, default=DEFAULT_SOURCE_DATASET_DIR)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--basicts-datasets-dir", type=Path, default=DEFAULT_BASICTS_DATASETS_DIR)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    parser.add_argument("--days-1m", type=int, default=31, help="Days kept for the 1m train window")
    parser.add_argument(
        "--copy-meta",
        action="store_true",
        help="Copy meta/sensor id files instead of creating symlinks",
    )
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_shape(dataset_dir: Path) -> tuple[int, int, int]:
    shape_path = dataset_dir / "shape.npy"
    if shape_path.exists():
        shape = np.load(shape_path).astype(np.int64).tolist()
        return tuple(int(x) for x in shape)
    desc = json.loads((dataset_dir / "desc.json").read_text(encoding="utf-8"))
    return tuple(int(x) for x in desc["shape"])


def dump_memmap(path: Path, array: np.ndarray) -> None:
    fp = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp


def write_desc(
    target_dir: Path,
    dataset_name: str,
    data_shape: tuple[int, int, int],
    input_len: int,
    output_len: int,
    train_len: int,
    valid_len: int,
    test_len: int,
    frequency_minutes: int,
) -> None:
    total_len = train_len + valid_len + test_len
    desc = {
        "name": dataset_name,
        "domain": "traffic flow",
        "shape": list(data_shape),
        "num_time_steps": int(data_shape[0]),
        "num_nodes": int(data_shape[1]),
        "num_features": int(data_shape[2]),
        "feature_description": ["flow", "time of day", "day of week"],
        "has_graph": True,
        "frequency (minutes)": frequency_minutes,
        "regular_settings": {
            "INPUT_LEN": input_len,
            "OUTPUT_LEN": output_len,
            "TRAIN_VAL_TEST_RATIO": [
                train_len / total_len,
                valid_len / total_len,
                test_len / total_len,
            ],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0,
        },
    }
    (target_dir / "desc.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")


def reuse_small_file(source: Path, target: Path, copy_mode: bool) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy_mode:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)


def graph_file_name(graph_name: str) -> str:
    mapping = {
        "largeST_original": "adj_mx_largeST_original.pkl",
        "physical_forward": "adj_mx_physical_forward.pkl",
        "physical_reverse": "adj_mx_physical_reverse.pkl",
        "physical_bidir": "adj_mx_physical_bidir.pkl",
        "physical_ML_only": "adj_mx_physical_ML_only.pkl",
        "adaptive_only": "adj_mx_largeST_original.pkl",
        "adaptive_plus_phys": "adj_mx_physical_forward.pkl",
    }
    if graph_name not in mapping:
        raise KeyError(f"Unknown graph name: {graph_name}")
    return mapping[graph_name]


def main() -> None:
    args = parse_args()
    source_dataset_dir = ensure_exists(args.source_dataset_dir, "Source 5-minute dataset directory")
    graph_root = ensure_exists(args.graph_root, "Graph root directory")
    basicts_datasets_dir = args.basicts_datasets_dir.expanduser().resolve()
    basicts_datasets_dir.mkdir(parents=True, exist_ok=True)

    desc = json.loads((source_dataset_dir / "desc.json").read_text(encoding="utf-8"))
    source_shape = load_shape(source_dataset_dir)
    data = np.memmap(
        ensure_exists(source_dataset_dir / "data.dat", "source data.dat"),
        dtype="float32",
        mode="r",
        shape=source_shape,
    )

    total_len = int(source_shape[0])
    num_nodes = int(source_shape[1])
    num_features = int(source_shape[2])
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    frequency_minutes = int(desc["frequency (minutes)"])

    valid_len = int(total_len * desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"][1])
    test_len = int(total_len * desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"][2])
    full_train_len = total_len - valid_len - test_len
    onemonth_train_len = min(args.days_1m * 288, full_train_len)

    window_to_train_len = {
        "full": full_train_len,
        "1m": onemonth_train_len,
    }

    summary_rows = []
    for window in args.windows:
        if window not in window_to_train_len:
            raise ValueError(f"Unsupported window: {window}")
        train_len = int(window_to_train_len[window])
        kept_len = train_len + valid_len + test_len
        start_idx = total_len - kept_len
        subset = np.asarray(data[start_idx:], dtype=np.float32)
        data_shape = subset.shape
        if data_shape[1] != num_nodes or data_shape[2] != num_features:
            raise ValueError(f"Unexpected subset shape: {data_shape}")

        for graph_name in args.graphs:
            dataset_name = f"SD_LargeST_{window}_{graph_name}"
            target_dir = basicts_datasets_dir / dataset_name
            target_dir.mkdir(parents=True, exist_ok=True)

            dump_memmap(target_dir / "data.dat", subset)
            np.save(target_dir / "shape.npy", np.asarray(data_shape, dtype=np.int64))
            write_desc(
                target_dir=target_dir,
                dataset_name=dataset_name,
                data_shape=data_shape,
                input_len=input_len,
                output_len=output_len,
                train_len=train_len,
                valid_len=valid_len,
                test_len=test_len,
                frequency_minutes=frequency_minutes,
            )

            graph_source = ensure_exists(graph_root / graph_file_name(graph_name), f"{graph_name} adjacency")
            shutil.copy2(graph_source, target_dir / "adj_mx.pkl")

            for filename in ("meta.csv", "sensor_ids.npy", "sensor_ids.txt"):
                source_file = source_dataset_dir / filename
                if source_file.exists():
                    reuse_small_file(source_file, target_dir / filename, args.copy_meta)

            build_summary = {
                "source_dataset_dir": str(source_dataset_dir),
                "graph_root": str(graph_root),
                "window": window,
                "graph": graph_name,
                "dataset_name": dataset_name,
                "shape": list(data_shape),
                "train_len": train_len,
                "valid_len": valid_len,
                "test_len": test_len,
                "start_index": start_idx,
                "graph_source": str(graph_source),
            }
            (target_dir / "build_summary.json").write_text(
                json.dumps(build_summary, indent=2),
                encoding="utf-8",
            )
            summary_rows.append(build_summary)

    summary_path = basicts_datasets_dir / "SD_LargeST_dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(json.dumps({"num_datasets": len(summary_rows), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

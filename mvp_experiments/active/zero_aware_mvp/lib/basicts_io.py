from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"


def dataset_dir(dataset_name: str) -> Path:
    override_root = os.environ.get("BASICTS_DATASETS_ROOT")
    if override_root:
        return Path(override_root).expanduser() / dataset_name
    return BASICTS_ROOT / "datasets" / dataset_name


def load_desc(dataset_name: str) -> dict:
    path = dataset_dir(dataset_name) / "desc.json"
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_data(dataset_name: str) -> np.memmap:
    desc = load_desc(dataset_name)
    return np.memmap(
        dataset_dir(dataset_name) / "data.dat",
        dtype="float32",
        mode="r",
        shape=tuple(desc["shape"]),
    )


def load_split(dataset_name: str, split_filename: str = "split_indices.npz") -> dict[str, np.ndarray]:
    path = dataset_dir(dataset_name) / split_filename
    if not path.exists():
        desc = load_desc(dataset_name)
        total_len = int(desc["num_time_steps"])
        ratio = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
        valid_len = int(total_len * float(ratio[1]))
        test_len = int(total_len * float(ratio[2]))
        train_len = total_len - valid_len - test_len
        return {
            "train": np.arange(0, train_len, dtype=np.int64),
            "valid": np.arange(train_len, train_len + valid_len, dtype=np.int64),
            "test": np.arange(train_len + valid_len, total_len, dtype=np.int64),
        }
    with np.load(path) as data:
        return {
            "train": np.asarray(data["train_idx"], dtype=np.int64),
            "valid": np.asarray(data["val_idx"], dtype=np.int64),
            "test": np.asarray(data["test_idx"], dtype=np.int64),
        }


def load_unix_timestamps(dataset_name: str) -> np.ndarray | None:
    path = dataset_dir(dataset_name) / "unix_timestamps.npy"
    if not path.exists():
        return None
    return np.load(path).astype(np.int64)


def prediction_starts(
    total_len: int,
    split_idx: np.ndarray,
    input_len: int,
    output_len: int,
    mode: str,
    allow_val_test_context: bool = True,
) -> np.ndarray:
    split_idx = np.asarray(split_idx, dtype=np.int64)
    split_mask = np.zeros(total_len, dtype=np.int64)
    split_mask[split_idx] = 1
    prefix = np.concatenate([[0], np.cumsum(split_mask)])

    def inside_split(start: int, length: int) -> bool:
        end = start + length
        return (prefix[end] - prefix[start]) == length

    starts: list[int] = []
    for target_start in split_idx:
        target_start = int(target_start)
        if target_start < input_len:
            continue
        if target_start + output_len > total_len:
            continue
        if not inside_split(target_start, output_len):
            continue
        history_start = target_start - input_len
        if mode == "train" or not allow_val_test_context:
            if not inside_split(history_start, input_len):
                continue
        starts.append(target_start)
    return np.asarray(starts, dtype=np.int64)


def test_prediction_starts(dataset_name: str) -> np.ndarray:
    desc = load_desc(dataset_name)
    split = load_split(dataset_name)
    settings = desc["regular_settings"]
    return prediction_starts(
        total_len=int(desc["num_time_steps"]),
        split_idx=split["test"],
        input_len=int(settings["INPUT_LEN"]),
        output_len=int(settings["OUTPUT_LEN"]),
        mode="test",
    )


def train_targets(dataset_name: str, target_channel: int = 0) -> np.ndarray:
    data = load_data(dataset_name)
    split = load_split(dataset_name)
    return np.asarray(data[split["train"], :, target_channel], dtype=np.float32).copy()

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explicit split files for the old SD / SD_phys datasets."
    )
    parser.add_argument("--sd-dir", type=Path, default=DEFAULT_SD_DIR)
    parser.add_argument("--sd-phys-dir", type=Path, default=DEFAULT_SD_PHYS_DIR)
    parser.add_argument("--days-1m", type=int, default=31)
    parser.add_argument("--steps-per-day", type=int, default=96, help="Old SD uses 15-minute data by default")
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_num_steps(dataset_dir: Path) -> int:
    with (dataset_dir / "desc.json").open("r", encoding="utf-8") as fp:
        desc = json.load(fp)
    return int(desc["shape"][0])


def build_split_indices(total_len: int, days_1m: int, steps_per_day: int) -> dict[str, dict[str, np.ndarray]]:
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


def write_split_files(dataset_dir: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    for split_name, payload in splits.items():
        np.savez_compressed(dataset_dir / f"split_indices_{split_name}.npz", **payload)


def main() -> None:
    args = parse_args()
    sd_dir = ensure_exists(args.sd_dir, "SD dataset dir")
    total_len = load_num_steps(sd_dir)
    splits = build_split_indices(total_len, args.days_1m, args.steps_per_day)
    write_split_files(sd_dir, splits)

    output = {
        "sd_dir": str(sd_dir),
        "num_steps": total_len,
        "generated": [str(sd_dir / f"split_indices_{name}.npz") for name in splits],
    }

    sd_phys_dir = args.sd_phys_dir.expanduser().resolve()
    if sd_phys_dir.exists():
        phys_len = load_num_steps(sd_phys_dir)
        if phys_len != total_len:
            raise ValueError(
                f"SD_phys num_steps {phys_len} does not match SD num_steps {total_len}."
            )
        write_split_files(sd_phys_dir, splits)
        output["sd_phys_dir"] = str(sd_phys_dir)
        output["generated"] += [str(sd_phys_dir / f"split_indices_{name}.npz") for name in splits]

    print(output)


if __name__ == "__main__":
    main()

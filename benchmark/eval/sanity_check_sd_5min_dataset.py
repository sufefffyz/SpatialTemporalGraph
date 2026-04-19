#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from stgraph_ext.dataset import ExplicitSplitTimeSeriesForecastingDataset
from stgraph_ext.scaler import ExplicitSplitZScoreScaler


DEFAULT_DATASET_DIR = BASICTS_ROOT / "datasets" / "SD_5min_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check the SD_5min BasicTS dataset and explain why evaluation metrics may be zero."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the BasicTS dataset directory, e.g. BasicTS/datasets/SD_5min_full",
    )
    parser.add_argument(
        "--sample-windows",
        type=int,
        default=256,
        help="Maximum number of windows sampled per split/mode for quick diagnostics.",
    )
    return parser.parse_args()


def load_desc(dataset_dir: Path) -> dict:
    with (dataset_dir / "desc.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_data(dataset_dir: Path, shape: tuple[int, ...]) -> np.memmap:
    return np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)


def load_split(dataset_dir: Path, split_filename: str) -> dict[str, np.ndarray]:
    with np.load(dataset_dir / split_filename) as split_data:
        return {
            "train": np.asarray(split_data["train_idx"], dtype=np.int64),
            "valid": np.asarray(split_data["val_idx"], dtype=np.int64),
            "test": np.asarray(split_data["test_idx"], dtype=np.int64),
        }


def flow_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "zero_ratio": float(np.mean(np.isclose(values, 0.0, atol=5e-5))),
        "nan_ratio": float(np.mean(np.isnan(values))),
    }


def split_stats(indices: np.ndarray) -> dict[str, int | bool | None]:
    if len(indices) == 0:
        return {"count": 0, "start": None, "end": None, "contiguous": False}
    contiguous = bool(np.all(np.diff(indices) == 1)) if len(indices) > 1 else True
    return {
        "count": int(len(indices)),
        "start": int(indices[0]),
        "end": int(indices[-1]),
        "contiguous": contiguous,
    }


def select_prediction_starts(dataset: ExplicitSplitTimeSeriesForecastingDataset, limit: int) -> np.ndarray:
    starts = dataset.prediction_starts
    return starts[: min(limit, len(starts))]


def summarize_window_targets(
    dataset: ExplicitSplitTimeSeriesForecastingDataset,
    null_val: float,
    sample_windows: int,
) -> dict[str, float | int | None]:
    starts = select_prediction_starts(dataset, sample_windows)
    if len(starts) == 0:
        return {
            "window_count": 0,
            "target_mean": None,
            "target_std": None,
            "target_min": None,
            "target_max": None,
            "target_zero_ratio": None,
            "mae_valid_ratio": None,
            "mape_valid_ratio": None,
        }

    targets = []
    for idx in range(len(starts)):
        sample = dataset[idx]
        targets.append(sample["target"][..., 0])
    target = np.stack(targets, axis=0).astype(np.float32)

    if math.isnan(null_val):
        mae_mask = ~np.isnan(target)
    else:
        mae_mask = ~np.isclose(target, null_val, atol=5e-5)
    mape_mask = mae_mask & ~np.isclose(target, 0.0, atol=5e-5)

    return {
        "window_count": int(len(starts)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_zero_ratio": float(np.mean(np.isclose(target, 0.0, atol=5e-5))),
        "mae_valid_ratio": float(np.mean(mae_mask)),
        "mape_valid_ratio": float(np.mean(mape_mask)),
    }


def persistence_metrics(
    dataset: ExplicitSplitTimeSeriesForecastingDataset,
    null_val: float,
    sample_windows: int,
) -> dict[str, float | int | None]:
    starts = select_prediction_starts(dataset, sample_windows)
    if len(starts) == 0:
        return {"window_count": 0, "MAE": None, "RMSE": None, "MAPE": None}

    predictions = []
    targets = []
    for idx in range(len(starts)):
        sample = dataset[idx]
        history = sample["inputs"][..., 0]
        target = sample["target"][..., 0]
        last_value = history[-1:]
        pred = np.repeat(last_value, repeats=target.shape[0], axis=0)
        predictions.append(pred)
        targets.append(target)

    prediction = torch.tensor(np.stack(predictions, axis=0)[..., None], dtype=torch.float32)
    target = torch.tensor(np.stack(targets, axis=0)[..., None], dtype=torch.float32)

    return {
        "window_count": int(len(starts)),
        "MAE": float(masked_mae(prediction, target, null_val=null_val).item()),
        "RMSE": float(masked_rmse(prediction, target, null_val=null_val).item()),
        "MAPE": float(masked_mape(prediction, target, null_val=null_val).item()),
    }


def summarize_scaler(
    dataset_name: str,
    split_filename: str,
    sample_target: np.ndarray,
) -> dict[str, float | list[float]]:
    scaler = ExplicitSplitZScoreScaler(
        dataset_name=dataset_name,
        train_ratio=0.6,
        norm_each_channel=False,
        rescale=True,
        split_filename=split_filename,
    )
    mean = scaler.mean.detach().cpu().numpy()
    std = scaler.std.detach().cpu().numpy()
    sample_tensor = torch.tensor(sample_target[..., None], dtype=torch.float32)
    transformed = scaler.transform(sample_tensor)
    restored = scaler.inverse_transform(transformed)
    return {
        "mean_shape": list(mean.shape) if hasattr(mean, "shape") else [],
        "std_shape": list(std.shape) if hasattr(std, "shape") else [],
        "mean_min": float(np.min(mean)),
        "mean_max": float(np.max(mean)),
        "std_min": float(np.min(std)),
        "std_max": float(np.max(std)),
        "inverse_max_abs_error": float(torch.max(torch.abs(restored - sample_tensor)).item()),
    }


def build_dataset(dataset_name: str, split_filename: str, mode: str, input_len: int, output_len: int):
    return ExplicitSplitTimeSeriesForecastingDataset(
        dataset_name=dataset_name,
        train_val_test_ratio=[0.6, 0.2, 0.2],
        mode=mode,
        input_len=input_len,
        output_len=output_len,
        split_filename=split_filename,
    )


def detect_issues(report: dict) -> list[str]:
    findings: list[str] = []
    for split_name, split_report in report["splits"].items():
        for mode in ("train", "valid", "test"):
            mode_report = split_report["datasets"][mode]
            if mode_report["num_windows"] == 0:
                findings.append(f"{split_name}/{mode}: no valid windows were constructed.")
            elif mode_report["sample_targets"]["mae_valid_ratio"] == 0.0:
                findings.append(f"{split_name}/{mode}: every sampled target was masked out by MAE (likely NULL_VAL issue).")
            elif mode_report["sample_targets"]["target_zero_ratio"] is not None and mode_report["sample_targets"]["target_zero_ratio"] > 0.999:
                findings.append(f"{split_name}/{mode}: sampled targets are almost all zero.")
            metrics = mode_report["persistence_metrics"]
            if metrics["MAE"] == 0.0 and metrics["RMSE"] == 0.0:
                findings.append(f"{split_name}/{mode}: persistence baseline also returns zero metrics.")

    overall = report["flow_channel"]
    if overall["zero_ratio"] > 0.999:
        findings.append("Overall flow channel is almost entirely zero.")
    if overall["std"] == 0.0:
        findings.append("Overall flow channel has zero variance.")

    return findings


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    desc = load_desc(dataset_dir)
    shape = tuple(desc["shape"])
    data = load_data(dataset_dir, shape)
    flow = np.asarray(data[..., 0])

    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    null_val = desc["regular_settings"]["NULL_VAL"]
    dataset_name = dataset_dir.name

    os.chdir(BASICTS_ROOT)

    report = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_name,
        "shape": list(shape),
        "input_len": input_len,
        "output_len": output_len,
        "null_val": null_val,
        "flow_channel": flow_stats(flow),
        "splits": {},
    }

    for split_name in ("full", "1m"):
        split_filename = f"split_indices_{split_name}.npz"
        split_payload = load_split(dataset_dir, split_filename)
        split_report = {
            "file": str(dataset_dir / split_filename),
            "indices": {mode: split_stats(indices) for mode, indices in split_payload.items()},
            "datasets": {},
        }

        sample_target_for_scaler = None
        for mode in ("train", "valid", "test"):
            dataset = build_dataset(dataset_name, split_filename, mode, input_len, output_len)
            starts = dataset.prediction_starts
            mode_report = {
                "num_windows": int(len(dataset)),
                "first_prediction_start": int(starts[0]) if len(starts) else None,
                "last_prediction_start": int(starts[-1]) if len(starts) else None,
                "sample_targets": summarize_window_targets(dataset, null_val, args.sample_windows),
                "persistence_metrics": persistence_metrics(dataset, null_val, args.sample_windows),
            }
            split_report["datasets"][mode] = mode_report

            if sample_target_for_scaler is None and len(dataset) > 0:
                sample_target_for_scaler = dataset[0]["target"][..., 0]

        if sample_target_for_scaler is not None:
            split_report["scaler"] = summarize_scaler(dataset_name, split_filename, sample_target_for_scaler)
        else:
            split_report["scaler"] = None

        report["splits"][split_name] = split_report

    report["findings"] = detect_issues(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

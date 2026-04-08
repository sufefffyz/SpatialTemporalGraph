#!/usr/bin/env python3
"""
Stratified evaluation for PeMS forecasting checkpoints.

This script reads:
- datasets/<DATASET>/sensor_catalog.csv
- checkpoints/.../test_results/{inputs,predictions,targets}.npy

and reports metrics by:
- group: ALL / ML / OR / FR / Ramp
- horizon: overall / h3 / h6 / h12 (configurable)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(message, flush=True)


def resolve_existing_path(path_str: str, desc: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_num_samples(npy_path: Path, output_len: int, num_nodes: int, channels: int = 1) -> int:
    bytes_per_sample = np.dtype(np.float32).itemsize * output_len * num_nodes * channels
    total_bytes = npy_path.stat().st_size
    if total_bytes % bytes_per_sample != 0:
        raise ValueError(f"文件大小与预期 shape 不一致: {npy_path}")
    return total_bytes // bytes_per_sample


def load_memmap_array(path: Path, output_len: int, num_nodes: int) -> np.memmap:
    num_samples = infer_num_samples(path, output_len, num_nodes, channels=1)
    shape = (num_samples, output_len, num_nodes, 1)
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def masked_mae_np(pred: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - tgt)))


def masked_rmse_np(pred: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - tgt) ** 2)))


def masked_mape_np(pred: np.ndarray, tgt: np.ndarray) -> float:
    mask = np.abs(tgt) > 5e-5
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - tgt[mask]) / tgt[mask])))


def masked_wape_np(pred: np.ndarray, tgt: np.ndarray) -> float:
    # Match BasicTS masked_wape implementation: sum along horizon dim then mean.
    numerator = np.sum(np.abs(pred - tgt), axis=1)
    denominator = np.sum(np.abs(tgt), axis=1) + 5e-5
    return float(np.mean(numerator / denominator))


def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    return {
        "MAE": masked_mae_np(pred, tgt),
        "RMSE": masked_rmse_np(pred, tgt),
        "MAPE": masked_mape_np(pred, tgt),
        "WAPE": masked_wape_np(pred, tgt),
    }


def build_group_masks(sensor_catalog: pd.DataFrame, num_nodes: int) -> dict[str, np.ndarray]:
    if "Type" not in sensor_catalog.columns:
        raise ValueError("sensor_catalog.csv 缺少 Type 列")
    sensor_type = sensor_catalog["Type"].astype(str).str.upper().fillna("")
    if len(sensor_type) != num_nodes:
        raise ValueError(f"sensor_catalog 节点数 {len(sensor_type)} 与数据节点数 {num_nodes} 不一致")

    return {
        "ALL": np.ones(num_nodes, dtype=bool),
        "ML": (sensor_type == "ML").to_numpy(),
        "OR": (sensor_type == "OR").to_numpy(),
        "FR": (sensor_type == "FR").to_numpy(),
        "Ramp": sensor_type.isin(["OR", "FR"]).to_numpy(),
    }


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--run 格式应为 label=/abs/path/to/checkpoint_dir，收到: {spec}")
    label, path_str = spec.split("=", 1)
    return label, resolve_existing_path(path_str, f"checkpoint 目录 {label}")


def evaluate_run(
    label: str,
    ckpt_dir: Path,
    dataset_dir: Path,
    horizons: list[int],
) -> pd.DataFrame:
    desc = load_json(dataset_dir / "desc.json")
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    num_nodes = int(desc["num_nodes"])

    pred_path = ckpt_dir / "test_results" / "predictions.npy"
    tgt_path = ckpt_dir / "test_results" / "targets.npy"
    if not pred_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"{label} 缺少 test_results/predictions.npy 或 targets.npy，请先运行 evaluate 并开启 SAVE_RESULTS"
        )

    pred = load_memmap_array(pred_path, output_len, num_nodes)
    tgt = load_memmap_array(tgt_path, output_len, num_nodes)

    sensor_catalog = pd.read_csv(dataset_dir / "sensor_catalog.csv")
    masks = build_group_masks(sensor_catalog, num_nodes)

    rows = []
    for group_name, mask in masks.items():
        if mask.sum() == 0:
            continue

        group_pred = pred[:, :, mask, :]
        group_tgt = tgt[:, :, mask, :]

        overall_metrics = compute_metrics(group_pred, group_tgt)
        rows.append(
            {
                "run": label,
                "group": group_name,
                "horizon": "overall",
                "num_nodes": int(mask.sum()),
                **overall_metrics,
            }
        )

        for horizon in horizons:
            horizon_idx = horizon - 1
            if horizon_idx < 0 or horizon_idx >= output_len:
                continue
            pred_h = group_pred[:, horizon_idx : horizon_idx + 1, :, :]
            tgt_h = group_tgt[:, horizon_idx : horizon_idx + 1, :, :]
            horizon_metrics = compute_metrics(pred_h, tgt_h)
            rows.append(
                {
                    "run": label,
                    "group": group_name,
                    "horizon": f"h{horizon}",
                    "num_nodes": int(mask.sum()),
                    **horizon_metrics,
                }
            )

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stratified evaluation for PeMS checkpoint outputs.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Absolute path to BasicTS dataset dir, e.g. .../datasets/PEMSD3_2025_full_phys",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: label=/abs/path/to/checkpoint_dir . Can be passed multiple times.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=[3, 6, 12],
        help="Horizons to report in addition to overall. Default: 3 6 12",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path. Default: <dataset-dir>/stratified_eval.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = resolve_existing_path(args.dataset_dir, "dataset dir")
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else dataset_dir / "stratified_eval.csv"
    )

    frames = []
    for spec in args.run:
        label, ckpt_dir = parse_run_spec(spec)
        log(f"评估 {label}: {ckpt_dir}")
        frames.append(evaluate_run(label, ckpt_dir, dataset_dir, args.horizons))

    result_df = pd.concat(frames, ignore_index=True)
    result_df = result_df.sort_values(["group", "horizon", "run"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    log("")
    log(result_df.round(4).to_string(index=False))
    log("")
    log(f"已保存到: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

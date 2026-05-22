#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify unpredictable anomaly contributions to SD baseline MAE/RMSE."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Diagnostic summary.json with run paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--median-min", type=float, default=100.0)
    parser.add_argument("--extreme-ratio", type=float, default=0.5)
    parser.add_argument("--recent-drop-lookback", type=int, default=6)
    parser.add_argument("--recent-drop-min-prev", type=float, default=100.0)
    parser.add_argument("--recent-drop-min-delta", type=float, default=100.0)
    parser.add_argument("--recent-drop-ratio", type=float, default=0.5)
    parser.add_argument(
        "--input-zero-target-min-count",
        type=int,
        default=6,
        help="Mark target-node input as poor when at least this many history flow values are zero.",
    )
    parser.add_argument(
        "--input-zero-sample-share",
        type=float,
        default=0.05,
        help="Mark a whole test sample as poor when flow-zero share over [input_len, nodes] exceeds this value.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def load_saved_array(
    path: Path,
    dtype: str,
    output_len: int,
    num_nodes: int,
) -> np.ndarray:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        itemsize = np.dtype(dtype).itemsize
        denom = output_len * num_nodes * itemsize
        size = path.stat().st_size
        if size % denom != 0:
            raise ValueError(f"Cannot infer raw memmap shape for {path}: bytes={size}, denominator={denom}")
        num_samples = size // denom
        arr = np.memmap(path, dtype=dtype, mode="r", shape=(num_samples, output_len, num_nodes))
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [samples, horizon, nodes], got {arr.shape} at {path}")
    return arr


def weekday_codes(data: np.ndarray, indices: np.ndarray, slots_per_day: int) -> np.ndarray:
    if data.shape[-1] >= 3:
        codes = np.rint(np.asarray(data[indices, 0, 2]) * 7).astype(np.int16)
        return np.clip(codes, 0, 6)
    return ((indices // slots_per_day) % 7).astype(np.int16)


def build_seasonal_reference(
    data: np.ndarray,
    test_start: int,
    slots_per_day: int,
    target_channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = data.shape[1]
    q10 = np.full((slots_per_day, 7, num_nodes), np.nan, dtype=np.float32)
    median = np.full((slots_per_day, 7, num_nodes), np.nan, dtype=np.float32)
    indices = np.arange(test_start)
    dows = weekday_codes(data, indices, slots_per_day)
    for slot in range(slots_per_day):
        slot_mask = (indices % slots_per_day) == slot
        for dow in range(7):
            selected = indices[slot_mask & (dows == dow)]
            if selected.size == 0:
                continue
            values = np.asarray(data[selected, :, target_channel], dtype=np.float32)
            values = np.where(np.isfinite(values) & (values != 0), values, np.nan)
            with np.errstate(invalid="ignore"):
                q10[slot, dow] = np.nanquantile(values, 0.10, axis=0)
                median[slot, dow] = np.nanmedian(values, axis=0)
    return q10, median


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    summary = load_json(args.summary)
    dataset_dir = args.dataset_dir.expanduser().resolve() if args.dataset_dir else Path(summary["dataset_dir"]).expanduser()
    desc = load_json(dataset_dir / "desc.json")
    regular = desc.get("regular_settings", {})
    shape = tuple(int(dim) for dim in desc["shape"])
    num_steps, num_nodes, _ = shape
    input_len = int(regular["INPUT_LEN"])
    output_len = int(regular["OUTPUT_LEN"])
    null_val = regular.get("NULL_VAL", 0.0)
    null_val = None if null_val is None else float(null_val)
    frequency = int(desc.get("frequency (minutes)", 1))
    slots_per_day = max(1, int(round(24 * 60 / frequency)))
    ratios = regular.get("TRAIN_VAL_TEST_RATIO", [0.6, 0.2, 0.2])
    test_start = int(round(num_steps * (float(ratios[0]) + float(ratios[1]))))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.memmap(dataset_dir / "data.dat", dtype=args.dtype, mode="r", shape=shape)
    seasonal_q10, seasonal_median = build_seasonal_reference(
        data, test_start, slots_per_day, args.target_channel
    )

    first_run = summary["runs"][0]
    target = load_saved_array(Path(first_run["result_dir"]) / "targets.npy", args.dtype, output_len, num_nodes)
    num_samples = int(target.shape[0])
    sample_indices = np.arange(num_samples, dtype=np.int64)

    valid_masks: list[np.ndarray] = []
    category_masks: dict[str, list[np.ndarray]] = {
        "contextual_low_q10": [],
        "contextual_low_extreme": [],
        "recent_drop": [],
        "input_zero_target_half": [],
        "input_zero_target_all": [],
        "input_zero_sample_high": [],
        "input_quality_union": [],
        "broad_union": [],
        "strict_union": [],
        "strict_plus_input_union": [],
        "broad_plus_input_union": [],
    }

    input_zero_count = np.zeros((num_samples, num_nodes), dtype=np.int16)
    sample_zero_share = np.zeros(num_samples, dtype=np.float32)
    for sample in range(num_samples):
        input_flow = np.asarray(
            data[test_start + sample : test_start + sample + input_len, :, args.target_channel],
            dtype=np.float32,
        )
        zero_mask = input_flow == 0
        input_zero_count[sample] = zero_mask.sum(axis=0)
        sample_zero_share[sample] = float(zero_mask.mean())
    input_zero_target_half_base = input_zero_count >= args.input_zero_target_min_count
    input_zero_target_all_base = input_zero_count >= input_len
    input_zero_sample_high_base = sample_zero_share >= args.input_zero_sample_share

    for h_idx in range(output_len):
        y = np.asarray(target[:, h_idx, :], dtype=np.float32)
        valid = np.isfinite(y)
        if null_val is not None and np.isfinite(null_val):
            valid &= ~np.isclose(y, null_val)

        full_idx = test_start + sample_indices + input_len + h_idx
        slots = (full_idx % slots_per_day).astype(np.int16)
        dows = weekday_codes(data, full_idx, slots_per_day)
        q10 = seasonal_q10[slots, dows, :]
        median = seasonal_median[slots, dows, :]

        contextual_low_q10 = (
            valid
            & np.isfinite(q10)
            & np.isfinite(median)
            & (median >= args.median_min)
            & (y <= q10)
        )
        contextual_low_extreme = contextual_low_q10 & (y <= args.extreme_ratio * median)

        prev_max = np.full_like(y, np.nan, dtype=np.float32)
        for lag in range(1, args.recent_drop_lookback + 1):
            prev = np.asarray(data[full_idx - lag, :, args.target_channel], dtype=np.float32)
            prev_max = np.fmax(prev_max, prev)
        recent_drop = (
            valid
            & np.isfinite(prev_max)
            & (prev_max >= args.recent_drop_min_prev)
            & (y <= args.recent_drop_ratio * prev_max)
            & ((prev_max - y) >= args.recent_drop_min_delta)
        )
        input_zero_target_half = valid & input_zero_target_half_base
        input_zero_target_all = valid & input_zero_target_all_base
        input_zero_sample_high = valid & input_zero_sample_high_base.reshape(-1, 1)
        input_quality_union = input_zero_target_half | input_zero_sample_high
        broad_union = contextual_low_q10 | recent_drop
        strict_union = contextual_low_extreme | recent_drop

        valid_masks.append(valid)
        category_masks["contextual_low_q10"].append(contextual_low_q10)
        category_masks["contextual_low_extreme"].append(contextual_low_extreme)
        category_masks["recent_drop"].append(recent_drop)
        category_masks["input_zero_target_half"].append(input_zero_target_half)
        category_masks["input_zero_target_all"].append(input_zero_target_all)
        category_masks["input_zero_sample_high"].append(input_zero_sample_high)
        category_masks["input_quality_union"].append(input_quality_union)
        category_masks["broad_union"].append(broad_union)
        category_masks["strict_union"].append(strict_union)
        category_masks["strict_plus_input_union"].append(strict_union | input_quality_union)
        category_masks["broad_plus_input_union"].append(broad_union | input_quality_union)

    total_valid_points = int(sum(mask.sum() for mask in valid_masks))
    category_count_rows = []
    for category, masks in category_masks.items():
        points = int(sum(mask.sum() for mask in masks))
        category_count_rows.append(
            {"category": category, "points": points, "point_share": points / total_valid_points}
        )
    write_csv(output_dir / "anomaly_point_counts.csv", category_count_rows)

    overlap_q10_drop = int(
        sum(
            (category_masks["contextual_low_q10"][h] & category_masks["recent_drop"][h]).sum()
            for h in range(output_len)
        )
    )
    overlap_extreme_drop = int(
        sum(
            (category_masks["contextual_low_extreme"][h] & category_masks["recent_drop"][h]).sum()
            for h in range(output_len)
        )
    )

    model_rows = []
    category_rows = []
    for run in summary["runs"]:
        model_name = run["name"]
        pred = load_saved_array(Path(run["result_dir"]) / "predictions.npy", args.dtype, output_len, num_nodes)
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0
        category_abs = {category: 0.0 for category in category_masks}
        category_sq = {category: 0.0 for category in category_masks}
        category_count = {category: 0 for category in category_masks}

        for h_idx in range(output_len):
            y = np.asarray(target[:, h_idx, :], dtype=np.float32)
            p = np.asarray(pred[:, h_idx, :], dtype=np.float32)
            valid = valid_masks[h_idx]
            abs_error = np.abs(p - y)
            sq_error = (p - y) ** 2
            total_abs += float(abs_error[valid].sum(dtype=np.float64))
            total_sq += float(sq_error[valid].sum(dtype=np.float64))
            total_count += int(valid.sum())
            for category, masks in category_masks.items():
                mask = masks[h_idx]
                category_abs[category] += float(abs_error[mask].sum(dtype=np.float64))
                category_sq[category] += float(sq_error[mask].sum(dtype=np.float64))
                category_count[category] += int(mask.sum())

        full_mae = total_abs / total_count
        full_rmse = math.sqrt(total_sq / total_count)
        row = {"model": model_name, "full_mae": full_mae, "full_rmse": full_rmse, "valid_points": total_count}
        clean_prefixes = {
            "strict_union": "strict_clean",
            "broad_union": "broad_clean",
            "strict_plus_input_union": "strict_plus_input_clean",
            "broad_plus_input_union": "broad_plus_input_clean",
        }
        for category, prefix in clean_prefixes.items():
            clean_count = total_count - category_count[category]
            clean_abs = total_abs - category_abs[category]
            clean_sq = total_sq - category_sq[category]
            clean_mae = clean_abs / clean_count
            clean_rmse = math.sqrt(clean_sq / clean_count)
            row.update(
                {
                    f"{prefix}_excluded_points": category_count[category],
                    f"{prefix}_point_share": category_count[category] / total_count,
                    f"{prefix}_mae": clean_mae,
                    f"{prefix}_rmse": clean_rmse,
                    f"{prefix}_delta_mae": full_mae - clean_mae,
                    f"{prefix}_delta_rmse": full_rmse - clean_rmse,
                    f"{prefix}_mae_error_share": category_abs[category] / total_abs,
                    f"{prefix}_mse_error_share": category_sq[category] / total_sq,
                }
            )
        model_rows.append(row)

        for category in category_masks:
            clean_count = total_count - category_count[category]
            clean_mae = (total_abs - category_abs[category]) / clean_count
            clean_rmse = math.sqrt((total_sq - category_sq[category]) / clean_count)
            category_rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "points": category_count[category],
                    "point_share": category_count[category] / total_count,
                    "mae_contribution_per_all_point": category_abs[category] / total_count,
                    "mae_error_share": category_abs[category] / total_abs,
                    "mse_contribution_per_all_point": category_sq[category] / total_count,
                    "mse_error_share": category_sq[category] / total_sq,
                    "mae_after_excluding_category": clean_mae,
                    "rmse_after_excluding_category": clean_rmse,
                    "delta_mae": full_mae - clean_mae,
                    "delta_rmse": full_rmse - clean_rmse,
                }
            )

    model_rows = sorted(model_rows, key=lambda item: item["full_mae"])
    write_csv(output_dir / "model_metric_impact.csv", model_rows)
    write_csv(output_dir / "category_contribution_by_model.csv", category_rows)

    report_lines = [
        "# SD Unpredictable Anomaly Metric Impact",
        "",
        "## Anomaly Point Counts",
        "",
        "| Category | Points | Point share |",
        "|---|---:|---:|",
    ]
    for row in category_count_rows:
        report_lines.append(f"| {row['category']} | {row['points']} | {100 * row['point_share']:.3f}% |")
        report_lines.extend(
        [
            "",
            f"Overlap contextual_low_q10 and recent_drop: {overlap_q10_drop} points.",
            f"Overlap contextual_low_extreme and recent_drop: {overlap_extreme_drop} points.",
            f"Sample-level input zero share threshold: {args.input_zero_sample_share:.3f}.",
            f"Target-node input zero-count threshold: {args.input_zero_target_min_count}/{input_len}.",
            "",
            "## Strict-Union Clean Metrics",
            "",
            "| Model | Full MAE | Full RMSE | Anom MAE share | Anom MSE share | Clean MAE | Clean RMSE | delta MAE | delta RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        report_lines.append(
            f"| {row['model']} | {row['full_mae']:.3f} | {row['full_rmse']:.3f} | "
            f"{100 * row['strict_clean_mae_error_share']:.2f}% | "
            f"{100 * row['strict_clean_mse_error_share']:.2f}% | "
            f"{row['strict_clean_mae']:.3f} | {row['strict_clean_rmse']:.3f} | "
            f"{row['strict_clean_delta_mae']:.3f} | {row['strict_clean_delta_rmse']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Strict Plus Input-Quality Clean Metrics",
            "",
            "| Model | Full MAE | Full RMSE | Anom MAE share | Anom MSE share | Clean MAE | Clean RMSE | delta MAE | delta RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        report_lines.append(
            f"| {row['model']} | {row['full_mae']:.3f} | {row['full_rmse']:.3f} | "
            f"{100 * row['strict_plus_input_clean_mae_error_share']:.2f}% | "
            f"{100 * row['strict_plus_input_clean_mse_error_share']:.2f}% | "
            f"{row['strict_plus_input_clean_mae']:.3f} | {row['strict_plus_input_clean_rmse']:.3f} | "
            f"{row['strict_plus_input_clean_delta_mae']:.3f} | "
            f"{row['strict_plus_input_clean_delta_rmse']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Broad-Union Clean Metrics",
            "",
            "| Model | Full MAE | Full RMSE | Anom MAE share | Anom MSE share | Clean MAE | Clean RMSE | delta MAE | delta RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        report_lines.append(
            f"| {row['model']} | {row['full_mae']:.3f} | {row['full_rmse']:.3f} | "
            f"{100 * row['broad_clean_mae_error_share']:.2f}% | "
            f"{100 * row['broad_clean_mse_error_share']:.2f}% | "
            f"{row['broad_clean_mae']:.3f} | {row['broad_clean_rmse']:.3f} | "
            f"{row['broad_clean_delta_mae']:.3f} | {row['broad_clean_delta_rmse']:.3f} |"
        )
    (output_dir / "metric_impact_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    params = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "dataset": summary["dataset_name"],
        "frequency_minutes": frequency,
        "input_len": input_len,
        "output_len": output_len,
        "valid_points": total_valid_points,
        "definitions": {
            "contextual_low_q10": "same-slot same-weekday train median >= median_min and target <= train q10",
            "contextual_low_extreme": "contextual_low_q10 and target <= extreme_ratio * train median",
            "recent_drop": "max previous lookback target-node values >= min_prev, current <= ratio*prev_max, and prev_max-current >= min_delta",
            "input_zero_target_half": "target-node input history has at least input_zero_target_min_count zero flow values",
            "input_zero_target_all": "target-node input history has all zero flow values",
            "input_zero_sample_high": "whole input sample has flow-zero share >= input_zero_sample_share",
            "input_quality_union": "input_zero_target_half OR input_zero_sample_high",
            "strict_union": "contextual_low_extreme OR recent_drop",
            "broad_union": "contextual_low_q10 OR recent_drop",
            "strict_plus_input_union": "strict_union OR input_quality_union",
            "broad_plus_input_union": "broad_union OR input_quality_union",
        },
        "parameters": params,
        "overlap_contextual_low_q10_recent_drop": overlap_q10_drop,
        "overlap_contextual_low_extreme_recent_drop": overlap_extreme_drop,
        "outputs": [
            "anomaly_point_counts.csv",
            "model_metric_impact.csv",
            "category_contribution_by_model.csv",
            "metric_impact_report.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

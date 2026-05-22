#!/usr/bin/env python3
"""Evaluate classic baselines under the DTIGNN 30-to-1 setting.

The script consumes the Xuancheng DTIGNN-style NPZ produced by
``run_cityflow_dtignn_generation.py``.  It follows the experimental shape used
by "Modeling Network-level Traffic Flow Transitions on Sparse Data":

* chronological 6:2:2 split,
* 30 input steps,
* one-step-ahead prediction,
* MAE/RMSE/MAPE on traffic volume.

For sparse-observation settings, historical inputs are masked by
``observed_lsr_masks``.  Metrics are reported on all states, observed states,
and unobserved states so the effect of sparsity is visible instead of hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_ORDER = [
    "zero",
    "train_mean",
    "persistence",
    "moving_average",
    "seasonal_1h",
    "global_ar",
]


@dataclass(frozen=True)
class Split:
    target_times: np.ndarray
    train_targets: np.ndarray
    val_targets: np.ndarray
    test_targets: np.ndarray


@dataclass(frozen=True)
class MaskSpec:
    label: str
    missing_ratio: float
    mask: np.ndarray

    @property
    def observed_count(self) -> int:
        return int(self.mask.sum())

    @property
    def total_count(self) -> int:
        return int(self.mask.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, help="DTIGNN-style Xuancheng NPZ file.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV results.")
    parser.add_argument("--target-key", default="movement_volume_lsr")
    parser.add_argument("--input-window", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split-ratios", default="0.6,0.2,0.2")
    parser.add_argument(
        "--time-window-start",
        default=None,
        help="Optional target bucket start time in seconds or HH:MM. Example: 07:00.",
    )
    parser.add_argument(
        "--time-window-end",
        default=None,
        help="Optional exclusive target bucket start time in seconds or HH:MM. Example: 19:00.",
    )
    parser.add_argument(
        "--missing-ratios",
        default="dense,0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated mask labels. Use 'dense' plus ratios present in observed_lsr_masks.",
    )
    parser.add_argument("--seasonal-period", type=int, default=360, help="One hour for 10-second data.")
    parser.add_argument("--ar-train-rows", type=int, default=300_000)
    parser.add_argument("--ar-ridge", type=float, default=1.0)
    parser.add_argument("--ar-seed", type=int, default=2026)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--mape-eps", type=float, default=1e-9)
    parser.add_argument("--nonzero-eps", type=float, default=1e-9)
    parser.add_argument("--max-print-rows", type=int, default=80)
    return parser.parse_args()


def parse_ratios(text: str, expected: int) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != expected:
        raise ValueError(f"expected {expected} split ratios, got {parts}")
    total = sum(parts)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"split ratios must sum to 1, got {total}")
    return tuple(parts)  # type: ignore[return-value]


def parse_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if ":" not in value:
        return int(value)
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid time value {value!r}; expected seconds, HH:MM, or HH:MM:SS")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def scalar_to_py(value: np.ndarray) -> object:
    if value.shape == ():
        return value.tolist()
    return value.tolist()


def make_target_times(num_steps: int, input_window: int, horizon: int) -> np.ndarray:
    return np.arange(input_window + horizon - 1, num_steps, dtype=np.int64)


def filter_target_times(
    target_times: np.ndarray,
    bucket_start_s: np.ndarray,
    window_start_s: int | None,
    window_end_s: int | None,
) -> np.ndarray:
    selected = np.ones(len(target_times), dtype=bool)
    if window_start_s is not None:
        selected &= bucket_start_s[target_times] >= window_start_s
    if window_end_s is not None:
        selected &= bucket_start_s[target_times] < window_end_s
    return target_times[selected]


def make_split(target_times: np.ndarray, split_ratios: tuple[float, float, float]) -> Split:
    num_samples = len(target_times)
    if num_samples <= 0:
        raise ValueError("no target samples after time-window filtering")
    train_count = int(num_samples * split_ratios[0])
    val_count = int(num_samples * split_ratios[1])
    test_count = num_samples - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(f"invalid split counts: {(train_count, val_count, test_count)}")
    return Split(
        target_times=target_times,
        train_targets=target_times[:train_count],
        val_targets=target_times[train_count : train_count + val_count],
        test_targets=target_times[train_count + val_count :],
    )


def ratio_label(value: float) -> str:
    return f"{value:.6g}"


def load_masks(archive: np.lib.npyio.NpzFile, requested: str, target_shape: tuple[int, int]) -> list[MaskSpec]:
    labels = [part.strip() for part in requested.split(",") if part.strip()]
    masks: list[MaskSpec] = []
    seen: set[str] = set()
    for label in labels:
        if label == "dense":
            if label in seen:
                continue
            masks.append(MaskSpec(label="dense", missing_ratio=0.0, mask=np.ones(target_shape, dtype=bool)))
            seen.add(label)
            continue
        value = float(label)
        canonical = ratio_label(value)
        if canonical in seen:
            continue
        if "missing_ratios" not in archive or "observed_lsr_masks" not in archive:
            raise ValueError("NPZ has no missing_ratios/observed_lsr_masks; use only --missing-ratios dense")
        stored_ratios = np.asarray(archive["missing_ratios"], dtype=np.float64)
        matches = np.where(np.isclose(stored_ratios, value, rtol=1e-6, atol=1e-6))[0]
        if len(matches) == 0:
            raise ValueError(f"requested missing ratio {value} not found in NPZ ratios {stored_ratios.tolist()}")
        mask = np.asarray(archive["observed_lsr_masks"][int(matches[0])], dtype=bool)
        if mask.shape != target_shape:
            raise ValueError(f"mask shape {mask.shape} does not match target shape {target_shape}")
        masks.append(MaskSpec(label=canonical, missing_ratio=value, mask=mask))
        seen.add(canonical)
    return masks


def train_mean_prediction(values: np.ndarray, train_targets: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    train_y = values[train_targets].astype(np.float64, copy=False)
    means = np.zeros(observed_mask.shape, dtype=np.float32)
    if observed_mask.any():
        means[observed_mask] = train_y[:, observed_mask].mean(axis=0).astype(np.float32)
    global_by_turn = np.zeros(observed_mask.shape[1], dtype=np.float32)
    for turn_idx in range(observed_mask.shape[1]):
        turn_mask = observed_mask[:, turn_idx]
        if turn_mask.any():
            global_by_turn[turn_idx] = float(train_y[:, turn_mask, turn_idx].mean())
        else:
            global_by_turn[turn_idx] = float(train_y[:, :, turn_idx].mean())
    for turn_idx in range(observed_mask.shape[1]):
        missing_turn_mask = ~observed_mask[:, turn_idx]
        means[missing_turn_mask, turn_idx] = global_by_turn[turn_idx]
    return means


def moving_average_prediction(masked_values: np.ndarray, target_times: np.ndarray, input_window: int) -> np.ndarray:
    flat = masked_values.reshape(masked_values.shape[0], -1).astype(np.float32, copy=False)
    prefix = np.empty((flat.shape[0] + 1, flat.shape[1]), dtype=np.float32)
    prefix[0] = 0.0
    np.cumsum(flat, axis=0, dtype=np.float32, out=prefix[1:])
    sums = prefix[target_times] - prefix[target_times - input_window]
    return (sums / float(input_window)).reshape((len(target_times),) + masked_values.shape[1:])


def seasonal_prediction(
    masked_values: np.ndarray,
    target_times: np.ndarray,
    fallback: np.ndarray,
    period: int,
) -> np.ndarray:
    pred = np.empty((len(target_times),) + masked_values.shape[1:], dtype=np.float32)
    source_times = target_times - period
    valid = source_times >= 0
    if valid.any():
        pred[valid] = masked_values[source_times[valid]]
    if (~valid).any():
        pred[~valid] = fallback[~valid]
    return pred


def fit_global_ar(
    masked_values: np.ndarray,
    train_targets: np.ndarray,
    observed_mask: np.ndarray,
    input_window: int,
    train_rows: int,
    ridge: float,
    seed: int,
) -> np.ndarray:
    observed_flat = observed_mask.reshape(-1)
    observed_indices = np.flatnonzero(observed_flat)
    if len(observed_indices) == 0:
        coef = np.zeros(input_window + 1, dtype=np.float32)
        return coef

    rng = np.random.default_rng(seed)
    rows = min(train_rows, len(train_targets) * len(observed_indices))
    time_idx = rng.choice(train_targets, size=rows, replace=True)
    feat_idx = rng.choice(observed_indices, size=rows, replace=True)
    flat = masked_values.reshape(masked_values.shape[0], -1).astype(np.float32, copy=False)
    offsets = np.arange(input_window, 0, -1, dtype=np.int64)
    lag_times = time_idx[:, None] - offsets[None, :]
    design = flat[lag_times, feat_idx[:, None]].astype(np.float64, copy=False)
    design = np.concatenate([np.ones((rows, 1), dtype=np.float64), design], axis=1)
    target = flat[time_idx, feat_idx].astype(np.float64, copy=False)

    gram = design.T @ design
    gram += ridge * np.eye(gram.shape[0], dtype=np.float64)
    gram[0, 0] -= ridge
    rhs = design.T @ target
    coef = np.linalg.solve(gram, rhs)
    return coef.astype(np.float32)


def predict_global_ar(
    masked_values: np.ndarray,
    target_times: np.ndarray,
    input_window: int,
    coef: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    flat = masked_values.reshape(masked_values.shape[0], -1).astype(np.float32, copy=False)
    output = np.empty((len(target_times), flat.shape[1]), dtype=np.float32)
    lag_coef = coef[1:].astype(np.float32, copy=False)
    offsets = np.arange(input_window, 0, -1, dtype=np.int64)
    for start in range(0, len(target_times), batch_size):
        end = min(start + batch_size, len(target_times))
        lag_times = target_times[start:end, None] - offsets[None, :]
        history = flat[lag_times]
        pred = coef[0] + np.tensordot(history, lag_coef, axes=([1], [0]))
        np.maximum(pred, 0.0, out=pred)
        output[start:end] = pred
    return output.reshape((len(target_times),) + masked_values.shape[1:])


def paper_mape(abs_err: np.ndarray, target: np.ndarray, eps: float) -> float:
    nonzero = np.abs(target) > eps
    terms = np.empty_like(abs_err, dtype=np.float64)
    terms[nonzero] = abs_err[nonzero] / np.abs(target[nonzero])
    terms[~nonzero] = (abs_err[~nonzero] > eps).astype(np.float64)
    return float(terms.mean())


def metric_row(
    prediction: np.ndarray,
    target: np.ndarray,
    feature_mask: np.ndarray,
    model: str,
    mask_spec: MaskSpec,
    eval_scope: str,
    elapsed_s: float,
    eps: float,
    nonzero_eps: float,
) -> dict[str, object] | None:
    flat_mask = feature_mask.reshape(-1)
    if not flat_mask.any():
        return None
    pred = prediction.reshape(prediction.shape[0], -1)[:, flat_mask].astype(np.float64, copy=False)
    true = target.reshape(target.shape[0], -1)[:, flat_mask].astype(np.float64, copy=False)
    err = pred - true
    abs_err = np.abs(err)
    sq_err = err * err
    target_abs = np.abs(true)
    nonzero = target_abs > nonzero_eps
    nonzero_mae = float(abs_err[nonzero].mean()) if nonzero.any() else float("nan")
    return {
        "missing_ratio": mask_spec.missing_ratio,
        "mask_label": mask_spec.label,
        "observed_features": mask_spec.observed_count,
        "total_features": mask_spec.total_count,
        "observed_feature_rate": mask_spec.observed_count / mask_spec.total_count,
        "model": model,
        "eval_scope": eval_scope,
        "n_values": int(true.size),
        "target_mean": float(true.mean()),
        "target_zero_rate": float((target_abs <= nonzero_eps).mean()),
        "target_nonzero_count": int(nonzero.sum()),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "mape": paper_mape(abs_err, true, eps),
        "wape": float(abs_err.sum() / max(target_abs.sum(), eps)),
        "nonzero_mae": nonzero_mae,
        "elapsed_s": elapsed_s,
    }


def evaluate_model(
    prediction: np.ndarray,
    target: np.ndarray,
    model: str,
    mask_spec: MaskSpec,
    eps: float,
    nonzero_eps: float,
    elapsed_s: float,
) -> list[dict[str, object]]:
    all_mask = np.ones_like(mask_spec.mask, dtype=bool)
    scopes: list[tuple[str, np.ndarray]] = [
        ("all", all_mask),
        ("observed", mask_spec.mask),
        ("unobserved", ~mask_spec.mask),
    ]
    rows: list[dict[str, object]] = []
    for scope, feature_mask in scopes:
        row = metric_row(
            prediction=prediction,
            target=target,
            feature_mask=feature_mask,
            model=model,
            mask_spec=mask_spec,
            eval_scope=scope,
            elapsed_s=elapsed_s,
            eps=eps,
            nonzero_eps=nonzero_eps,
        )
        if row is not None:
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_compact_table(rows: list[dict[str, object]], max_rows: int) -> None:
    all_rows = [row for row in rows if row["eval_scope"] == "all"]
    all_rows.sort(key=lambda row: (float(row["missing_ratio"]), MODEL_ORDER.index(str(row["model"]))))
    print("mask_label,model,scope,MAE,RMSE,MAPE,WAPE,target_zero_rate")
    for row in all_rows[:max_rows]:
        print(
            f"{row['mask_label']},{row['model']},{row['eval_scope']},"
            f"{float(row['mae']):.6f},{float(row['rmse']):.6f},"
            f"{float(row['mape']):.6f},{float(row['wape']):.6f},"
            f"{float(row['target_zero_rate']):.6f}"
        )


def main() -> None:
    args = parse_args()
    start_all = time.time()
    input_path = Path(args.input_npz)
    output_dir = Path(args.output_dir)
    split_ratios = parse_ratios(args.split_ratios, expected=3)
    window_start_s = parse_seconds(args.time_window_start)
    window_end_s = parse_seconds(args.time_window_end)
    if window_start_s is not None and window_end_s is not None and window_start_s >= window_end_s:
        raise ValueError("--time-window-start must be earlier than --time-window-end")

    with np.load(input_path, allow_pickle=False) as archive:
        if args.target_key not in archive:
            raise KeyError(f"{args.target_key} not found in {input_path}")
        values = np.asarray(archive[args.target_key], dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(f"expected [time, road, turn], got {values.shape}")
        masks = load_masks(archive, args.missing_ratios, values.shape[1:])
        if "bucket_start_s" in archive:
            bucket_start_s = np.asarray(archive["bucket_start_s"], dtype=np.float64)
        else:
            bucket_seconds = float(scalar_to_py(archive["bucket_seconds"])) if "bucket_seconds" in archive else 1.0
            start_second = float(scalar_to_py(archive["start_second"])) if "start_second" in archive else 0.0
            bucket_start_s = start_second + np.arange(values.shape[0], dtype=np.float64) * bucket_seconds
        metadata = {
            "input_npz": str(input_path),
            "target_key": args.target_key,
            "date": scalar_to_py(archive["date"]) if "date" in archive else None,
            "source_config": scalar_to_py(archive["source_config"]) if "source_config" in archive else None,
            "signal_policy": scalar_to_py(archive["signal_policy"]) if "signal_policy" in archive else None,
            "shape": list(values.shape),
            "bucket_seconds": scalar_to_py(archive["bucket_seconds"]) if "bucket_seconds" in archive else None,
        }

    target_times = make_target_times(values.shape[0], args.input_window, args.horizon)
    target_times = filter_target_times(target_times, bucket_start_s, window_start_s, window_end_s)
    split = make_split(target_times, split_ratios)
    target_test = values[split.test_targets]
    rows: list[dict[str, object]] = []
    ar_coefficients: dict[str, list[float]] = {}

    for mask_spec in masks:
        print(
            f"[mask] {mask_spec.label} observed={mask_spec.observed_count}/{mask_spec.total_count} "
            f"({mask_spec.observed_count / mask_spec.total_count:.3f})"
        )
        masked_values = values * mask_spec.mask[None, :, :]

        baselines: dict[str, np.ndarray] = {}
        model_start = time.time()
        baselines["zero"] = np.zeros_like(target_test, dtype=np.float32)
        rows.extend(
            evaluate_model(
                baselines["zero"],
                target_test,
                "zero",
                mask_spec,
                args.mape_eps,
                args.nonzero_eps,
                time.time() - model_start,
            )
        )

        model_start = time.time()
        mean_feature = train_mean_prediction(values, split.train_targets, mask_spec.mask)
        baselines["train_mean"] = np.broadcast_to(mean_feature, target_test.shape)
        rows.extend(
            evaluate_model(
                baselines["train_mean"],
                target_test,
                "train_mean",
                mask_spec,
                args.mape_eps,
                args.nonzero_eps,
                time.time() - model_start,
            )
        )

        model_start = time.time()
        baselines["persistence"] = masked_values[split.test_targets - args.horizon]
        rows.extend(
            evaluate_model(
                baselines["persistence"],
                target_test,
                "persistence",
                mask_spec,
                args.mape_eps,
                args.nonzero_eps,
                time.time() - model_start,
            )
        )

        model_start = time.time()
        baselines["moving_average"] = moving_average_prediction(masked_values, split.test_targets, args.input_window)
        rows.extend(
            evaluate_model(
                baselines["moving_average"],
                target_test,
                "moving_average",
                mask_spec,
                args.mape_eps,
                args.nonzero_eps,
                time.time() - model_start,
            )
        )

        model_start = time.time()
        baselines["seasonal_1h"] = seasonal_prediction(
            masked_values,
            split.test_targets,
            fallback=baselines["persistence"],
            period=args.seasonal_period,
        )
        rows.extend(
            evaluate_model(
                baselines["seasonal_1h"],
                target_test,
                "seasonal_1h",
                mask_spec,
                args.mape_eps,
                args.nonzero_eps,
                time.time() - model_start,
            )
        )

        model_start = time.time()
        coef = fit_global_ar(
            masked_values=masked_values,
            train_targets=split.train_targets,
            observed_mask=mask_spec.mask,
            input_window=args.input_window,
            train_rows=args.ar_train_rows,
            ridge=args.ar_ridge,
            seed=args.ar_seed,
        )
        ar_coefficients[mask_spec.label] = [float(x) for x in coef.tolist()]
        ar_pred = predict_global_ar(
            masked_values=masked_values,
            target_times=split.test_targets,
            input_window=args.input_window,
            coef=coef,
            batch_size=args.predict_batch_size,
        )
        rows.extend(
            evaluate_model(
                ar_pred,
                target_test,
                "global_ar",
                mask_spec,
                args.mape_eps,
                args.nonzero_eps,
                time.time() - model_start,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classic_baselines_metrics.csv"
    json_path = output_dir / "classic_baselines_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "metadata": metadata,
        "setting": {
            "paper": "Modeling Network-level Traffic Flow Transitions on Sparse Data",
            "split_ratios": list(split_ratios),
            "input_window": args.input_window,
            "horizon": args.horizon,
            "time_window_start_s": window_start_s,
            "time_window_end_s": window_end_s,
            "seasonal_period": args.seasonal_period,
            "metrics": ["MAE", "RMSE", "MAPE", "WAPE", "nonzero_MAE"],
            "target_samples_after_time_filter": int(len(split.target_times)),
            "first_target_time_s": float(bucket_start_s[split.target_times[0]]),
            "last_target_time_s": float(bucket_start_s[split.target_times[-1]]),
            "target_test_steps": int(len(split.test_targets)),
            "train_samples": int(len(split.train_targets)),
            "val_samples": int(len(split.val_targets)),
            "test_samples": int(len(split.test_targets)),
        },
        "reported_deviation_note": (
            "These are lightweight classic baselines, not the paper's neural baselines "
            "STGCN/STSGCN/ASTGCN/ASTGNN/GraphWaveNet. Sparse inputs are masked using "
            "the generated observed_lsr_masks; evaluation is reported separately for all, "
            "observed, and unobserved road-turn states."
        ),
        "ar_coefficients": ar_coefficients,
        "rows": rows,
        "elapsed_s": time.time() - start_all,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")
    print_compact_table(rows, args.max_print_rows)


if __name__ == "__main__":
    main()

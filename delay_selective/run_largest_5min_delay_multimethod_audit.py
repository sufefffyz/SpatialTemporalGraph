#!/usr/bin/env python3
"""Multi-method delay-effect audit for LargeST SD/GLA/GBA 5-minute datasets.

This script complements ``run_largest_5min_delay_audit.py`` with three things
needed for delay-selective validation:

1. A strict default confidence threshold, ``corr >= 0.8``.
2. Multiple delay estimators inspired by current baselines:
   - exact 5-minute lagged Pearson MCC on residualized signals;
   - STDDE-style smoothed MCC using natural cubic spline interpolation;
   - LIFT-style FFT cross-correlation with absolute lead-lag scores.
3. Month-level and week-daily windows for checking delay stability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_largest_5min_delay_audit import (
        DEFAULT_DATASET_CANDIDATES,
        compute_lag_scores,
        discover_default_datasets,
        edge_distances_km,
        load_graph_variants,
        load_json,
        load_lat_lng,
        parse_bins,
        parse_dataset_specs,
        residualize_matrix,
        select_edges,
        write_csv,
        write_json,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from run_largest_5min_delay_audit import (
        DEFAULT_DATASET_CANDIDATES,
        compute_lag_scores,
        discover_default_datasets,
        edge_distances_km,
        load_graph_variants,
        load_json,
        load_lat_lng,
        parse_bins,
        parse_dataset_specs,
        residualize_matrix,
        select_edges,
        write_csv,
        write_json,
    )


METHOD_CHOICES = [
    "mcc_5min_resid",
    "mcc_5min_raw",
    "stdde_spline_fft_mcc",
    "lift_fft_abs",
    "diff_mcc_5min",
]

WINDOW_CHOICES = ["train_prefix", "month", "week_daily"]

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict multi-method delay audits on SD/GLA/GBA 5-minute LargeST datasets."
    )
    parser.add_argument(
        "--output-dir",
        default="delay_selective/outputs/largest_5min_delay_multimethod_audit",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as NAME:PATH. Can be repeated. Defaults to SD/GLA/GBA 5min candidates.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-file", default="split_indices_full.npz")
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=20160,
        help="For train_prefix only: 0 means use the whole selected split.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=20000,
        help="0 means score all eligible graph edges. Large GLA/GBA runs can be expensive.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mcc_5min_resid", "stdde_spline_fft_mcc", "lift_fft_abs"],
        choices=METHOD_CHOICES,
        help="Delay estimators to compare.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["month", "week_daily"],
        choices=WINDOW_CHOICES,
        help="Temporal windows to audit.",
    )
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum lag in native 5-minute steps.")
    parser.add_argument(
        "--interp-minutes",
        type=int,
        default=1,
        help="Interpolation step for stdde_spline_fft_mcc. Default follows the old SD notebook.",
    )
    parser.add_argument(
        "--graph-variants",
        nargs="+",
        default=["distthre"],
        choices=["distthre", "physical_dir", "physical_bidir"],
        help="Default uses only the LargeST built-in road-network distance graph.",
    )
    parser.add_argument("--residualize", default="time_of_day", choices=["none", "mean", "time_of_day"])
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.80,
        help="Minimum correlation confidence for accepting a non-zero delay.",
    )
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--min-edge-std", type=float, default=1e-6)
    parser.add_argument("--score-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--month-start-day", type=int, default=0)
    parser.add_argument("--month-num-days", type=int, default=31)
    parser.add_argument("--week-start-day", type=int, default=0)
    parser.add_argument("--week-num-days", type=int, default=7)
    parser.add_argument(
        "--distance-bins-km",
        default="0,1,2,5,10,20,50,100,inf",
        help="Comma-separated edge distance bins in kilometers when metadata has Lat/Lng.",
    )
    parser.add_argument(
        "--include-corr-columns",
        action="store_true",
        help="Write per-lag correlation columns into edge CSVs. Off by default to keep files light.",
    )
    parser.add_argument(
        "--allow-non-5min",
        action="store_true",
        help="Do not reject datasets whose desc.json frequency is not 5 minutes.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any default/requested dataset path is missing.",
    )
    return parser.parse_args()


def select_time_indices(desc: dict[str, Any], dataset_dir: Path, split_file: str, split: str) -> np.ndarray:
    num_time_steps = int(desc["num_time_steps"])
    if split == "all":
        return np.arange(num_time_steps, dtype=np.int64)

    split_path = dataset_dir / split_file
    if split_path.exists():
        splits = np.load(split_path)
        key = f"{split}_idx"
        if key not in splits:
            raise KeyError(f"{split_path} does not contain {key}")
        return np.asarray(splits[key], dtype=np.int64)

    ratios = desc.get("regular_settings", {}).get("TRAIN_VAL_TEST_RATIO", [0.6, 0.2, 0.2])
    train_len = int(num_time_steps * float(ratios[0]))
    val_len = int(num_time_steps * float(ratios[1]))
    if split == "train":
        return np.arange(0, train_len, dtype=np.int64)
    if split == "val":
        return np.arange(train_len, train_len + val_len, dtype=np.int64)
    if split == "test":
        return np.arange(train_len + val_len, num_time_steps, dtype=np.int64)
    raise ValueError(f"Unsupported split: {split}")


def load_flow_at_indices(
    dataset_dir: Path,
    desc: dict[str, Any],
    time_indices: np.ndarray,
    max_time_steps: int,
) -> np.ndarray:
    shape = tuple(int(x) for x in desc["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected BasicTS data shape [T,N,C], got {shape}")
    if max_time_steps > 0:
        time_indices = time_indices[: min(max_time_steps, time_indices.shape[0])]
    data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    return np.asarray(data[time_indices, :, 0], dtype=np.float32)


def load_flow_step_slice(
    dataset_dir: Path,
    desc: dict[str, Any],
    start_step: int,
    num_steps: int,
) -> np.ndarray:
    shape = tuple(int(x) for x in desc["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected BasicTS data shape [T,N,C], got {shape}")
    end_step = min(int(desc["num_time_steps"]), max(0, start_step) + max(0, num_steps))
    if end_step <= start_step:
        raise ValueError(f"Empty flow window: start_step={start_step}, num_steps={num_steps}")
    data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    return np.asarray(data[start_step:end_step, :, 0], dtype=np.float32)


def get_window_specs(
    desc: dict[str, Any],
    dataset_dir: Path,
    args: argparse.Namespace,
    frequency: int,
    steps_per_day: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "train_prefix" in args.windows:
        time_indices = select_time_indices(desc, dataset_dir, args.split_file, args.split)
        if args.max_time_steps > 0:
            time_indices = time_indices[: min(args.max_time_steps, time_indices.shape[0])]
        specs.append(
            {
                "window": "train_prefix",
                "label": f"{args.split}_prefix",
                "time_indices": time_indices,
                "start_day": None,
                "num_days": None,
                "day_offset": None,
                "day_in_window": None,
                "weekday": None,
            }
        )

    if "month" in args.windows:
        start_step = args.month_start_day * steps_per_day
        num_steps = args.month_num_days * steps_per_day
        actual_steps = max(0, min(int(desc["num_time_steps"]) - start_step, num_steps))
        specs.append(
            {
                "window": "month",
                "label": f"days_{args.month_start_day}_{args.month_start_day + args.month_num_days - 1}",
                "start_step": start_step,
                "num_steps": actual_steps,
                "start_day": args.month_start_day,
                "num_days": actual_steps / steps_per_day,
                "day_offset": None,
                "day_in_window": None,
                "weekday": None,
            }
        )

    if "week_daily" in args.windows:
        for day_in_window in range(args.week_num_days):
            day_offset = args.week_start_day + day_in_window
            start_step = day_offset * steps_per_day
            actual_steps = max(0, min(int(desc["num_time_steps"]) - start_step, steps_per_day))
            specs.append(
                {
                    "window": "week_daily",
                    "label": f"day_{day_offset}",
                    "start_step": start_step,
                    "num_steps": actual_steps,
                    "start_day": day_offset,
                    "num_days": actual_steps / steps_per_day,
                    "day_offset": day_offset,
                    "day_in_window": day_in_window,
                    "weekday": WEEKDAY_NAMES[day_in_window % len(WEEKDAY_NAMES)],
                }
            )
    # Keep the native frequency explicit in run_summary for sanity checks.
    for spec in specs:
        spec["frequency_minutes"] = frequency
    return specs


def load_window_flow(
    dataset_dir: Path,
    desc: dict[str, Any],
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    if "time_indices" in spec:
        return load_flow_at_indices(
            dataset_dir=dataset_dir,
            desc=desc,
            time_indices=np.asarray(spec["time_indices"], dtype=np.int64),
            max_time_steps=0,
        )
    return load_flow_step_slice(
        dataset_dir=dataset_dir,
        desc=desc,
        start_step=int(spec["start_step"]),
        num_steps=int(spec["num_steps"]),
    )


def interpolate_matrix(
    flow: np.ndarray,
    native_minutes: int,
    interp_minutes: int,
    method: str = "natural_cubic_spline",
) -> tuple[np.ndarray, str]:
    if interp_minutes <= 0:
        raise ValueError("--interp-minutes must be positive")
    if interp_minutes == native_minutes:
        return flow.astype(np.float32, copy=False), "none_native_resolution"
    if interp_minutes > native_minutes or native_minutes % interp_minutes != 0:
        raise ValueError(
            f"Interpolation step must evenly refine the native grid: native={native_minutes}, interp={interp_minutes}"
        )

    old_t = np.arange(flow.shape[0], dtype=np.float64) * native_minutes
    new_t = np.arange(0, old_t[-1] + interp_minutes, interp_minutes, dtype=np.float64)
    interp = np.empty((len(new_t), flow.shape[1]), dtype=np.float32)
    if method == "natural_cubic_spline":
        try:
            from scipy.interpolate import CubicSpline
        except Exception:
            method = "linear_fallback_no_scipy"
        else:
            for node_idx in range(flow.shape[1]):
                series = np.asarray(flow[:, node_idx], dtype=np.float64)
                if not np.isfinite(series).all():
                    finite = np.isfinite(series)
                    if finite.sum() < 2:
                        interp[:, node_idx] = np.nan
                        continue
                    series = np.interp(old_t, old_t[finite], series[finite])
                spline = CubicSpline(old_t, series, bc_type="natural")
                interp[:, node_idx] = spline(new_t).astype(np.float32)
            return interp, "natural_cubic_spline"

    for node_idx in range(flow.shape[1]):
        series = np.asarray(flow[:, node_idx], dtype=np.float64)
        finite = np.isfinite(series)
        if finite.sum() < 2:
            interp[:, node_idx] = np.nan
        else:
            interp[:, node_idx] = np.interp(new_t, old_t[finite], series[finite]).astype(np.float32)
    return interp, method


def prepare_method_flow(
    raw_flow: np.ndarray,
    method: str,
    frequency: int,
    steps_per_day: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    meta: dict[str, Any] = {"method": method}
    if method == "mcc_5min_raw":
        meta.update({"estimator": "exact_lagged_pearson", "residualize_applied": "none"})
        return raw_flow.astype(np.float32, copy=False), frequency, meta

    if method == "mcc_5min_resid":
        residualize = args.residualize
        if residualize == "time_of_day" and raw_flow.shape[0] <= steps_per_day:
            residualize = "mean"
            meta["residualize_note"] = "time_of_day_falls_back_to_mean_for_single_day_window"
        flow = residualize_matrix(raw_flow, residualize, steps_per_day)
        meta.update({"estimator": "exact_lagged_pearson", "residualize_applied": residualize})
        return flow, frequency, meta

    if method == "diff_mcc_5min":
        flow = np.diff(raw_flow.astype(np.float32, copy=False), axis=0)
        meta.update({"estimator": "exact_lagged_pearson", "residualize_applied": "first_difference"})
        return flow, frequency, meta

    if method == "stdde_spline_fft_mcc":
        flow, interp_method = interpolate_matrix(
            raw_flow,
            native_minutes=frequency,
            interp_minutes=args.interp_minutes,
            method="natural_cubic_spline",
        )
        meta.update(
            {
                "estimator": "fft_full_window_cross_correlation",
                "paper_reference": "STDDE fixed-delay preprocessing: smooth/interpolate then maximize Pearson MCC",
                "interpolation_method": interp_method,
                "residualize_applied": "none",
            }
        )
        return flow, args.interp_minutes, meta

    if method == "lift_fft_abs":
        meta.update(
            {
                "estimator": "fft_full_window_cross_correlation_abs",
                "paper_reference": "LIFT lead estimator: normalized window FFT cross-correlation and max absolute score",
                "residualize_applied": "none",
            }
        )
        return raw_flow.astype(np.float32, copy=False), frequency, meta

    raise ValueError(f"Unsupported method: {method}")


def fill_and_normalize_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    finite = np.isfinite(x)
    count = finite.sum(axis=0)
    safe_count = np.maximum(count, 1)
    means = np.where(finite, x, 0.0).sum(axis=0) / safe_count
    filled = np.where(finite, x, means[None, :])
    std = filled.std(axis=0)
    ok = (count >= 3) & np.isfinite(std) & (std > 0)
    normalized = np.zeros_like(filled, dtype=np.float64)
    normalized[:, ok] = (filled[:, ok] - means[None, ok]) / std[None, ok]
    return normalized, ok


def compute_fft_lag_scores(
    x: np.ndarray,
    edges: np.ndarray,
    max_lag: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr_by_lag = np.full((max_lag + 1, edges.shape[0]), np.nan, dtype=np.float64)
    count_by_lag = np.zeros((max_lag + 1, edges.shape[0]), dtype=np.float64)
    chunk_size = max(1, chunk_size)
    n = int(x.shape[0])
    if n < 3:
        return corr_by_lag, count_by_lag

    for start in range(0, edges.shape[0], chunk_size):
        end = min(edges.shape[0], start + chunk_size)
        src_idx = edges[start:end, 0]
        dst_idx = edges[start:end, 1]
        src, src_ok = fill_and_normalize_columns(x[:, src_idx])
        dst, dst_ok = fill_and_normalize_columns(x[:, dst_idx])
        ok = src_ok & dst_ok
        if not ok.any():
            continue

        # This matches the LIFT-style convention: source leads target by lag k
        # when target[t] aligns with source[t-k]. We keep only non-negative lags.
        src_fft = np.fft.rfft(src[:, ok], axis=0)
        dst_fft = np.fft.rfft(dst[:, ok], axis=0)
        corr = np.fft.irfft(dst_fft * np.conj(src_fft), n=n, axis=0) / n
        cols = np.arange(start, end, dtype=np.int64)[ok]
        corr_by_lag[:, cols] = corr[: max_lag + 1]
        for lag in range(max_lag + 1):
            count_by_lag[lag, cols] = max(0, n - lag)
    return corr_by_lag, count_by_lag


def score_delay(
    flow: np.ndarray,
    edges: np.ndarray,
    method: str,
    max_lag_steps: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if method in {"stdde_spline_fft_mcc", "lift_fft_abs"}:
        corr_by_lag, count_by_lag = compute_fft_lag_scores(
            x=flow,
            edges=edges,
            max_lag=max_lag_steps,
            chunk_size=chunk_size,
        )
    else:
        corr_by_lag, count_by_lag = compute_lag_scores(
            x=flow,
            edges=edges,
            max_lag=max_lag_steps,
            chunk_size=chunk_size,
        )

    has_any_corr = np.isfinite(corr_by_lag).any(axis=0)
    if method == "lift_fft_abs":
        score_by_lag = np.abs(corr_by_lag)
        score_mode = "absolute_corr"
    else:
        score_by_lag = corr_by_lag
        score_mode = "signed_positive_corr"

    score_for_argmax = np.where(np.isfinite(score_by_lag), score_by_lag, -np.inf)
    best_lags = np.argmax(score_for_argmax, axis=0)
    best_lags = np.where(has_any_corr, best_lags, 0)
    best_corrs = corr_by_lag[best_lags, np.arange(edges.shape[0])]
    best_scores = score_by_lag[best_lags, np.arange(edges.shape[0])]
    zero_corrs = corr_by_lag[0]
    zero_scores = score_by_lag[0]
    improvements = best_scores - zero_scores
    return corr_by_lag, count_by_lag, best_lags, best_corrs, best_scores, improvements, score_mode


def nan_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).any():
        return None
    return float(np.nanquantile(values, q))


def summarize_distance_bins(
    distances_km: np.ndarray | None,
    best_lags: np.ndarray,
    effective_lags: np.ndarray,
    improvements: np.ndarray,
    best_scores: np.ndarray,
    bins: np.ndarray,
    min_corr: float,
    min_improvement: float,
    high_conf_nonzero: np.ndarray,
) -> list[dict[str, Any]]:
    if distances_km is None:
        return []
    rows: list[dict[str, Any]] = []
    corr_filtered = best_scores < min_corr
    effective_nan = ~np.isfinite(effective_lags)
    effective_zero = np.isfinite(effective_lags) & (effective_lags == 0)
    low_improvement_nonzero = (
        (best_lags > 0)
        & (best_scores >= min_corr)
        & (improvements < min_improvement)
    )
    for lo, hi in zip(bins[:-1], bins[1:]):
        if math.isinf(hi):
            mask = distances_km >= lo
            label = f"[{lo:g}, inf)"
        else:
            mask = (distances_km >= lo) & (distances_km < hi)
            label = f"[{lo:g}, {hi:g})"
        n = int(mask.sum())
        rows.append(
            {
                "distance_bin_km": label,
                "num_edges": n,
                "raw_best_zero_ratio": float(np.mean(best_lags[mask] == 0)) if n else None,
                "invalid_corr_ratio": float(np.mean(corr_filtered[mask])) if n else None,
                "corr_filtered_ratio": float(np.mean(corr_filtered[mask])) if n else None,
                "effective_nan_ratio": float(np.mean(effective_nan[mask])) if n else None,
                "effective_zero_ratio": float(np.mean(effective_zero[mask])) if n else None,
                "effective_zero_or_invalid_ratio": float(
                    np.mean(effective_zero[mask] | effective_nan[mask])
                ) if n else None,
                "low_improvement_nonzero_ratio": float(
                    np.mean(low_improvement_nonzero[mask])
                ) if n else None,
                "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero[mask])) if n else None,
                "median_raw_best_lag_steps": nan_quantile(best_lags[mask], 0.5) if n else None,
                "median_effective_lag_steps": nan_quantile(effective_lags[mask], 0.5) if n else None,
                "median_best_score": nan_quantile(best_scores[mask], 0.5) if n else None,
                "median_corr_improvement": nan_quantile(improvements[mask], 0.5) if n else None,
            }
        )
    return rows


def summarize_scores(
    dataset: str,
    dataset_dir: Path,
    graph: str,
    window_spec: dict[str, Any],
    method: str,
    method_meta: dict[str, Any],
    flow: np.ndarray,
    method_minutes: int,
    adj: np.ndarray,
    edges: np.ndarray,
    num_eligible_edges: int,
    max_lag_steps: int,
    score_mode: str,
    best_lags: np.ndarray,
    best_corrs: np.ndarray,
    best_scores: np.ndarray,
    zero_corrs: np.ndarray,
    zero_scores: np.ndarray,
    improvements: np.ndarray,
    high_conf_nonzero: np.ndarray,
    effective_lags: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    corr_filtered = best_scores < args.min_corr
    effective_nan = ~np.isfinite(effective_lags)
    effective_zero = np.isfinite(effective_lags) & (effective_lags == 0)
    low_improvement_nonzero = (
        (best_lags > 0)
        & (best_scores >= args.min_corr)
        & (improvements < args.min_improvement)
    )
    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "graph": graph,
        "window": window_spec["window"],
        "window_label": window_spec["label"],
        "day_offset": window_spec.get("day_offset"),
        "day_in_window": window_spec.get("day_in_window"),
        "weekday": window_spec.get("weekday"),
        "window_start_day": window_spec.get("start_day"),
        "window_num_days": window_spec.get("num_days"),
        "method": method,
        "estimator": method_meta.get("estimator"),
        "score_mode": score_mode,
        "frequency_minutes": int(window_spec["frequency_minutes"]),
        "method_step_minutes": int(method_minutes),
        "num_time_steps_used": int(flow.shape[0]),
        "num_nodes": int(flow.shape[1]),
        "num_graph_edges": int((adj > 0).sum()),
        "num_eligible_edges": int(num_eligible_edges),
        "num_edges_scored": int(edges.shape[0]),
        "max_lag_steps": int(max_lag_steps),
        "max_lag_minutes": int(max_lag_steps * method_minutes),
        "min_corr": float(args.min_corr),
        "min_improvement": float(args.min_improvement),
        "residualize_applied": method_meta.get("residualize_applied"),
        "interpolation_method": method_meta.get("interpolation_method"),
        "raw_best_zero_ratio": float(np.mean(best_lags == 0)) if edges.shape[0] else None,
        "raw_nonzero_ratio": float(np.mean(best_lags > 0)) if edges.shape[0] else None,
        "invalid_corr_ratio": float(np.mean(corr_filtered)) if edges.shape[0] else None,
        "corr_filtered_ratio": float(np.mean(corr_filtered)) if edges.shape[0] else None,
        "effective_nan_ratio": float(np.mean(effective_nan)) if edges.shape[0] else None,
        "effective_zero_ratio": float(np.mean(effective_zero)) if edges.shape[0] else None,
        "effective_zero_or_invalid_ratio": float(
            np.mean(effective_zero | effective_nan)
        ) if edges.shape[0] else None,
        "low_improvement_nonzero_ratio": float(
            np.mean(low_improvement_nonzero)
        ) if edges.shape[0] else None,
        "high_conf_nonzero_ratio": float(np.mean(high_conf_nonzero)) if edges.shape[0] else None,
        "median_raw_best_lag_steps": nan_quantile(best_lags, 0.5),
        "median_raw_best_lag_minutes": nan_quantile(best_lags * method_minutes, 0.5),
        "median_effective_lag_steps": nan_quantile(effective_lags, 0.5),
        "median_effective_lag_minutes": nan_quantile(effective_lags * method_minutes, 0.5),
        "p90_effective_lag_minutes": nan_quantile(effective_lags * method_minutes, 0.9),
        "median_zero_lag_corr": nan_quantile(zero_corrs, 0.5),
        "median_best_lag_corr": nan_quantile(best_corrs, 0.5),
        "median_zero_lag_score": nan_quantile(zero_scores, 0.5),
        "median_best_score": nan_quantile(best_scores, 0.5),
        "median_corr_improvement": nan_quantile(improvements, 0.5),
    }


def make_edge_rows(
    dataset: str,
    graph: str,
    window_spec: dict[str, Any],
    method: str,
    method_minutes: int,
    edges: np.ndarray,
    distances_km: np.ndarray | None,
    corr_by_lag: np.ndarray,
    count_by_lag: np.ndarray,
    best_lags: np.ndarray,
    best_corrs: np.ndarray,
    best_scores: np.ndarray,
    zero_corrs: np.ndarray,
    zero_scores: np.ndarray,
    improvements: np.ndarray,
    high_conf_nonzero: np.ndarray,
    effective_lags: np.ndarray,
    corr_filtered: np.ndarray,
    low_improvement_nonzero: np.ndarray,
    include_corr_columns: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (src, dst) in enumerate(edges):
        best_lag = int(best_lags[i])
        effective_lag = float(effective_lags[i])
        effective_lag_steps = effective_lag if np.isfinite(effective_lag) else float("nan")
        effective_lag_minutes = (
            effective_lag * method_minutes if np.isfinite(effective_lag) else float("nan")
        )
        row = {
            "dataset": dataset,
            "graph": graph,
            "window": window_spec["window"],
            "window_label": window_spec["label"],
            "day_offset": window_spec.get("day_offset"),
            "day_in_window": window_spec.get("day_in_window"),
            "weekday": window_spec.get("weekday"),
            "method": method,
            "source_index": int(src),
            "target_index": int(dst),
            "edge_distance_km": float(distances_km[i]) if distances_km is not None else None,
            "best_lag_steps": best_lag,
            "best_lag_minutes": int(best_lag * method_minutes),
            "effective_lag_steps": effective_lag_steps,
            "effective_lag_minutes": effective_lag_minutes,
            "zero_lag_corr": float(zero_corrs[i]) if np.isfinite(zero_corrs[i]) else None,
            "zero_lag_score": float(zero_scores[i]) if np.isfinite(zero_scores[i]) else None,
            "best_lag_corr": float(best_corrs[i]) if np.isfinite(best_corrs[i]) else None,
            "best_lag_score": float(best_scores[i]) if np.isfinite(best_scores[i]) else None,
            "corr_improvement": float(improvements[i]) if np.isfinite(improvements[i]) else None,
            "corr_filtered_delay": bool(corr_filtered[i]),
            "low_improvement_nonzero_delay": bool(low_improvement_nonzero[i]),
            "high_conf_nonzero_delay": bool(high_conf_nonzero[i]),
            "valid_pairs_at_best_lag": int(count_by_lag[best_lag, i]),
        }
        if include_corr_columns:
            for lag in range(corr_by_lag.shape[0]):
                value = corr_by_lag[lag, i]
                row[f"corr_lag_{lag}"] = float(value) if np.isfinite(value) else None
        rows.append(row)
    return rows


def score_graph_window_method(
    dataset: str,
    dataset_dir: Path,
    graph_name: str,
    adj: np.ndarray,
    raw_flow: np.ndarray,
    window_spec: dict[str, Any],
    method: str,
    frequency: int,
    steps_per_day: int,
    lat_lng: tuple[np.ndarray, np.ndarray] | None,
    bins_km: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    base_flow_for_edges = raw_flow
    edges, num_eligible_edges = select_edges(
        adj=adj,
        x=base_flow_for_edges,
        max_edges=args.max_edges,
        min_edge_std=args.min_edge_std,
        seed=args.seed,
    )
    if edges.shape[0] == 0:
        summary = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir),
            "graph": graph_name,
            "window": window_spec["window"],
            "window_label": window_spec["label"],
            "method": method,
            "num_graph_edges": int((adj > 0).sum()),
            "num_eligible_edges": int(num_eligible_edges),
            "num_edges_scored": 0,
            "error": "no eligible nonconstant edges",
        }
        return [], summary, []

    flow, method_minutes, method_meta = prepare_method_flow(
        raw_flow=raw_flow,
        method=method,
        frequency=frequency,
        steps_per_day=steps_per_day,
        args=args,
    )
    max_lag_steps = max(0, int((args.max_lag * frequency) // method_minutes))
    if flow.shape[0] <= max_lag_steps + 2:
        summary = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir),
            "graph": graph_name,
            "window": window_spec["window"],
            "window_label": window_spec["label"],
            "method": method,
            "num_graph_edges": int((adj > 0).sum()),
            "num_eligible_edges": int(num_eligible_edges),
            "num_edges_scored": int(edges.shape[0]),
            "error": "window_too_short_for_requested_lag",
        }
        return [], summary, []

    corr_by_lag, count_by_lag, best_lags, best_corrs, best_scores, improvements, score_mode = score_delay(
        flow=flow,
        edges=edges,
        method=method,
        max_lag_steps=max_lag_steps,
        chunk_size=args.score_chunk_size,
    )
    zero_corrs = corr_by_lag[0]
    zero_scores = np.abs(zero_corrs) if score_mode == "absolute_corr" else zero_corrs
    corr_filtered = best_scores < args.min_corr
    low_improvement_nonzero = (
        (best_lags > 0)
        & (best_scores >= args.min_corr)
        & (improvements < args.min_improvement)
    )
    high_conf_nonzero = (
        (best_lags > 0)
        & (best_scores >= args.min_corr)
        & (improvements >= args.min_improvement)
    )
    # Effective lag semantics:
    # - corr-filtered estimates are not valid delays and stay NaN.
    # - when corr passes, keep the argmax/best-score lag even if the
    #   improvement over zero lag is below the strict high-confidence cutoff.
    effective_lags = best_lags.astype(np.float64)
    effective_lags[corr_filtered] = np.nan
    distances_km = edge_distances_km(lat_lng, edges)

    summary = summarize_scores(
        dataset=dataset,
        dataset_dir=dataset_dir,
        graph=graph_name,
        window_spec=window_spec,
        method=method,
        method_meta=method_meta,
        flow=flow,
        method_minutes=method_minutes,
        adj=adj,
        edges=edges,
        num_eligible_edges=num_eligible_edges,
        max_lag_steps=max_lag_steps,
        score_mode=score_mode,
        best_lags=best_lags,
        best_corrs=best_corrs,
        best_scores=best_scores,
        zero_corrs=zero_corrs,
        zero_scores=zero_scores,
        improvements=improvements,
        high_conf_nonzero=high_conf_nonzero,
        effective_lags=effective_lags,
        args=args,
    )
    distance_rows = summarize_distance_bins(
        distances_km=distances_km,
        best_lags=best_lags,
        effective_lags=effective_lags,
        improvements=improvements,
        best_scores=best_scores,
        bins=bins_km,
        min_corr=args.min_corr,
        min_improvement=args.min_improvement,
        high_conf_nonzero=high_conf_nonzero,
    )
    for row in distance_rows:
        row.update(
            {
                "dataset": dataset,
                "graph": graph_name,
                "window": window_spec["window"],
                "window_label": window_spec["label"],
                "day_offset": window_spec.get("day_offset"),
                "day_in_window": window_spec.get("day_in_window"),
                "weekday": window_spec.get("weekday"),
                "method": method,
                "method_step_minutes": int(method_minutes),
            }
        )
    edge_rows = make_edge_rows(
        dataset=dataset,
        graph=graph_name,
        window_spec=window_spec,
        method=method,
        method_minutes=method_minutes,
        edges=edges,
        distances_km=distances_km,
        corr_by_lag=corr_by_lag,
        count_by_lag=count_by_lag,
        best_lags=best_lags,
        best_corrs=best_corrs,
        best_scores=best_scores,
        zero_corrs=zero_corrs,
        zero_scores=zero_scores,
        improvements=improvements,
        high_conf_nonzero=high_conf_nonzero,
        effective_lags=effective_lags,
        corr_filtered=corr_filtered,
        low_improvement_nonzero=low_improvement_nonzero,
        include_corr_columns=args.include_corr_columns,
    )
    return edge_rows, summary, distance_rows


def audit_one_dataset(
    name: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    output_root: Path,
    bins_km: np.ndarray,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    desc = load_json(dataset_dir / "desc.json")
    frequency = int(desc.get("frequency (minutes)", -1))
    if frequency != 5 and not args.allow_non_5min:
        raise ValueError(
            f"{name} frequency is {frequency} minutes in {dataset_dir / 'desc.json'}, "
            "but this audit is restricted to 5-minute versions."
        )
    steps_per_day = max(1, int(round(1440 / max(1, frequency))))
    graphs = load_graph_variants(dataset_dir, args.graph_variants)
    lat_lng = load_lat_lng(dataset_dir, int(desc["num_nodes"]))
    window_specs = get_window_specs(desc, dataset_dir, args, frequency, steps_per_day)

    dataset_out = output_root / name
    edge_rows_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    distance_rows_all: list[dict[str, Any]] = []

    for window_spec in window_specs:
        if int(window_spec.get("num_steps", 1)) == 0 and "time_indices" not in window_spec:
            continue
        raw_flow = load_window_flow(dataset_dir, desc, window_spec, args)
        print(
            f"[multi-delay] {name} window={window_spec['window']} label={window_spec['label']} "
            f"steps={raw_flow.shape[0]}",
            flush=True,
        )
        for graph_name, adj in graphs.items():
            for method in args.methods:
                print(
                    f"[multi-delay] {name}/{graph_name}/{window_spec['label']}/{method}",
                    flush=True,
                )
                edge_rows, summary, distance_rows = score_graph_window_method(
                    dataset=name,
                    dataset_dir=dataset_dir,
                    graph_name=graph_name,
                    adj=adj,
                    raw_flow=raw_flow,
                    window_spec=window_spec,
                    method=method,
                    frequency=frequency,
                    steps_per_day=steps_per_day,
                    lat_lng=lat_lng,
                    bins_km=bins_km,
                    args=args,
                )
                edge_rows_all.extend(edge_rows)
                if summary is not None:
                    summary_rows.append(summary)
                distance_rows_all.extend(distance_rows)

    write_csv(dataset_out / "edge_delay_scores.csv", edge_rows_all)
    write_csv(dataset_out / "summary.csv", summary_rows)
    write_csv(dataset_out / "distance_bin_summary.csv", distance_rows_all)
    dataset_summary = {
        "dataset": name,
        "dataset_dir": str(dataset_dir),
        "desc": desc,
        "methods": args.methods,
        "windows": args.windows,
        "summary_rows": summary_rows,
        "distance_bin_rows": distance_rows_all,
    }
    write_json(dataset_out / "summary.json", dataset_summary)
    return dataset_summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    bins_km = parse_bins(args.distance_bins_km)

    if args.dataset:
        dataset_specs = parse_dataset_specs(args.dataset)
        skipped: list[dict[str, str]] = []
        if args.require_all:
            for name, path in dataset_specs.items():
                if not path.exists():
                    raise FileNotFoundError(f"{name} dataset path does not exist: {path}")
    else:
        dataset_specs, skipped = discover_default_datasets(require_all=args.require_all)

    results = []
    errors = []
    for name, path in dataset_specs.items():
        if not path.exists():
            item = {"dataset": name, "reason": "path_missing", "checked": str(path)}
            skipped.append(item)
            if args.require_all:
                raise FileNotFoundError(item)
            continue
        print(f"[multi-delay] dataset={name} path={path}", flush=True)
        try:
            results.append(audit_one_dataset(name, path, args, output_root, bins_km))
        except Exception as exc:
            errors.append({"dataset": name, "path": str(path), "error": repr(exc)})
            if args.require_all:
                raise

    combined_summary_rows: list[dict[str, Any]] = []
    combined_distance_rows: list[dict[str, Any]] = []
    for result in results:
        combined_summary_rows.extend(result["summary_rows"])
        combined_distance_rows.extend(result["distance_bin_rows"])
    write_csv(output_root / "all_summary.csv", combined_summary_rows)
    write_csv(output_root / "all_distance_bin_summary.csv", combined_distance_rows)
    write_json(
        output_root / "run_summary.json",
        {
            "args": vars(args),
            "default_dataset_candidates": DEFAULT_DATASET_CANDIDATES,
            "num_datasets_completed": len(results),
            "completed_datasets": [result["dataset"] for result in results],
            "skipped": skipped,
            "errors": errors,
            "output_dir": str(output_root),
        },
    )
    print(
        json.dumps(
            {
                "completed_datasets": [result["dataset"] for result in results],
                "skipped": skipped,
                "errors": errors,
                "output_dir": str(output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

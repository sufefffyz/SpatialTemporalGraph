#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import sparse

    HAS_SCIPY = True
except ModuleNotFoundError:
    sparse = None
    HAS_SCIPY = False


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"


@dataclass
class RunSpec:
    name: str
    result_dir: Path


def parse_key_path(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path).expanduser()
    return None, Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run decoupled frequency / distribution diagnostics on BasicTS saved test predictions."
    )
    parser.add_argument("--dataset-name", default="SD_5min_full")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run as name=/path/to/ckpt_or_test_results. Can be repeated.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Root to recursively scan for BasicTS test_results directories.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only keep discovered result paths containing this substring. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Drop discovered result paths containing this substring. Can be repeated.",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Cap discovered runs. 0 means no cap.")
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--target-channel", type=int, default=0, help="Target channel in dataset data.dat for seasonal scaling.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--moving-window", type=int, default=12)
    parser.add_argument(
        "--decomp-methods",
        nargs="+",
        default=["moving_average"],
        choices=["moving_average", "fft_lowpass"],
        help="Low/high decomposition methods to evaluate. Use both to compare moving-average and frequency-domain splits.",
    )
    parser.add_argument(
        "--fft-cutoff-period",
        type=float,
        default=None,
        help="FFT low-pass cutoff in time steps. Low frequency keeps periods >= this value. Defaults to --moving-window.",
    )
    parser.add_argument("--peak-q", type=float, default=0.90)
    parser.add_argument("--worst-pct", type=float, default=0.10, help="Top fraction of test windows for worst-window MAE.")
    parser.add_argument(
        "--alignment-max-shift",
        type=int,
        default=3,
        help="Maximum sample shift for time-alignment diagnostics. 0 disables shifted-MAE curves.",
    )
    parser.add_argument(
        "--alignment-time-windows",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Temporal tolerance windows for relaxed peak hit metrics.",
    )
    parser.add_argument(
        "--alignment-hop-ks",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Spatial hop tolerances for relaxed peak hit metrics. k>0 requires --adj-path.",
    )
    parser.add_argument(
        "--alignment-directed",
        action="store_true",
        help="Use directed adjacency for relaxed spatial hit metrics. Defaults to symmetrized adjacency.",
    )
    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Only compute time/space alignment and conditional ShiftGain metrics; skip standard and decomposition metrics.",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Only compute patch-level joint ST shift metrics; requires --joint-st-patch-len > 0.",
    )
    parser.add_argument(
        "--condition-high-q",
        type=float,
        default=0.75,
        help="Per-node/horizon quantile threshold for the high-volume conditional ShiftGain bin.",
    )
    parser.add_argument(
        "--condition-ramp-q",
        type=float,
        default=0.90,
        help="Per-node/horizon quantile threshold for the ramp conditional ShiftGain bin.",
    )
    parser.add_argument(
        "--joint-st-conditions",
        nargs="+",
        default=["high_volume", "peak", "ramp"],
        choices=["all", "normal", "high_volume", "peak", "ramp"],
        help="Traffic regimes for joint spatiotemporal STShiftGain. Defaults to the event-like regimes to keep the oracle-style spatial search interpretable.",
    )
    parser.add_argument(
        "--joint-st-workers",
        type=int,
        default=1,
        help="Number of worker threads for joint spatiotemporal STShiftGain over hop/time-shift tasks.",
    )
    parser.add_argument(
        "--joint-st-patch-len",
        type=int,
        default=0,
        help="If positive, compute patch-level joint ST shift metrics over this many forecast steps.",
    )
    parser.add_argument(
        "--joint-st-patch-conditions",
        nargs="+",
        default=["high_volume", "peak", "ramp"],
        choices=["all", "normal", "high_volume", "peak", "ramp"],
        help="Traffic regimes for patch-level joint ST shift metrics.",
    )
    parser.add_argument(
        "--joint-st-patch-chunk-size",
        type=int,
        default=32768,
        help="Chunk size for patch-level joint ST shift metric computation.",
    )
    parser.add_argument(
        "--joint-st-patch-topk",
        type=int,
        default=0,
        help="If positive, keep only the top-k strongest non-self graph neighbors for patch-level joint ST shift metrics.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=None,
        help="Seasonal period for MASE/RMSSE scaling. Defaults to one day inferred from dataset frequency.",
    )
    parser.add_argument(
        "--null-val",
        default=None,
        help="Target null value for standard metrics. Defaults to desc.json regular_settings.NULL_VAL; use 'none' to disable.",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 13)), help="1-based horizons.")
    parser.add_argument("--adj-path", type=Path, default=None, help="Optional adjacency pickle/npy for spatial residual diagnostics.")
    parser.add_argument("--max-edge-pairs", type=int, default=20000)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    override = os.environ.get("BASICTS_DATASETS_ROOT")
    if override:
        return (Path(override).expanduser() / args.dataset_name).resolve()
    return (BASICTS_ROOT / "datasets" / args.dataset_name).resolve()


def load_dataset_desc(args: argparse.Namespace) -> dict | None:
    desc_path = dataset_dir(args) / "desc.json"
    if not desc_path.exists():
        return None
    return load_json(desc_path)


def load_dataset_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_nodes is not None and args.output_len is not None:
        return int(args.num_nodes), int(args.output_len)
    desc = load_dataset_desc(args)
    if desc is None:
        missing = []
        if args.num_nodes is None:
            missing.append("--num-nodes")
        if args.output_len is None:
            missing.append("--output-len")
        raise FileNotFoundError(
            f"Dataset desc not found at {dataset_dir(args) / 'desc.json'}; provide {' and '.join(missing)}."
        )
    num_nodes = int(args.num_nodes or desc["num_nodes"])
    output_len = int(args.output_len or desc["regular_settings"]["OUTPUT_LEN"])
    return num_nodes, output_len


def normalize_result_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "test_results":
        return path
    return path / "test_results"


def default_run_name(result_dir: Path) -> str:
    parts = result_dir.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx + 1 < len(parts):
            name = parts[idx + 1]
            aliases = {
                "STGCNChebGraphConv": "STGCN",
            }
            return aliases.get(name, name)
    run_dir = result_dir.parent
    variant_dir = run_dir.parent
    if len(run_dir.name) >= 8 and all(ch in "0123456789abcdef" for ch in run_dir.name.lower()):
        return f"{variant_dir.name}__{run_dir.name[:8]}"
    return run_dir.name


def discover_runs(args: argparse.Namespace) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen: set[Path] = set()

    for item in args.run:
        name, path = parse_key_path(item)
        result_dir = normalize_result_dir(path)
        if result_dir in seen:
            continue
        seen.add(result_dir)
        runs.append(RunSpec(name=name or default_run_name(result_dir), result_dir=result_dir))

    for root in args.search_root:
        root = root.expanduser()
        if not root.exists():
            log(f"[warn] search root missing, skip: {root}")
            continue
        for pred_path in root.rglob("test_results/predictions.npy"):
            result_dir = pred_path.parent.resolve()
            path_str = str(result_dir)
            if args.include and not all(token in path_str for token in args.include):
                continue
            if args.exclude and any(token in path_str for token in args.exclude):
                continue
            if not (result_dir / "targets.npy").exists():
                continue
            if result_dir in seen:
                continue
            seen.add(result_dir)
            runs.append(RunSpec(name=default_run_name(result_dir), result_dir=result_dir))

    runs.sort(key=lambda run: str(run.result_dir))
    if args.max_runs and len(runs) > args.max_runs:
        runs = runs[: args.max_runs]
    return runs


def try_np_load(path: Path) -> np.ndarray | None:
    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        return None


def infer_raw_memmap(path: Path, dtype: str, output_len: int, num_nodes: int, target_dim: int) -> np.memmap:
    itemsize = np.dtype(dtype).itemsize
    denom = output_len * num_nodes * target_dim * itemsize
    size = path.stat().st_size
    if size % denom != 0:
        raise ValueError(
            f"Cannot infer BasicTS raw memmap shape for {path}: file bytes={size}, "
            f"expected divisible by output_len*num_nodes*target_dim*itemsize={denom}."
        )
    num_samples = size // denom
    return np.memmap(path, dtype=dtype, mode="r", shape=(num_samples, output_len, num_nodes, target_dim))


def load_saved_array(path: Path, dtype: str, output_len: int, num_nodes: int, target_dim: int) -> np.ndarray:
    arr = try_np_load(path)
    if arr is None:
        arr = infer_raw_memmap(path, dtype, output_len, num_nodes, target_dim)
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected saved array with shape [samples, horizon, nodes], got {arr.shape} at {path}")
    if arr.shape[1] != output_len or arr.shape[2] != num_nodes:
        raise ValueError(
            f"Shape mismatch at {path}: got {arr.shape}, expected [*, {output_len}, {num_nodes}]"
        )
    return np.asarray(arr, dtype=np.float32)


def finite_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = np.isfinite(pred) & np.isfinite(target)
    return pred, target, valid


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - target[valid])))


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return float("nan")
    err = pred[valid] - target[valid]
    return float(np.sqrt(np.mean(err * err)))


def masked_wape(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return float("nan")
    denom = float(np.sum(np.abs(target[valid])))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(pred[valid] - target[valid])) / denom)


def parse_null_value(value: str | int | float | None, desc: dict | None) -> float | None:
    if value is None:
        if desc is None:
            return None
        value = desc.get("regular_settings", {}).get("NULL_VAL")
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null", "off", "disable", "disabled"}:
            return None
        if lowered in {"nan", "np.nan"}:
            return float("nan")
    return float(value)


def valid_target_mask(target: np.ndarray, null_val: float | None) -> np.ndarray:
    valid = np.isfinite(target)
    if null_val is None:
        return valid
    if isinstance(null_val, float) and math.isnan(null_val):
        return valid
    return valid & ~np.isclose(target, null_val, atol=5e-5, rtol=0.0)


def infer_seasonal_period(desc: dict | None, explicit_period: int | None) -> int:
    if explicit_period is not None:
        return int(explicit_period)
    if desc is None:
        return 0
    frequency = desc.get("frequency (minutes)")
    try:
        frequency = float(frequency)
    except (TypeError, ValueError):
        return 0
    if frequency <= 0:
        return 0
    return max(1, int(round(1440.0 / frequency)))


def load_seasonal_scales(
    args: argparse.Namespace,
    desc: dict | None,
    num_nodes: int,
    seasonal_period: int,
    null_val: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if desc is None or seasonal_period <= 0:
        return None, None
    data_path = dataset_dir(args) / "data.dat"
    if not data_path.exists():
        log(f"[warn] data.dat missing, MASE/RMSSE disabled: {data_path}")
        return None, None
    shape = tuple(int(dim) for dim in desc["shape"])
    if len(shape) != 3 or shape[1] != num_nodes:
        log(f"[warn] dataset shape {shape} does not match num_nodes={num_nodes}; MASE/RMSSE disabled")
        return None, None
    if not (0 <= args.target_channel < shape[2]):
        raise ValueError(f"--target-channel {args.target_channel} outside dataset feature dimension {shape[2]}")

    ratios = desc.get("regular_settings", {}).get("TRAIN_VAL_TEST_RATIO", [0.6, 0.2, 0.2])
    total_len = shape[0]
    valid_len = int(total_len * float(ratios[1]))
    test_len = int(total_len * float(ratios[2]))
    train_len = total_len - valid_len - test_len
    if train_len <= seasonal_period:
        log(f"[warn] train_len={train_len} <= seasonal_period={seasonal_period}; MASE/RMSSE disabled")
        return None, None

    data = np.memmap(data_path, dtype="float32", mode="r", shape=shape)
    series = np.asarray(data[:train_len, :, args.target_channel], dtype=np.float32)
    current = series[seasonal_period:]
    previous = series[:-seasonal_period]
    valid = np.isfinite(current) & np.isfinite(previous)
    if null_val is not None and not (isinstance(null_val, float) and math.isnan(null_val)):
        valid &= ~np.isclose(current, null_val, atol=5e-5, rtol=0.0)
        valid &= ~np.isclose(previous, null_val, atol=5e-5, rtol=0.0)
    diff = current - previous
    with np.errstate(invalid="ignore"):
        scale_abs = np.nanmean(np.where(valid, np.abs(diff), np.nan), axis=0)
        scale_sq = np.nanmean(np.where(valid, diff * diff, np.nan), axis=0)
    scale_abs = np.where((scale_abs > 1e-8) & np.isfinite(scale_abs), scale_abs, np.nan).astype(np.float32)
    scale_sq = np.where((scale_sq > 1e-8) & np.isfinite(scale_sq), scale_sq, np.nan).astype(np.float32)
    return scale_abs, scale_sq


def masked_mase(
    pred: np.ndarray,
    target: np.ndarray,
    scale_abs: np.ndarray | None,
    mask: np.ndarray | None = None,
) -> float:
    if scale_abs is None:
        return float("nan")
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    scale = np.asarray(scale_abs, dtype=np.float32).reshape(1, -1)
    valid &= np.isfinite(scale) & (scale > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred - target)[valid] / np.broadcast_to(scale, pred.shape)[valid]))


def masked_rmsse(
    pred: np.ndarray,
    target: np.ndarray,
    scale_sq: np.ndarray | None,
    mask: np.ndarray | None = None,
) -> float:
    if scale_sq is None:
        return float("nan")
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    scale = np.asarray(scale_sq, dtype=np.float32).reshape(1, -1)
    valid &= np.isfinite(scale) & (scale > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(((pred - target) * (pred - target))[valid] / np.broadcast_to(scale, pred.shape)[valid])))


def per_window_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    abs_err = np.where(valid, np.abs(pred - target), np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(abs_err, axis=1)


def worst_pct_mae(pred: np.ndarray, target: np.ndarray, pct: float, mask: np.ndarray | None = None) -> float:
    losses = per_window_mae(pred, target, mask)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return float("nan")
    pct = min(max(float(pct), 0.0), 1.0)
    if pct <= 0:
        return float(np.max(losses))
    k = max(1, int(math.ceil(losses.size * pct)))
    return float(np.mean(np.sort(losses)[-k:]))


def peak_detection_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    q: float,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(np.where(valid, target, np.nan), q, axis=0).astype(np.float32)
    threshold_grid = thresholds.reshape(1, -1)
    threshold_valid = np.isfinite(threshold_grid)
    true_peak = valid & threshold_valid & (target >= threshold_grid)
    pred_peak = valid & threshold_valid & (pred >= threshold_grid)
    tp = int(np.sum(true_peak & pred_peak))
    pred_count = int(np.sum(pred_peak))
    true_count = int(np.sum(true_peak))
    precision = tp / pred_count if pred_count else float("nan")
    recall = tp / true_count if true_count else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    return {
        "peak_precision": float(precision),
        "peak_recall": float(recall),
        "peak_F1": float(f1),
        "peak_true_count": true_count,
        "peak_pred_count": pred_count,
        "peak_tp_count": tp,
    }


def peak_masks(
    pred: np.ndarray,
    target: np.ndarray,
    q: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(np.where(valid, target, np.nan), q, axis=0).astype(np.float32)
    threshold_grid = thresholds.reshape(1, -1)
    threshold_valid = np.isfinite(threshold_grid)
    true_peak = valid & threshold_valid & (target >= threshold_grid)
    pred_peak = valid & threshold_valid & (pred >= threshold_grid)
    return true_peak, pred_peak, valid & threshold_valid


def shifted_mae_curve(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
) -> list[dict[str, float | int]]:
    max_shift = int(max(0, max_shift))
    rows: list[dict[str, float | int]] = []
    pred, target, valid = finite_pair(pred, target)
    valid &= mask
    num_steps = pred.shape[0]
    for delta in range(-max_shift, max_shift + 1):
        if delta < 0:
            pred_slice = pred[: num_steps + delta]
            target_slice = target[-delta:]
            valid_slice = valid[-delta:] & np.isfinite(pred_slice)
        elif delta > 0:
            pred_slice = pred[delta:]
            target_slice = target[: num_steps - delta]
            valid_slice = valid[: num_steps - delta] & np.isfinite(pred_slice)
        else:
            pred_slice = pred
            target_slice = target
            valid_slice = valid
        if not np.any(valid_slice):
            mae = float("nan")
            count = 0
        else:
            mae = float(np.mean(np.abs(pred_slice[valid_slice] - target_slice[valid_slice])))
            count = int(np.sum(valid_slice))
        rows.append({"time_shift_delta": delta, "shift_MAE": mae, "valid_count": count})
    return rows


def best_shift_summary(curve_rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    zero_mae = next((float(row["shift_MAE"]) for row in curve_rows if int(row["time_shift_delta"]) == 0), float("nan"))
    finite = [row for row in curve_rows if np.isfinite(float(row["shift_MAE"]))]
    if not finite:
        return {
            "best_time_shift_delta": 0,
            "best_shift_MAE": float("nan"),
            "zero_shift_MAE": zero_mae,
            "shift_gain": float("nan"),
        }
    best = min(finite, key=lambda row: (float(row["shift_MAE"]), abs(int(row["time_shift_delta"]))))
    best_mae = float(best["shift_MAE"])
    return {
        "best_time_shift_delta": int(best["time_shift_delta"]),
        "best_shift_MAE": best_mae,
        "zero_shift_MAE": zero_mae,
        "shift_gain": (zero_mae - best_mae) / zero_mae if zero_mae and np.isfinite(zero_mae) else float("nan"),
    }


def quantile_condition_mask(target: np.ndarray, base_mask: np.ndarray, q: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(np.where(base_mask, target, np.nan), q, axis=0).astype(np.float32)
    threshold_grid = thresholds.reshape(1, -1)
    return base_mask & np.isfinite(threshold_grid) & (target >= threshold_grid)


def ramp_condition_mask(target: np.ndarray, base_mask: np.ndarray, q: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    ramp = np.full_like(target, np.nan, dtype=np.float32)
    ramp_valid = np.zeros_like(base_mask, dtype=bool)
    if target.shape[0] <= 1:
        return ramp_valid
    ramp[1:] = np.abs(target[1:] - target[:-1])
    ramp_valid[1:] = base_mask[1:] & base_mask[:-1] & np.isfinite(ramp[1:])
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(np.where(ramp_valid, ramp, np.nan), q, axis=0).astype(np.float32)
    threshold_grid = thresholds.reshape(1, -1)
    return ramp_valid & np.isfinite(threshold_grid) & (ramp >= threshold_grid)


def conditional_shift_masks(
    target: np.ndarray,
    base_mask: np.ndarray,
    peak_q: float,
    high_q: float,
    ramp_q: float,
) -> dict[str, np.ndarray]:
    all_mask = np.asarray(base_mask, dtype=bool)
    peak = quantile_condition_mask(target, all_mask, peak_q)
    high_volume = quantile_condition_mask(target, all_mask, high_q)
    ramp = ramp_condition_mask(target, all_mask, ramp_q)
    normal = all_mask & ~peak & ~ramp
    return {
        "all": all_mask,
        "peak": peak,
        "high_volume": high_volume,
        "ramp": ramp,
        "normal": normal,
    }


def conditional_shift_rows(
    system: str,
    horizon: int,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
    peak_q: float,
    high_q: float,
    ramp_q: float,
) -> tuple[list[dict], list[dict]]:
    metric_rows: list[dict] = []
    curve_rows: list[dict] = []
    for condition, condition_mask in conditional_shift_masks(target, mask, peak_q, high_q, ramp_q).items():
        curve = shifted_mae_curve(pred, target, condition_mask, max_shift)
        summary = best_shift_summary(curve)
        zero_count = next((int(row["valid_count"]) for row in curve if int(row["time_shift_delta"]) == 0), 0)
        metric_rows.append(
            {
                "system": system,
                "horizon": horizon,
                "condition": condition,
                "condition_valid_count": zero_count,
                **summary,
            }
        )
        for row in curve:
            curve_rows.append(
                {
                    "system": system,
                    "horizon": horizon,
                    "condition": condition,
                    **row,
                }
            )
    return metric_rows, curve_rows


def build_reach_lists(adj: np.ndarray | None, hop_ks: list[int], directed: bool) -> dict[int, list[np.ndarray]]:
    if adj is None:
        return {}
    max_k = max([0, *hop_ks])
    work = np.asarray(adj)
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError(f"Adjacency must be square, got {work.shape}")
    if max_k <= 0:
        return {0: [np.asarray([idx], dtype=np.int64) for idx in range(work.shape[0])]}
    base = work > 0
    if not directed:
        base = base | base.T
    num_nodes = base.shape[0]
    neighbors = [set(np.flatnonzero(base[idx]).tolist()) | {idx} for idx in range(num_nodes)]
    reach_sets = [{idx} for idx in range(num_nodes)]
    frontier_sets = [{idx} for idx in range(num_nodes)]
    reach_by_k: dict[int, list[np.ndarray]] = {
        0: [np.asarray([idx], dtype=np.int64) for idx in range(num_nodes)]
    }
    for hop in range(1, max_k + 1):
        next_frontiers: list[set[int]] = []
        for idx in range(num_nodes):
            next_nodes: set[int] = set()
            for node in frontier_sets[idx]:
                next_nodes.update(neighbors[node])
            next_nodes.difference_update(reach_sets[idx])
            reach_sets[idx].update(next_nodes)
            next_frontiers.append(next_nodes)
        frontier_sets = next_frontiers
        reach_by_k[hop] = [np.asarray(sorted(items), dtype=np.int64) for items in reach_sets]
    return {k: reach_by_k[k] for k in sorted(set(hop_ks)) if k in reach_by_k}


def build_reach_indices(adj: np.ndarray | None, hop_ks: list[int], directed: bool) -> dict[int, object]:
    reach_by_k = build_reach_lists(adj, hop_ks, directed)
    if not reach_by_k:
        return {}
    if HAS_SCIPY:
        reach_sparse: dict[int, object] = {}
        for hop, indices_by_node in reach_by_k.items():
            num_nodes = len(indices_by_node)
            rows = []
            cols = []
            for node, indices in enumerate(indices_by_node):
                rows.extend([node] * len(indices))
                cols.extend(indices.tolist())
            data = np.ones(len(rows), dtype=np.uint8)
            reach_sparse[hop] = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        return {k: reach_sparse[k] for k in sorted(set(hop_ks)) if k in reach_sparse}
    return {k: reach_by_k[k] for k in sorted(set(hop_ks)) if k in reach_by_k}


def pad_reach_lists(reach_by_k: dict[int, list[np.ndarray]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    padded_by_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for hop_k, indices_by_node in reach_by_k.items():
        max_degree = max((len(indices) for indices in indices_by_node), default=0)
        if max_degree == 0:
            continue
        padded = np.zeros((len(indices_by_node), max_degree), dtype=np.int64)
        valid = np.zeros((len(indices_by_node), max_degree), dtype=bool)
        for node, indices in enumerate(indices_by_node):
            count = len(indices)
            if count:
                padded[node, :count] = indices
                valid[node, :count] = True
        padded_by_k[hop_k] = (padded, valid)
    return padded_by_k


def limit_reach_lists_by_topk(
    adj: np.ndarray | None,
    reach_by_k: dict[int, list[np.ndarray]],
    topk: int,
    directed: bool,
) -> dict[int, list[np.ndarray]]:
    topk = int(topk)
    if adj is None or topk <= 0:
        return reach_by_k
    weights = np.asarray(adj, dtype=np.float32)
    if not directed:
        weights = np.maximum(weights, weights.T)
    limited: dict[int, list[np.ndarray]] = {}
    for hop_k, reach_lists in reach_by_k.items():
        if hop_k <= 0:
            limited[hop_k] = reach_lists
            continue
        limited_lists: list[np.ndarray] = []
        for node, indices in enumerate(reach_lists):
            indices = np.asarray(indices, dtype=np.int64)
            non_self = indices[indices != node]
            if non_self.size:
                scores = weights[node, non_self]
                order = np.lexsort((non_self, -scores))
                non_self = non_self[order[:topk]]
            limited_lists.append(np.asarray([node, *non_self.tolist()], dtype=np.int64))
        limited[hop_k] = limited_lists
    return limited


def shifted_slices(
    pred: np.ndarray,
    target: np.ndarray,
    base_mask: np.ndarray,
    condition_mask: np.ndarray,
    delta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_steps = pred.shape[0]
    if delta < 0:
        return pred[: num_steps + delta], target[-delta:], base_mask[-delta:] & condition_mask[-delta:]
    if delta > 0:
        return pred[delta:], target[: num_steps - delta], base_mask[: num_steps - delta] & condition_mask[: num_steps - delta]
    return pred, target, base_mask & condition_mask


def spatial_min_abs_error(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    reach_indices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(target)
    if reach_indices is None:
        pair_valid = valid & np.isfinite(pred)
        return np.where(pair_valid, np.abs(pred - target), np.nan).astype(np.float32)

    if isinstance(reach_indices, tuple):
        padded, index_valid = reach_indices
        err = np.full(target.shape, np.nan, dtype=np.float32)
        chunk_size = 32
        for start in range(0, target.shape[1], chunk_size):
            end = min(start + chunk_size, target.shape[1])
            node_indices = padded[start:end]
            node_index_valid = index_valid[start:end]
            candidate_pred = pred[:, node_indices]
            candidate_finite = np.isfinite(candidate_pred) & node_index_valid.reshape(1, end - start, -1)
            candidate_err = np.abs(candidate_pred - target[:, start:end, None])
            candidate_err = np.where(candidate_finite, candidate_err, np.inf)
            best = np.min(candidate_err, axis=2)
            keep = valid[:, start:end] & np.isfinite(best)
            err[:, start:end] = np.where(keep, best, np.nan)
        return err

    pred_finite = np.isfinite(pred)
    err = np.full(target.shape, np.nan, dtype=np.float32)
    for node, indices in enumerate(reach_indices):
        node_valid = valid[:, node]
        if not np.any(node_valid):
            continue
        candidate_pred = pred[:, indices]
        candidate_finite = pred_finite[:, indices]
        candidate_err = np.abs(candidate_pred - target[:, node : node + 1])
        candidate_err = np.where(candidate_finite, candidate_err, np.inf)
        best = np.min(candidate_err, axis=1)
        keep = node_valid & np.isfinite(best)
        err[keep, node] = best[keep]
    return err


def spatial_min_abs_error_sparse(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    reach_indices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(target)
    if reach_indices is None:
        pair_valid = valid & np.isfinite(pred)
        return np.where(pair_valid, np.abs(pred - target), np.nan).astype(np.float32)
    if isinstance(reach_indices, tuple):
        padded, index_valid = reach_indices
        reach_list = [padded[node, index_valid[node]] for node in range(padded.shape[0])]
    else:
        reach_list = reach_indices

    err = np.full(target.shape, np.nan, dtype=np.float32)
    pred_finite = np.isfinite(pred)
    for node, indices in enumerate(reach_list):
        rows = np.flatnonzero(valid[:, node])
        if rows.size == 0 or indices.size == 0:
            continue
        candidate_pred = pred[np.ix_(rows, indices)]
        candidate_finite = pred_finite[np.ix_(rows, indices)]
        candidate_err = np.abs(candidate_pred - target[rows, node : node + 1])
        candidate_err = np.where(candidate_finite, candidate_err, np.inf)
        best = np.min(candidate_err, axis=1)
        keep = np.isfinite(best)
        if np.any(keep):
            err[rows[keep], node] = best[keep]
    return err


def finite_mean_and_count(values: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(values)
    count = int(np.sum(finite))
    if count == 0:
        return float("nan"), 0
    return float(np.mean(values[finite])), count


def spatial_condition_min_abs_error_stats(
    pred: np.ndarray,
    target: np.ndarray,
    condition_valid: np.ndarray,
    reach_indices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, int]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(condition_valid, dtype=bool) & np.isfinite(target)
    if reach_indices is None:
        pair_valid = valid & np.isfinite(pred)
        if not np.any(pair_valid):
            return float("nan"), 0
        err = np.abs(pred[pair_valid] - target[pair_valid])
        return float(np.mean(err)), int(err.size)

    if not isinstance(reach_indices, tuple):
        st_error = spatial_min_abs_error_sparse(pred, target, valid, reach_indices)
        return finite_mean_and_count(np.where(valid, st_error, np.nan))

    time_idx, node_idx = np.nonzero(valid)
    if time_idx.size == 0:
        return float("nan"), 0

    padded, index_valid = reach_indices
    total = 0.0
    count = 0
    chunk_size = 65536
    for start in range(0, time_idx.size, chunk_size):
        end = min(start + chunk_size, time_idx.size)
        rows = time_idx[start:end]
        nodes = node_idx[start:end]
        neighbor_idx = padded[nodes]
        neighbor_valid = index_valid[nodes]
        candidate_pred = pred[rows[:, None], neighbor_idx]
        candidate_finite = np.isfinite(candidate_pred) & neighbor_valid
        candidate_err = np.abs(candidate_pred - target[rows, nodes][:, None])
        candidate_err = np.where(candidate_finite, candidate_err, np.inf)
        best = np.min(candidate_err, axis=1)
        keep = np.isfinite(best)
        if np.any(keep):
            total += float(np.sum(best[keep], dtype=np.float64))
            count += int(np.sum(keep))
    if count == 0:
        return float("nan"), 0
    return total / float(count), count


def best_st_shift_summary(
    curve_rows: list[dict[str, float | int]],
    exact_zero_mae: float,
) -> dict[str, float | int]:
    zero_st_mae = next((float(row["st_shift_MAE"]) for row in curve_rows if int(row["time_shift_delta"]) == 0), float("nan"))
    finite = [row for row in curve_rows if np.isfinite(float(row["st_shift_MAE"]))]
    if not finite:
        return {
            "best_time_shift_delta": 0,
            "best_ST_MAE": float("nan"),
            "zero_shift_ST_MAE": zero_st_mae,
            "exact_zero_MAE": exact_zero_mae,
            "st_shift_gain": float("nan"),
            "spatial_gain_at_zero": float("nan"),
        }
    best = min(finite, key=lambda row: (float(row["st_shift_MAE"]), abs(int(row["time_shift_delta"]))))
    best_mae = float(best["st_shift_MAE"])
    return {
        "best_time_shift_delta": int(best["time_shift_delta"]),
        "best_ST_MAE": best_mae,
        "zero_shift_ST_MAE": zero_st_mae,
        "exact_zero_MAE": exact_zero_mae,
        "st_shift_gain": exact_zero_mae - best_mae if np.isfinite(exact_zero_mae) and np.isfinite(best_mae) else float("nan"),
        "spatial_gain_at_zero": exact_zero_mae - zero_st_mae if np.isfinite(exact_zero_mae) and np.isfinite(zero_st_mae) else float("nan"),
    }


def joint_st_shift_rows(
    system: str,
    horizon: int,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
    hop_ks: list[int],
    reach_lists_by_k: dict[int, list[np.ndarray] | tuple[np.ndarray, np.ndarray]],
    peak_q: float,
    high_q: float,
    ramp_q: float,
    joint_conditions: list[str],
    joint_st_workers: int,
) -> tuple[list[dict], list[dict]]:
    all_conditions = conditional_shift_masks(target, mask, peak_q, high_q, ramp_q)
    conditions = {condition: all_conditions[condition] for condition in joint_conditions if condition in all_conditions}
    exact_zero_by_condition = {
        condition: spatial_condition_min_abs_error_stats(pred, target, mask & condition_mask, None)
        for condition, condition_mask in conditions.items()
    }
    curves_by_key: dict[tuple[int, str], list[dict[str, float | int]]] = {}
    curve_rows: list[dict] = []
    metric_rows: list[dict] = []

    def compute_hop_delta(hop_k: int, delta: int) -> list[tuple[int, str, dict[str, float | int]]]:
        if hop_k == 0:
            reach_indices = None
        else:
            reach_indices = reach_lists_by_k.get(hop_k)
            if reach_indices is None:
                return []
        rows: list[tuple[int, str, dict[str, float | int]]] = []
        for condition, condition_mask in conditions.items():
            pred_slice, target_slice, condition_slice = shifted_slices(pred, target, mask, condition_mask, delta)
            mae, count = spatial_condition_min_abs_error_stats(pred_slice, target_slice, condition_slice, reach_indices)
            rows.append(
                (
                    hop_k,
                    condition,
                    {
                        "time_shift_delta": delta,
                        "st_shift_MAE": mae,
                        "valid_count": count,
                    },
                )
            )
        return rows

    tasks = [
        (hop_k, delta)
        for hop_k in sorted(set(hop_ks))
        for delta in range(-int(max_shift), int(max_shift) + 1)
    ]
    workers = int(max(1, joint_st_workers))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_task_rows = [row for rows in pool.map(lambda item: compute_hop_delta(*item), tasks) for row in rows]
    else:
        all_task_rows = [row for task in tasks for row in compute_hop_delta(*task)]

    for hop_k, condition, row in all_task_rows:
        curves_by_key.setdefault((hop_k, condition), []).append(row)
        curve_rows.append(
            {
                "system": system,
                "horizon": horizon,
                "alignment_hop_k": hop_k,
                "condition": condition,
                **row,
            }
        )

    for (hop_k, condition), curve in curves_by_key.items():
        exact_zero_mae, exact_count = exact_zero_by_condition[condition]
        zero_count = next((int(row["valid_count"]) for row in curve if int(row["time_shift_delta"]) == 0), 0)
        metric_rows.append(
            {
                "system": system,
                "horizon": horizon,
                "alignment_hop_k": hop_k,
                "condition": condition,
                "exact_valid_count": exact_count,
                "condition_valid_count": zero_count,
                **best_st_shift_summary(curve, exact_zero_mae),
            }
        )
    return metric_rows, curve_rows


def patch_condition_masks(
    target: np.ndarray,
    base_mask: np.ndarray,
    patch_len: int,
    peak_q: float,
    high_q: float,
    ramp_q: float,
) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float32)
    base_mask = np.asarray(base_mask, dtype=bool)
    patch_len = int(patch_len)
    num_steps, output_len, num_nodes = target.shape
    num_patch_starts = output_len - patch_len + 1
    if patch_len <= 0 or num_patch_starts <= 0:
        empty = np.zeros((num_steps, 0, num_nodes), dtype=bool)
        return {"all": empty, "peak": empty, "high_volume": empty, "ramp": empty, "normal": empty}

    patch_valid = np.ones((num_steps, num_patch_starts, num_nodes), dtype=bool)
    for offset in range(patch_len):
        patch_valid &= base_mask[:, offset : offset + num_patch_starts, :]

    center_offset = patch_len // 2
    center_values = target[:, center_offset : center_offset + num_patch_starts, :]
    center_valid = base_mask[:, center_offset : center_offset + num_patch_starts, :]
    with np.errstate(invalid="ignore"):
        high_thresholds = np.nanquantile(np.where(base_mask, target, np.nan), high_q, axis=(0, 1)).astype(np.float32)
        peak_thresholds = np.nanquantile(np.where(base_mask, target, np.nan), peak_q, axis=(0, 1)).astype(np.float32)
    high_grid = high_thresholds.reshape(1, 1, num_nodes)
    peak_grid = peak_thresholds.reshape(1, 1, num_nodes)
    high_volume = patch_valid & center_valid & np.isfinite(high_grid) & (center_values >= high_grid)
    peak = patch_valid & center_valid & np.isfinite(peak_grid) & (center_values >= peak_grid)

    ramp = np.full_like(target, np.nan, dtype=np.float32)
    ramp_valid = np.zeros_like(base_mask, dtype=bool)
    if output_len > 1:
        ramp[:, 1:, :] = np.abs(target[:, 1:, :] - target[:, :-1, :])
        ramp_valid[:, 1:, :] = base_mask[:, 1:, :] & base_mask[:, :-1, :] & np.isfinite(ramp[:, 1:, :])
    with np.errstate(invalid="ignore"):
        ramp_thresholds = np.nanquantile(np.where(ramp_valid, ramp, np.nan), ramp_q, axis=(0, 1)).astype(np.float32)
    ramp_grid = ramp_thresholds.reshape(1, 1, num_nodes)
    ramp_center = ramp[:, center_offset : center_offset + num_patch_starts, :]
    ramp_center_valid = ramp_valid[:, center_offset : center_offset + num_patch_starts, :]
    ramp_mask = patch_valid & ramp_center_valid & np.isfinite(ramp_grid) & (ramp_center >= ramp_grid)
    normal = patch_valid & ~peak & ~ramp_mask
    return {
        "all": patch_valid,
        "peak": peak,
        "high_volume": high_volume,
        "ramp": ramp_mask,
        "normal": normal,
    }


def patch_self_mae(
    pred: np.ndarray,
    target: np.ndarray,
    sample_idx: np.ndarray,
    patch_start: np.ndarray,
    node_idx: np.ndarray,
    patch_len: int,
) -> np.ndarray:
    err = np.zeros(sample_idx.shape[0], dtype=np.float64)
    valid = np.ones(sample_idx.shape[0], dtype=bool)
    for offset in range(patch_len):
        pred_values = pred[sample_idx, patch_start + offset, node_idx]
        target_values = target[sample_idx, patch_start + offset, node_idx]
        pair_valid = np.isfinite(pred_values) & np.isfinite(target_values)
        valid &= pair_valid
        err += np.where(pair_valid, np.abs(pred_values - target_values), 0.0)
    return np.where(valid, err / float(patch_len), np.nan)


def patch_neighbor_mae(
    pred: np.ndarray,
    target: np.ndarray,
    sample_idx: np.ndarray,
    target_patch_start: np.ndarray,
    pred_patch_start: np.ndarray,
    node_idx: np.ndarray,
    patch_len: int,
    reach_indices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if reach_indices is None:
        err = np.zeros(sample_idx.shape[0], dtype=np.float64)
        valid = np.ones(sample_idx.shape[0], dtype=bool)
        for offset in range(patch_len):
            pred_values = pred[sample_idx, pred_patch_start + offset, node_idx]
            target_values = target[sample_idx, target_patch_start + offset, node_idx]
            pair_valid = np.isfinite(pred_values) & np.isfinite(target_values)
            valid &= pair_valid
            err += np.where(pair_valid, np.abs(pred_values - target_values), 0.0)
        return np.where(valid, err / float(patch_len), np.nan)
    if isinstance(reach_indices, tuple):
        padded, index_valid = reach_indices
        neighbor_idx = padded[node_idx]
        neighbor_valid = index_valid[node_idx]
    else:
        max_degree = max((len(reach_indices[node]) for node in node_idx), default=0)
        if max_degree == 0:
            return np.full(sample_idx.shape[0], np.nan, dtype=np.float64)
        neighbor_idx = np.zeros((sample_idx.shape[0], max_degree), dtype=np.int64)
        neighbor_valid = np.zeros((sample_idx.shape[0], max_degree), dtype=bool)
        for row, node in enumerate(node_idx):
            indices = reach_indices[int(node)]
            count = len(indices)
            neighbor_idx[row, :count] = indices
            neighbor_valid[row, :count] = True

    err = np.zeros(neighbor_idx.shape, dtype=np.float64)
    valid = neighbor_valid.copy()
    for offset in range(patch_len):
        target_values = target[sample_idx, target_patch_start + offset, node_idx]
        candidate_pred = pred[sample_idx[:, None], (pred_patch_start + offset)[:, None], neighbor_idx]
        pair_valid = np.isfinite(candidate_pred) & np.isfinite(target_values[:, None])
        valid &= pair_valid
        err += np.where(pair_valid, np.abs(candidate_pred - target_values[:, None]), 0.0)
    err = np.where(valid, err / float(patch_len), np.inf)
    best = np.min(err, axis=1)
    return np.where(np.isfinite(best), best, np.nan)


def patch_st_shift_stats(
    pred: np.ndarray,
    target: np.ndarray,
    patch_mask: np.ndarray,
    patch_len: int,
    max_shift: int,
    reach_indices: list[np.ndarray] | tuple[np.ndarray, np.ndarray] | None,
    chunk_size: int,
) -> dict[str, float | int]:
    sample_all, patch_start_all, node_all = np.nonzero(patch_mask)
    total_count = int(sample_all.size)
    if total_count == 0:
        return {
            "condition_valid_count": 0,
            "exact_patch_MAE": float("nan"),
            "zero_shift_patch_ST_MAE": float("nan"),
            "best_patch_ST_MAE": float("nan"),
            "st_shift_gain": float("nan"),
            "spatial_gain_at_zero": float("nan"),
            "extra_time_gain_after_spatial": float("nan"),
            "avg_best_patch_shift_delta": float("nan"),
            "avg_abs_best_patch_shift_delta": float("nan"),
        }

    output_len = pred.shape[1]
    max_patch_start = output_len - patch_len
    exact_total = 0.0
    zero_total = 0.0
    best_total = 0.0
    delta_total = 0.0
    abs_delta_total = 0.0
    count = 0
    chunk_size = int(max(1, chunk_size))
    for start in range(0, total_count, chunk_size):
        end = min(start + chunk_size, total_count)
        sample_idx = sample_all[start:end]
        patch_start = patch_start_all[start:end]
        node_idx = node_all[start:end]
        exact_mae = patch_self_mae(pred, target, sample_idx, patch_start, node_idx, patch_len)
        zero_mae = patch_neighbor_mae(pred, target, sample_idx, patch_start, patch_start, node_idx, patch_len, reach_indices)
        best_mae = np.full(end - start, np.inf, dtype=np.float64)
        best_delta = np.zeros(end - start, dtype=np.int16)
        for delta in range(-int(max_shift), int(max_shift) + 1):
            pred_patch_start = patch_start + delta
            valid_delta = (pred_patch_start >= 0) & (pred_patch_start <= max_patch_start)
            if not np.any(valid_delta):
                continue
            candidate = np.full(end - start, np.nan, dtype=np.float64)
            candidate[valid_delta] = patch_neighbor_mae(
                pred,
                target,
                sample_idx[valid_delta],
                patch_start[valid_delta],
                pred_patch_start[valid_delta],
                node_idx[valid_delta],
                patch_len,
                reach_indices,
            )
            candidate_finite = np.isfinite(candidate)
            better = candidate_finite & (
                (candidate < best_mae)
                | (np.isclose(candidate, best_mae, atol=1e-12, rtol=0.0) & (abs(delta) < np.abs(best_delta)))
            )
            best_mae = np.where(better, candidate, best_mae)
            best_delta = np.where(better, delta, best_delta)

        best_mae = np.where(np.isfinite(best_mae), best_mae, np.nan)
        valid = np.isfinite(exact_mae) & np.isfinite(zero_mae) & np.isfinite(best_mae)
        if not np.any(valid):
            continue
        exact_total += float(np.sum(exact_mae[valid], dtype=np.float64))
        zero_total += float(np.sum(zero_mae[valid], dtype=np.float64))
        best_total += float(np.sum(best_mae[valid], dtype=np.float64))
        delta_total += float(np.sum(best_delta[valid], dtype=np.float64))
        abs_delta_total += float(np.sum(np.abs(best_delta[valid]), dtype=np.float64))
        count += int(np.sum(valid))

    if count == 0:
        exact = zero = best = float("nan")
        avg_delta = avg_abs_delta = float("nan")
    else:
        exact = exact_total / float(count)
        zero = zero_total / float(count)
        best = best_total / float(count)
        avg_delta = delta_total / float(count)
        avg_abs_delta = abs_delta_total / float(count)
    return {
        "condition_valid_count": count,
        "exact_patch_MAE": exact,
        "zero_shift_patch_ST_MAE": zero,
        "best_patch_ST_MAE": best,
        "st_shift_gain": exact - best if np.isfinite(exact) and np.isfinite(best) else float("nan"),
        "spatial_gain_at_zero": exact - zero if np.isfinite(exact) and np.isfinite(zero) else float("nan"),
        "extra_time_gain_after_spatial": zero - best if np.isfinite(zero) and np.isfinite(best) else float("nan"),
        "avg_best_patch_shift_delta": avg_delta,
        "avg_abs_best_patch_shift_delta": avg_abs_delta,
    }


def joint_st_patch_shift_rows(
    system: str,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    patch_len: int,
    max_shift: int,
    hop_ks: list[int],
    reach_lists_by_k: dict[int, list[np.ndarray] | tuple[np.ndarray, np.ndarray]],
    peak_q: float,
    high_q: float,
    ramp_q: float,
    patch_conditions: list[str],
    chunk_size: int,
) -> list[dict]:
    patch_len = int(patch_len)
    if patch_len <= 0 or patch_len > pred.shape[1]:
        return []
    all_conditions = patch_condition_masks(target, mask, patch_len, peak_q, high_q, ramp_q)
    conditions = {condition: all_conditions[condition] for condition in patch_conditions if condition in all_conditions}
    rows: list[dict] = []
    for hop_k in sorted(set(hop_ks)):
        if hop_k == 0:
            reach_indices = None
        else:
            reach_indices = reach_lists_by_k.get(hop_k)
            if reach_indices is None:
                continue
        for condition, condition_mask in conditions.items():
            stats = patch_st_shift_stats(
                pred,
                target,
                condition_mask,
                patch_len,
                max_shift,
                reach_indices,
                chunk_size,
            )
            log(
                "[patch] "
                f"{system} k={hop_k} condition={condition} "
                f"count={stats['condition_valid_count']} "
                f"st_gain={stats['st_shift_gain']:.4g} "
                f"extra_time={stats['extra_time_gain_after_spatial']:.4g}"
            )
            rows.append(
                {
                    "system": system,
                    "alignment_hop_k": hop_k,
                    "condition": condition,
                    "patch_len": patch_len,
                    "max_patch_shift": int(max_shift),
                    **stats,
                }
            )
    return rows


def spatial_dilate(mask: np.ndarray, reach_indices: object | None) -> np.ndarray:
    if reach_indices is None:
        return mask
    if HAS_SCIPY and sparse.issparse(reach_indices):
        counts = reach_indices.dot(mask.T.astype(np.uint8)).T
        return np.asarray(counts > 0)
    out = np.zeros_like(mask, dtype=bool)
    for node, indices in enumerate(reach_indices):
        if indices.size == 1 and indices[0] == node:
            out[:, node] = mask[:, node]
        else:
            out[:, node] = np.any(mask[:, indices], axis=1)
    return out


def temporal_spatial_dilate(
    mask: np.ndarray,
    reach_indices: object | None,
    time_window: int,
) -> np.ndarray:
    spatial_mask = spatial_dilate(mask, reach_indices)
    return temporal_dilate(spatial_mask, time_window)


def temporal_dilate(mask: np.ndarray, time_window: int) -> np.ndarray:
    time_window = int(max(0, time_window))
    num_steps = mask.shape[0]
    out = np.zeros_like(mask, dtype=bool)
    for delta in range(-time_window, time_window + 1):
        if delta < 0:
            source = mask[: num_steps + delta]
            target_slice = slice(-delta, None)
        elif delta > 0:
            source = mask[delta:]
            target_slice = slice(0, num_steps - delta)
        else:
            source = mask
            target_slice = slice(None)
        out[target_slice] |= source
    return out


def nearest_peak_lag(
    source_peak: np.ndarray,
    target_peak: np.ndarray,
    time_window: int,
) -> tuple[float, float, int]:
    time_window = int(max(0, time_window))
    num_steps = source_peak.shape[0]
    best_lag = np.full(source_peak.shape, time_window + 1, dtype=np.int16)
    for delta in range(-time_window, time_window + 1):
        if delta < 0:
            candidate = source_peak[-delta:] & target_peak[: num_steps + delta]
            best_lag[-delta:] = np.where(candidate, np.minimum(best_lag[-delta:], abs(delta)), best_lag[-delta:])
        elif delta > 0:
            candidate = source_peak[: num_steps - delta] & target_peak[delta:]
            best_lag[: num_steps - delta] = np.where(candidate, np.minimum(best_lag[: num_steps - delta], delta), best_lag[: num_steps - delta])
        else:
            candidate = source_peak & target_peak
            best_lag = np.where(candidate, 0, best_lag)
    source_count = int(np.sum(source_peak))
    matched = source_peak & (best_lag <= time_window)
    matched_count = int(np.sum(matched))
    if matched_count == 0:
        return float("nan"), float("nan"), source_count
    return float(np.mean(best_lag[matched])), float(matched_count / source_count) if source_count else float("nan"), source_count


def relaxed_peak_hit_metrics(
    true_peak: np.ndarray,
    pred_peak: np.ndarray,
    hop_k: int,
    time_window: int,
    true_near: np.ndarray,
    pred_near: np.ndarray,
) -> dict[str, float | int]:
    pred_count = int(np.sum(pred_peak))
    true_count = int(np.sum(true_peak))
    pred_hit_count = int(np.sum(pred_peak & true_near))
    true_hit_count = int(np.sum(true_peak & pred_near))
    precision = pred_hit_count / pred_count if pred_count else float("nan")
    recall = true_hit_count / true_count if true_count else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    return {
        "alignment_hop_k": hop_k,
        "alignment_time_window": time_window,
        "relaxed_peak_precision": float(precision),
        "relaxed_peak_recall": float(recall),
        "relaxed_peak_F1": float(f1),
        "relaxed_pred_hit_count": pred_hit_count,
        "relaxed_true_hit_count": true_hit_count,
        "peak_pred_count": pred_count,
        "peak_true_count": true_count,
    }


def alignment_rows(
    system: str,
    horizon: int,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    q: float,
    max_shift: int,
    time_windows: list[int],
    hop_ks: list[int],
    reach_by_k: dict[int, object],
) -> tuple[list[dict], list[dict]]:
    shift_curve = shifted_mae_curve(pred, target, mask, max_shift)
    shift_summary = best_shift_summary(shift_curve)
    true_peak, pred_peak, _ = peak_masks(pred, target, q, mask)
    lag_window = int(max(time_windows)) if time_windows else int(max_shift)
    avg_lag, lag_match_rate, lag_source_count = nearest_peak_lag(true_peak, pred_peak, lag_window)
    curve_rows = [
        {
            "system": system,
            "horizon": horizon,
            **row,
        }
        for row in shift_curve
    ]

    rows: list[dict] = []
    for hop_k in sorted(set(hop_ks)):
        if hop_k > 0 and hop_k not in reach_by_k:
            continue
        reach_indices = reach_by_k.get(hop_k)
        if hop_k == 0:
            reach_indices = None
        true_spatial = spatial_dilate(true_peak, reach_indices)
        pred_spatial = spatial_dilate(pred_peak, reach_indices)
        for time_window in sorted(set(time_windows)):
            true_near = temporal_dilate(true_spatial, time_window)
            pred_near = temporal_dilate(pred_spatial, time_window)
            rows.append(
                {
                    "system": system,
                    "horizon": horizon,
                    "peak_lag_window": lag_window,
                    "avg_peak_lag_true_to_pred": avg_lag,
                    "peak_lag_match_rate": lag_match_rate,
                    "peak_lag_true_count": lag_source_count,
                    **shift_summary,
                    **relaxed_peak_hit_metrics(true_peak, pred_peak, hop_k, time_window, true_near, pred_near),
                }
            )
    return rows, curve_rows


def standard_performance_row(
    system: str,
    horizon: int,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    scale_abs: np.ndarray | None,
    scale_sq: np.ndarray | None,
    peak_q: float,
    worst_pct: float,
) -> dict[str, float | int | str]:
    peak_metrics = peak_detection_metrics(pred, target, peak_q, mask)
    peak = quantile_peak_mask(np.where(mask, target, np.nan), peak_q)
    normal = mask & ~peak
    normal_mae = masked_mae(pred, target, normal)
    peak_mae = masked_mae(pred, target, peak & mask)
    return {
        "system": system,
        "horizon": horizon,
        "MAE": masked_mae(pred, target, mask),
        "RMSE": masked_rmse(pred, target, mask),
        "WAPE": masked_wape(pred, target, mask),
        "W1": wasserstein_1d(pred, target, mask),
        "MASE": masked_mase(pred, target, scale_abs, mask),
        "RMSSE": masked_rmsse(pred, target, scale_sq, mask),
        "worst_pct": worst_pct,
        "worst_pct_MAE": worst_pct_mae(pred, target, worst_pct, mask),
        "normal_MAE": normal_mae,
        "peak_MAE": peak_mae,
        "peak_over_normal_MAE": (
            peak_mae / normal_mae
            if normal_mae and np.isfinite(normal_mae)
            else float("nan")
        ),
        "valid_count": int(np.sum(np.isfinite(pred) & np.isfinite(target) & mask)),
        **peak_metrics,
    }


def wasserstein_1d(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred, target, valid = finite_pair(pred, target)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return float("nan")
    x = np.sort(pred[valid].reshape(-1))
    y = np.sort(target[valid].reshape(-1))
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    return float(np.mean(np.abs(x - y)))


def centered_moving_average(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(series, dtype=np.float32).copy()
    window = int(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(series, ((left, right), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0, dtype=np.float64)
    cumsum = np.vstack([np.zeros((1, padded.shape[1]), dtype=np.float64), cumsum])
    low = (cumsum[window:] - cumsum[:-window]) / float(window)
    return low.astype(np.float32)


def fill_nonfinite_for_fft(series: np.ndarray) -> np.ndarray:
    work = np.asarray(series, dtype=np.float64)
    valid = np.isfinite(work)
    if np.all(valid):
        return work
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(np.where(valid, work, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    return np.where(valid, work, col_mean.reshape(1, -1))


def fft_lowpass(series: np.ndarray, cutoff_period: float) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    if cutoff_period <= 1 or series.shape[0] <= 1:
        return series.copy()
    n_steps = int(series.shape[0])
    cutoff_freq = 1.0 / float(cutoff_period)
    freqs = np.fft.rfftfreq(n_steps, d=1.0)
    keep = freqs <= cutoff_freq + 1e-12

    work = fill_nonfinite_for_fft(series)
    spectrum = np.fft.rfft(work, axis=0)
    spectrum[~keep, :] = 0.0
    low = np.fft.irfft(spectrum, n=n_steps, axis=0).astype(np.float32)
    low[~np.isfinite(series)] = np.nan
    return low


def decompose_series(series: np.ndarray, method: str, moving_window: int, fft_cutoff_period: float) -> tuple[np.ndarray, np.ndarray]:
    if method == "moving_average":
        low = centered_moving_average(series, moving_window)
    elif method == "fft_lowpass":
        low = fft_lowpass(series, fft_cutoff_period)
    else:
        raise ValueError(f"Unknown decomposition method: {method}")
    high = np.asarray(series, dtype=np.float32) - low
    return low, high


def metric_row(
    system: str,
    horizon: int,
    decomposition_method: str,
    component: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int | str]:
    return {
        "system": system,
        "horizon": horizon,
        "decomposition_method": decomposition_method,
        "component": component,
        "MAE": masked_mae(pred, target),
        "RMSE": masked_rmse(pred, target),
        "WAPE": masked_wape(pred, target),
        "W1": wasserstein_1d(pred, target),
    }


def quantile_peak_mask(target: np.ndarray, q: float) -> np.ndarray:
    thresholds = np.nanquantile(target, q, axis=0).astype(np.float32)
    return target >= thresholds.reshape(1, -1)


def peak_rows(
    system: str,
    horizon: int,
    pred: np.ndarray,
    target: np.ndarray,
    q: float,
    base_mask: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    threshold_target = np.where(base_mask, target, np.nan) if base_mask is not None else target
    peak = quantile_peak_mask(threshold_target, q)
    normal = ~peak
    rows = []
    for split_name, split_mask in [("normal", normal), (f"peak_q{q:.2f}", peak)]:
        combined_mask = split_mask if base_mask is None else (split_mask & base_mask)
        rows.append(
            {
                "system": system,
                "horizon": horizon,
                "metric_scope": "standard",
                "window_type": split_name,
                "MAE": masked_mae(pred, target, combined_mask),
                "RMSE": masked_rmse(pred, target, combined_mask),
                "WAPE": masked_wape(pred, target, combined_mask),
                "W1": wasserstein_1d(pred, target, combined_mask),
                "valid_count": int(np.sum(np.isfinite(pred) & np.isfinite(target) & combined_mask)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


STANDARD_RANK_METRICS = [
    ("MAE", "lower"),
    ("WAPE", "lower"),
    ("MASE", "lower"),
    ("worst_pct_MAE", "lower"),
    ("peak_F1", "higher"),
]


def rank_values(values: dict[str, float], direction: str) -> dict[str, float]:
    finite = [(system, value) for system, value in values.items() if np.isfinite(value)]
    finite.sort(key=lambda item: item[1], reverse=direction == "higher")
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(finite):
        j = idx + 1
        while j < len(finite) and finite[j][1] == finite[idx][1]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[finite[k][0]] = avg_rank
        idx = j
    for system in values:
        ranks.setdefault(system, float("nan"))
    return ranks


def standard_rank_tables(standard_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    systems = sorted({row["system"] for row in standard_rows})
    horizons = sorted({int(row["horizon"]) for row in standard_rows})
    by_key = {(row["system"], int(row["horizon"])): row for row in standard_rows}
    detail_rows: list[dict] = []
    rank_values_by_system: dict[str, list[float]] = {system: [] for system in systems}

    for horizon in horizons:
        for metric, direction in STANDARD_RANK_METRICS:
            values = {}
            for system in systems:
                row = by_key.get((system, horizon), {})
                try:
                    values[system] = float(row.get(metric, float("nan")))
                except (TypeError, ValueError):
                    values[system] = float("nan")
            ranks = rank_values(values, direction)
            for system in systems:
                rank = ranks[system]
                if np.isfinite(rank):
                    rank_values_by_system[system].append(rank)
                detail_rows.append(
                    {
                        "system": system,
                        "horizon": horizon,
                        "metric": metric,
                        "direction": direction,
                        "value": values[system],
                        "rank": rank,
                    }
                )

    summary_rows = []
    for system in systems:
        ranks = np.asarray(rank_values_by_system[system], dtype=np.float64)
        finite = ranks[np.isfinite(ranks)]
        summary_rows.append(
            {
                "system": system,
                "avg_standard_rank": float(np.mean(finite)) if finite.size else float("nan"),
                "rank_count": int(finite.size),
            }
        )
    summary_rows.sort(key=lambda row: (float(row["avg_standard_rank"]) if np.isfinite(float(row["avg_standard_rank"])) else float("inf"), row["system"]))
    return detail_rows, summary_rows


def unwrap_adj(payload) -> np.ndarray:
    if isinstance(payload, tuple) and len(payload) == 3:
        payload = payload[2]
    elif isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    return np.asarray(payload, dtype=np.float32)


def load_adjacency(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    with path.open("rb") as fp:
        return unwrap_adj(pickle.load(fp))


def sample_edges(adj: np.ndarray, max_edges: int) -> np.ndarray:
    work = np.asarray(adj)
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError(f"Adjacency must be square, got {work.shape}")
    rows, cols = np.where(work > 0)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    if len(rows) > max_edges:
        idx = np.linspace(0, len(rows) - 1, max_edges, dtype=int)
        rows, cols = rows[idx], cols[idx]
    return np.stack([rows, cols], axis=1) if len(rows) else np.empty((0, 2), dtype=np.int64)


def pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return float("nan")
    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def residual_structure_row(
    system: str,
    horizon: int,
    decomposition_method: str,
    component: str,
    residual: np.ndarray,
) -> dict[str, float | int | str]:
    residual = np.asarray(residual, dtype=np.float32)
    valid = np.isfinite(residual)
    if not np.any(valid):
        return {
            "system": system,
            "horizon": horizon,
            "decomposition_method": decomposition_method,
            "component": component,
            "bias": float("nan"),
            "residual_std": float("nan"),
            "mean_abs_residual": float("nan"),
            "under_prediction_rate": float("nan"),
            "temporal_lag1_corr": float("nan"),
            "node_bias_std": float("nan"),
            "valid_count": 0,
        }
    values = residual[valid]
    lag1 = pair_corr(residual[:-1, :].reshape(-1), residual[1:, :].reshape(-1)) if residual.shape[0] > 1 else float("nan")
    with np.errstate(invalid="ignore"):
        node_bias = np.nanmean(np.where(valid, residual, np.nan), axis=0)
    return {
        "system": system,
        "horizon": horizon,
        "decomposition_method": decomposition_method,
        "component": component,
        "bias": float(np.mean(values)),
        "residual_std": float(np.std(values)),
        "mean_abs_residual": float(np.mean(np.abs(values))),
        "under_prediction_rate": float(np.mean(values > 0)),
        "temporal_lag1_corr": lag1,
        "node_bias_std": float(np.nanstd(node_bias)),
        "valid_count": int(np.sum(valid)),
    }


def spatial_rows(
    system: str,
    horizon: int,
    decomposition_method: str,
    component: str,
    residual: np.ndarray,
    edges: np.ndarray,
    adj: np.ndarray,
) -> dict[str, float | int | str]:
    if len(edges) == 0:
        return {
            "system": system,
            "horizon": horizon,
            "decomposition_method": decomposition_method,
            "component": component,
            "edge_residual_corr": float("nan"),
            "residual_dirichlet": float("nan"),
            "num_edges": 0,
        }

    corrs = []
    diffs = []
    weights = []
    for src, dst in edges:
        corrs.append(pair_corr(residual[:, src], residual[:, dst]))
        diff = residual[:, src] - residual[:, dst]
        valid = np.isfinite(diff)
        if np.any(valid):
            diffs.append(float(np.mean(diff[valid] * diff[valid])))
            weights.append(float(adj[src, dst]))
    finite_corrs = np.asarray([item for item in corrs if np.isfinite(item)], dtype=np.float64)
    if diffs and np.sum(weights) > 0:
        dirichlet = float(np.average(np.asarray(diffs), weights=np.asarray(weights)))
    elif diffs:
        dirichlet = float(np.mean(diffs))
    else:
        dirichlet = float("nan")
    return {
        "system": system,
        "horizon": horizon,
        "decomposition_method": decomposition_method,
        "component": component,
        "edge_residual_corr": float(np.mean(finite_corrs)) if finite_corrs.size else float("nan"),
        "residual_dirichlet": dirichlet,
        "num_edges": int(len(edges)),
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_markdown(
    report: dict,
    standard_rows: list[dict],
    rank_summary_rows: list[dict],
    alignment_metric_rows: list[dict],
    conditional_shift_metric_rows: list[dict],
    joint_st_shift_metric_rows: list[dict],
    joint_st_patch_metric_rows: list[dict],
    component_rows: list[dict],
    peak_metric_rows: list[dict],
    residual_metric_rows: list[dict],
    spatial_metric_rows: list[dict],
) -> str:
    lines = [
        "# Decoupled Spatiotemporal Diagnostic Summary",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Dataset dir: `{report['dataset_dir']}`",
        f"- Adjacency path: `{report['adj_path'] or 'none'}`",
        f"- Alignment directed adjacency: {report['alignment_directed']}",
        f"- Runs analyzed: {report['num_runs']}",
        f"- Decomposition methods: {', '.join(report['decomposition_methods'])}",
        f"- Moving-average window: {report['moving_window']}",
        f"- FFT cutoff period: {report['fft_cutoff_period']}",
        f"- Seasonal period for MASE/RMSSE: {report['seasonal_period'] or 'disabled'}",
        f"- Worst-window fraction: {report['worst_pct']}",
        f"- Alignment max shift: {report['alignment_max_shift']}",
        f"- Alignment time windows: {', '.join(str(item) for item in report['alignment_time_windows'])}",
        f"- Alignment hop tolerances: {', '.join(str(item) for item in report['alignment_hop_ks'])}",
        f"- Alignment-only mode: {report['alignment_only']}",
        f"- Patch-only mode: {report['patch_only']}",
        f"- Conditional high-volume quantile: {report['condition_high_q']}",
        f"- Conditional ramp quantile: {report['condition_ramp_q']}",
        f"- Joint ST patch length: {report['joint_st_patch_len']}",
        f"- Joint ST patch conditions: {', '.join(report['joint_st_patch_conditions'])}",
        f"- Joint ST patch top-k graph neighbors: {report['joint_st_patch_topk'] or 'all'}",
        f"- Horizons: {', '.join(str(h) for h in report['horizons'])}",
        "",
        "## Runs",
        "",
    ]
    for run in report["runs"]:
        lines.append(f"- `{run['name']}`: `{run['result_dir']}` shape={run.get('shape', 'unknown')}")

    lines.extend(["", "## Standard Performance Metrics", ""])
    if standard_rows:
        lines.append("| System | Horizon | MAE | RMSE | WAPE | MASE | RMSSE | Worst MAE | Peak F1 | Full W1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in standard_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['MAE']:.4g} | {row['RMSE']:.4g} | "
                f"{row['WAPE']:.4g} | {row['MASE']:.4g} | {row['RMSSE']:.4g} | "
                f"{row['worst_pct_MAE']:.4g} | {row['peak_F1']:.4g} | {row['W1']:.4g} |"
            )
    else:
        lines.append("No standard performance metrics were generated.")

    if rank_summary_rows:
        lines.extend(["", "## GIFT-Style Average Rank", ""])
        lines.append("| Rank | System | Avg Rank | Rank Count |")
        lines.append("|---:|---|---:|---:|")
        for idx, row in enumerate(rank_summary_rows, start=1):
            lines.append(f"| {idx} | {row['system']} | {row['avg_standard_rank']:.4g} | {row['rank_count']} |")

    lines.extend(["", "## Alignment Diagnostics", ""])
    if alignment_metric_rows:
        lines.append("Relaxed peak-hit metrics detect small time/node misalignment. Hop tolerances above 0 require adjacency.")
        lines.append("| System | Horizon | Hop k | Time Window | Shift Gain | Best Shift | Peak Lag | Relaxed Precision | Relaxed Recall | Relaxed F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in alignment_metric_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['alignment_hop_k']} | {row['alignment_time_window']} | "
                f"{row['shift_gain']:.4g} | {row['best_time_shift_delta']} | {row['avg_peak_lag_true_to_pred']:.4g} | "
                f"{row['relaxed_peak_precision']:.4g} | {row['relaxed_peak_recall']:.4g} | {row['relaxed_peak_F1']:.4g} |"
            )
    else:
        lines.append("No alignment diagnostics were generated.")

    lines.extend(["", "## Conditional ShiftGain Diagnostics", ""])
    if conditional_shift_metric_rows:
        lines.append("These metrics compute shifted-MAE improvement inside ground-truth-defined traffic regimes.")
        lines.append("| System | Horizon | Condition | Shift Gain | Best Shift | Zero MAE | Best MAE | Count |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for row in conditional_shift_metric_rows[:100]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['condition']} | "
                f"{row['shift_gain']:.4g} | {row['best_time_shift_delta']} | {row['zero_shift_MAE']:.4g} | "
                f"{row['best_shift_MAE']:.4g} | {row['condition_valid_count']} |"
            )
    else:
        lines.append("No conditional ShiftGain diagnostics were generated.")

    lines.extend(["", "## Joint Spatiotemporal ShiftGain Diagnostics", ""])
    if joint_st_shift_metric_rows:
        lines.append("STShiftGain allows both prediction-time shifts and k-hop spatial substitution, reported as absolute MAE reduction against exact same-node/time MAE.")
        lines.append("| System | Horizon | Hop k | Condition | STShiftGain (MAE drop) | Spatial Gain @0 (MAE drop) | Best Shift | Exact MAE | Best ST-MAE | Count |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in joint_st_shift_metric_rows[:120]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['alignment_hop_k']} | {row['condition']} | "
                f"{row['st_shift_gain']:.4g} | {row['spatial_gain_at_zero']:.4g} | {row['best_time_shift_delta']} | "
                f"{row['exact_zero_MAE']:.4g} | {row['best_ST_MAE']:.4g} | {row['condition_valid_count']} |"
            )
    if joint_st_patch_metric_rows:
        lines.extend(["", "## Patch-Level Joint Spatiotemporal Shift Diagnostics", ""])
        lines.append("Patch STShiftGain selects one k-hop neighbor and one forecast-patch offset for an entire patch, reported as absolute patch-MAE reduction.")
        lines.append("| System | Hop k | Condition | Patch | STShiftGain (MAE drop) | Spatial Gain @0 (MAE drop) | Extra Time Gain | Avg Best Patch Shift | Exact Patch MAE | Best Patch ST-MAE | Count |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in joint_st_patch_metric_rows:
            lines.append(
                f"| {row['system']} | {row['alignment_hop_k']} | {row['condition']} | {row['patch_len']} | "
                f"{row['st_shift_gain']:.4g} | {row['spatial_gain_at_zero']:.4g} | {row['extra_time_gain_after_spatial']:.4g} | "
                f"{row['avg_best_patch_shift_delta']:.4g} | {row['exact_patch_MAE']:.4g} | {row['best_patch_ST_MAE']:.4g} | "
                f"{row['condition_valid_count']} |"
            )
    else:
        lines.append("No joint spatiotemporal ShiftGain diagnostics were generated.")

    lines.extend(["", "## Decomposition-Dependent Low/High Metrics", ""])
    if component_rows:
        lines.append("| System | Method | Horizon | Component | MAE | RMSE | WAPE | W1 |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|")
        shown = [row for row in component_rows if row.get("component") in {"low", "high"}]
        for row in shown[:80]:
            lines.append(
                f"| {row['system']} | {row['decomposition_method']} | {row['horizon']} | {row['component']} | "
                f"{row['MAE']:.4g} | {row['RMSE']:.4g} | {row['WAPE']:.4g} | {row['W1']:.4g} |"
            )
    else:
        lines.append("No component metrics were generated.")

    lines.extend(["", "## Peak Metrics", ""])
    if peak_metric_rows:
        lines.append("| System | Horizon | Window | MAE | WAPE | W1 |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for row in peak_metric_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['window_type']} | "
                f"{row['MAE']:.4g} | {row['WAPE']:.4g} | {row['W1']:.4g} |"
            )

    if residual_metric_rows:
        lines.extend(["", "## Residual Structure Metrics", ""])
        lines.append("| System | Method | Horizon | Component | Bias | Abs Residual | Lag1 Corr | Under Rate | Node Bias Std |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for row in residual_metric_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['decomposition_method']} | {row['horizon']} | {row['component']} | "
                f"{row['bias']:.4g} | {row['mean_abs_residual']:.4g} | {row['temporal_lag1_corr']:.4g} | "
                f"{row['under_prediction_rate']:.4g} | {row['node_bias_std']:.4g} |"
            )

    if spatial_metric_rows:
        lines.extend(["", "## Spatial Structure Metrics", ""])
        lines.append("These are structural diagnostics, not direct forecast-performance ranking metrics.")
        lines.append("| System | Method | Horizon | Component | Edge Corr | Dirichlet | Edges |")
        lines.append("|---|---|---:|---|---:|---:|---:|")
        for row in spatial_metric_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['decomposition_method']} | {row['horizon']} | {row['component']} | "
                f"{row['edge_residual_corr']:.4g} | {row['residual_dirichlet']:.4g} | {row['num_edges']} |"
            )
    else:
        lines.extend(["", "## Spatial Residual Metrics", "", "Skipped; no matching adjacency was provided."])

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    desc = load_dataset_desc(args)
    num_nodes, output_len = load_dataset_shape(args)
    horizons = sorted({h for h in args.horizons if 1 <= h <= output_len})
    if not horizons:
        raise ValueError(f"No valid horizons in {args.horizons}; output_len={output_len}")
    decomposition_methods = list(dict.fromkeys(args.decomp_methods))
    fft_cutoff_period = float(args.fft_cutoff_period or args.moving_window)
    if fft_cutoff_period <= 0:
        raise ValueError(f"--fft-cutoff-period must be positive, got {fft_cutoff_period}")
    if not (0 < args.peak_q < 1):
        raise ValueError(f"--peak-q must be in (0, 1), got {args.peak_q}")
    if not (0 < args.worst_pct <= 1):
        raise ValueError(f"--worst-pct must be in (0, 1], got {args.worst_pct}")
    if args.alignment_max_shift < 0:
        raise ValueError(f"--alignment-max-shift must be >= 0, got {args.alignment_max_shift}")
    if args.patch_only and args.joint_st_patch_len <= 0:
        raise ValueError("--patch-only requires --joint-st-patch-len > 0.")
    if not (0 < args.condition_high_q < 1):
        raise ValueError(f"--condition-high-q must be in (0, 1), got {args.condition_high_q}")
    if not (0 < args.condition_ramp_q < 1):
        raise ValueError(f"--condition-ramp-q must be in (0, 1), got {args.condition_ramp_q}")
    alignment_time_windows = sorted({int(item) for item in args.alignment_time_windows if int(item) >= 0})
    alignment_hop_ks = sorted({int(item) for item in args.alignment_hop_ks if int(item) >= 0})
    if not alignment_time_windows:
        alignment_time_windows = [0]
    if not alignment_hop_ks:
        alignment_hop_ks = [0]
    null_val = parse_null_value(args.null_val, desc)
    seasonal_period = infer_seasonal_period(desc, args.seasonal_period)
    scale_abs, scale_sq = load_seasonal_scales(args, desc, num_nodes, seasonal_period, null_val)

    runs = discover_runs(args)
    if not runs:
        raise FileNotFoundError("No BasicTS test_results directories were found. Pass --run or --search-root.")

    adj = None
    edges = np.empty((0, 2), dtype=np.int64)
    if args.adj_path is not None:
        adj = load_adjacency(args.adj_path)
        if adj.shape != (num_nodes, num_nodes):
            raise ValueError(f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}")
        edges = sample_edges(adj, args.max_edge_pairs)
    reach_by_k = build_reach_indices(adj, alignment_hop_ks, directed=args.alignment_directed)
    if adj is None and any(k > 0 for k in alignment_hop_ks):
        log("[warn] --alignment-hop-ks includes k>0 but no --adj-path was provided; spatial relaxed-hit rows will be skipped.")

    standard_rows: list[dict] = []
    alignment_metric_rows: list[dict] = []
    time_shift_curve_rows: list[dict] = []
    conditional_shift_metric_rows: list[dict] = []
    conditional_time_shift_curve_rows: list[dict] = []
    joint_st_shift_metric_rows: list[dict] = []
    joint_st_time_shift_curve_rows: list[dict] = []
    joint_st_patch_metric_rows: list[dict] = []
    component_rows: list[dict] = []
    peak_metric_rows: list[dict] = []
    distribution_rows: list[dict] = []
    residual_metric_rows: list[dict] = []
    spatial_metric_rows: list[dict] = []
    run_reports: list[dict] = []
    reach_lists_by_k = build_reach_lists(adj, alignment_hop_ks, directed=args.alignment_directed)
    reach_padded_by_k = pad_reach_lists(reach_lists_by_k)
    reach_value_by_k: dict[int, list[np.ndarray] | tuple[np.ndarray, np.ndarray]] = {
        hop_k: reach_padded_by_k.get(hop_k, reach_lists)
        for hop_k, reach_lists in reach_lists_by_k.items()
    }
    patch_reach_lists_by_k = limit_reach_lists_by_topk(
        adj,
        reach_lists_by_k,
        args.joint_st_patch_topk,
        directed=args.alignment_directed,
    )
    patch_reach_padded_by_k = pad_reach_lists(patch_reach_lists_by_k)
    patch_reach_value_by_k: dict[int, list[np.ndarray] | tuple[np.ndarray, np.ndarray]] = {
        hop_k: patch_reach_padded_by_k.get(hop_k, reach_lists)
        for hop_k, reach_lists in patch_reach_lists_by_k.items()
    }

    for run in runs:
        pred_path = run.result_dir / "predictions.npy"
        target_path = run.result_dir / "targets.npy"
        if not pred_path.exists() or not target_path.exists():
            log(f"[warn] skip missing predictions/targets: {run.result_dir}")
            continue

        log(f"[run] {run.name}: {run.result_dir}")
        pred = load_saved_array(pred_path, args.dtype, output_len, num_nodes, args.target_dim)
        target = load_saved_array(target_path, args.dtype, output_len, num_nodes, args.target_dim)
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch for {run.name}: {pred.shape} vs {target.shape}")

        run_reports.append({"name": run.name, "result_dir": str(run.result_dir), "shape": list(pred.shape)})
        if args.joint_st_patch_len > 0:
            joint_st_patch_metric_rows.extend(
                joint_st_patch_shift_rows(
                    run.name,
                    pred,
                    target,
                    valid_target_mask(target, null_val),
                    args.joint_st_patch_len,
                    args.alignment_max_shift,
                    alignment_hop_ks,
                    patch_reach_value_by_k,
                    args.peak_q,
                    args.condition_high_q,
                    args.condition_ramp_q,
                    args.joint_st_patch_conditions,
                    args.joint_st_patch_chunk_size,
                )
            )

        if args.patch_only:
            continue

        for horizon in horizons:
            h_idx = horizon - 1
            pred_h = pred[:, h_idx, :]
            target_h = target[:, h_idx, :]
            target_mask = valid_target_mask(target_h, null_val)

            if not args.alignment_only:
                standard_row = standard_performance_row(
                    run.name,
                    horizon,
                    pred_h,
                    target_h,
                    target_mask,
                    scale_abs,
                    scale_sq,
                    args.peak_q,
                    args.worst_pct,
                )
                standard_rows.append(standard_row)
                peak_metric_rows.extend(peak_rows(run.name, horizon, pred_h, target_h, args.peak_q, target_mask))
            align_rows, shift_rows = alignment_rows(
                run.name,
                horizon,
                pred_h,
                target_h,
                target_mask,
                args.peak_q,
                args.alignment_max_shift,
                alignment_time_windows,
                alignment_hop_ks,
                reach_by_k,
            )
            alignment_metric_rows.extend(align_rows)
            time_shift_curve_rows.extend(shift_rows)
            cond_rows, cond_curve_rows = conditional_shift_rows(
                run.name,
                horizon,
                pred_h,
                target_h,
                target_mask,
                args.alignment_max_shift,
                args.peak_q,
                args.condition_high_q,
                args.condition_ramp_q,
            )
            conditional_shift_metric_rows.extend(cond_rows)
            conditional_time_shift_curve_rows.extend(cond_curve_rows)
            st_rows, st_curve_rows = joint_st_shift_rows(
                run.name,
                horizon,
                pred_h,
                target_h,
                target_mask,
                args.alignment_max_shift,
                alignment_hop_ks,
                reach_value_by_k,
                args.peak_q,
                args.condition_high_q,
                args.condition_ramp_q,
                args.joint_st_conditions,
                args.joint_st_workers,
            )
            joint_st_shift_metric_rows.extend(st_rows)
            joint_st_time_shift_curve_rows.extend(st_curve_rows)

            if not args.alignment_only:
                for method in decomposition_methods:
                    pred_low, pred_high = decompose_series(pred_h, method, args.moving_window, fft_cutoff_period)
                    target_low, target_high = decompose_series(target_h, method, args.moving_window, fft_cutoff_period)

                    components = {
                        "full": (pred_h, target_h),
                        "low": (pred_low, target_low),
                        "high": (pred_high, target_high),
                    }
                    for component, (pred_component, target_component) in components.items():
                        row = metric_row(run.name, horizon, method, component, pred_component, target_component)
                        component_rows.append(row)
                        distribution_rows.append(
                            {
                                "system": run.name,
                                "horizon": horizon,
                                "decomposition_method": method,
                                "component": component,
                                "W1": row["W1"],
                            }
                        )
                        residual_metric_rows.append(
                            residual_structure_row(run.name, horizon, method, component, target_component - pred_component)
                        )

                    if adj is not None:
                        residual_components = {
                            "full": target_h - pred_h,
                            "low": target_low - pred_low,
                            "high": target_high - pred_high,
                        }
                        for component, residual in residual_components.items():
                            spatial_metric_rows.append(spatial_rows(run.name, horizon, method, component, residual, edges, adj))

    report = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir(args)),
        "num_nodes": num_nodes,
        "output_len": output_len,
        "horizons": horizons,
        "decomposition_methods": decomposition_methods,
        "moving_window": args.moving_window,
        "fft_cutoff_period": fft_cutoff_period,
        "seasonal_period": seasonal_period,
        "target_channel": args.target_channel,
        "null_val": null_val,
        "peak_q": args.peak_q,
        "worst_pct": args.worst_pct,
        "alignment_max_shift": args.alignment_max_shift,
        "alignment_time_windows": alignment_time_windows,
        "alignment_hop_ks": alignment_hop_ks,
        "alignment_directed": bool(args.alignment_directed),
        "alignment_only": bool(args.alignment_only),
        "patch_only": bool(args.patch_only),
        "adj_path": str(args.adj_path.expanduser().resolve()) if args.adj_path is not None else None,
        "condition_high_q": args.condition_high_q,
        "condition_ramp_q": args.condition_ramp_q,
        "joint_st_conditions": args.joint_st_conditions,
        "joint_st_workers": int(max(1, args.joint_st_workers)),
        "joint_st_patch_len": int(args.joint_st_patch_len),
        "joint_st_patch_conditions": args.joint_st_patch_conditions,
        "joint_st_patch_chunk_size": int(args.joint_st_patch_chunk_size),
        "joint_st_patch_topk": int(args.joint_st_patch_topk),
        "num_runs": len(run_reports),
        "runs": run_reports,
    }
    rank_detail_rows, rank_summary_rows = standard_rank_tables(standard_rows)

    write_csv(args.output_dir / "standard_performance_metrics.csv", standard_rows)
    write_csv(args.output_dir / "standard_rank_details.csv", rank_detail_rows)
    write_csv(args.output_dir / "standard_rank_summary.csv", rank_summary_rows)
    write_csv(args.output_dir / "alignment_metrics.csv", alignment_metric_rows)
    write_csv(args.output_dir / "time_shift_curve.csv", time_shift_curve_rows)
    write_csv(args.output_dir / "conditional_shift_metrics.csv", conditional_shift_metric_rows)
    write_csv(args.output_dir / "conditional_time_shift_curve.csv", conditional_time_shift_curve_rows)
    write_csv(args.output_dir / "joint_st_shift_metrics.csv", joint_st_shift_metric_rows)
    write_csv(args.output_dir / "joint_st_time_shift_curve.csv", joint_st_time_shift_curve_rows)
    write_csv(args.output_dir / "joint_st_patch_shift_metrics.csv", joint_st_patch_metric_rows)
    write_csv(args.output_dir / "joint_st_patch_shift_summary.csv", joint_st_patch_metric_rows)
    write_csv(args.output_dir / "component_metrics.csv", component_rows)
    write_csv(args.output_dir / "peak_window_metrics.csv", peak_metric_rows)
    write_csv(args.output_dir / "distribution_metrics.csv", distribution_rows)
    write_csv(args.output_dir / "residual_structure_metrics.csv", residual_metric_rows)
    write_csv(args.output_dir / "spatial_residual_metrics.csv", spatial_metric_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    (args.output_dir / "diagnostic_summary.md").write_text(
        build_markdown(
            report,
            standard_rows,
            rank_summary_rows,
            alignment_metric_rows,
            conditional_shift_metric_rows,
            joint_st_shift_metric_rows,
            joint_st_patch_metric_rows,
            component_rows,
            peak_metric_rows,
            residual_metric_rows,
            spatial_metric_rows,
        ),
        encoding="utf-8",
    )

    log(json.dumps({"output_dir": str(args.output_dir), "num_runs": len(run_reports)}, indent=2))


if __name__ == "__main__":
    main()

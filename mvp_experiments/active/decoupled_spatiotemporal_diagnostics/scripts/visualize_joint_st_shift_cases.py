#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"


@dataclass
class CaseMatch:
    condition: str
    horizon: int
    sample_index: int
    node: int
    target_value: float
    exact_pred: float
    exact_error: float
    zero_spatial_node: int
    zero_spatial_pred: float
    zero_spatial_error: float
    best_time_delta: int
    best_time_pred: float
    best_time_error: float
    best_st_delta: int
    best_st_node: int
    best_st_pred: float
    best_st_error: float
    time_shift_gain: float
    spatial_gain_at_zero: float
    st_shift_gain: float
    extra_time_gain_after_spatial: float

    def rank_score(self, rank_by: str) -> float:
        if rank_by == "abs_spatial_gain":
            return self.exact_error - self.zero_spatial_error
        if rank_by == "abs_st_gain":
            return self.exact_error - self.best_st_error
        if rank_by == "st_shift_gain":
            return self.st_shift_gain
        if rank_by == "spatial_gain_at_zero":
            return self.spatial_gain_at_zero
        return self.extra_time_gain_after_spatial * self.exact_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize concrete PatchTST-style joint spatiotemporal shift matches."
    )
    parser.add_argument("--dataset-name", default="SD")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--run", required=True, help="Model run as name=/path/to/test_results or /path/to/checkpoint.")
    parser.add_argument(
        "--comparison-run",
        action="append",
        default=[],
        help="Optional extra model run as name=/path/to/test_results to overlay exact same-node prediction.",
    )
    parser.add_argument("--adj-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["peak", "ramp"], choices=["high_volume", "peak", "ramp"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[12])
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--top-cases", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--peak-q", type=float, default=0.90)
    parser.add_argument("--condition-high-q", type=float, default=0.75)
    parser.add_argument("--condition-ramp-q", type=float, default=0.90)
    parser.add_argument("--window-before", type=int, default=36)
    parser.add_argument("--window-after", type=int, default=36)
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alignment-directed", action="store_true")
    parser.add_argument("--require-time-shift", action="store_true")
    parser.add_argument("--require-spatial-change", action="store_true")
    parser.add_argument("--require-zero-spatial-change", action="store_true")
    parser.add_argument(
        "--rank-by",
        choices=[
            "abs_extra_time_gain",
            "abs_spatial_gain",
            "abs_st_gain",
            "st_shift_gain",
            "spatial_gain_at_zero",
        ],
        default="abs_extra_time_gain",
    )
    parser.add_argument("--allow-repeated-nodes", action="store_true")
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    return (BASICTS_ROOT / "datasets" / args.dataset_name).resolve()


def dataset_desc(args: argparse.Namespace) -> dict:
    path = dataset_dir(args) / "desc.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset desc not found: {path}")
    return load_json(path)


def dataset_settings(args: argparse.Namespace) -> tuple[int, int, int, float | None]:
    desc = dataset_desc(args)
    regular = desc.get("regular_settings", {})
    num_nodes = int(desc["num_nodes"])
    output_len = int(regular["OUTPUT_LEN"])
    null_val = regular.get("NULL_VAL")
    null_val = None if null_val is None else float(null_val)
    frequency = int(desc.get("frequency (minutes)", 1))
    return num_nodes, output_len, frequency, null_val


def parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), normalize_result_dir(Path(path))
    path = normalize_result_dir(Path(value))
    return canonical_model_name(path), path


def normalize_result_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "test_results":
        return path
    return path / "test_results"


def canonical_model_name(result_dir: Path) -> str:
    parts = result_dir.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx + 1 < len(parts):
            aliases = {"STGCNChebGraphConv": "STGCN"}
            return aliases.get(parts[idx + 1], parts[idx + 1])
    return result_dir.parent.name


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
        raise ValueError(f"Cannot infer BasicTS raw memmap shape for {path}: bytes={size}, denominator={denom}")
    num_samples = size // denom
    return np.memmap(path, dtype=dtype, mode="r", shape=(num_samples, output_len, num_nodes, target_dim))


def load_saved_array(path: Path, dtype: str, output_len: int, num_nodes: int, target_dim: int) -> np.ndarray:
    arr = try_np_load(path)
    if arr is None:
        arr = infer_raw_memmap(path, dtype, output_len, num_nodes, target_dim)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [samples, horizon, nodes], got {arr.shape} at {path}")
    if arr.shape[1] != output_len or arr.shape[2] != num_nodes:
        raise ValueError(f"Shape mismatch at {path}: got {arr.shape}, expected [*, {output_len}, {num_nodes}]")
    return arr


def load_adjacency(path: Path) -> np.ndarray:
    with path.expanduser().open("rb") as fp:
        obj = pickle.load(fp)
    if isinstance(obj, dict):
        for key in ("adj_mx", "adj", "matrix"):
            if key in obj:
                return np.asarray(obj[key], dtype=np.float32)
    if isinstance(obj, (tuple, list)) and len(obj) >= 3:
        return np.asarray(obj[-1], dtype=np.float32)
    return np.asarray(obj, dtype=np.float32)


def reach_lists(adj: np.ndarray, directed: bool) -> list[np.ndarray]:
    base = np.asarray(adj) > 0
    if not directed:
        base = base | base.T
    return [
        np.asarray(sorted(set(np.flatnonzero(base[node]).tolist()) | {node}), dtype=np.int64)
        for node in range(base.shape[0])
    ]


def valid_target_mask(target: np.ndarray, null_val: float | None) -> np.ndarray:
    valid = np.isfinite(target)
    if null_val is not None and np.isfinite(null_val):
        valid &= ~np.isclose(target, null_val)
    return valid


def quantile_condition_mask(target: np.ndarray, base_mask: np.ndarray, q: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(np.where(base_mask, target, np.nan), q, axis=0).astype(np.float32)
    return base_mask & np.isfinite(thresholds.reshape(1, -1)) & (target >= thresholds.reshape(1, -1))


def ramp_condition_mask(target: np.ndarray, base_mask: np.ndarray, q: float) -> np.ndarray:
    ramp = np.full_like(target, np.nan, dtype=np.float32)
    ramp_valid = np.zeros_like(base_mask, dtype=bool)
    ramp[1:] = np.abs(target[1:] - target[:-1])
    ramp_valid[1:] = base_mask[1:] & base_mask[:-1] & np.isfinite(ramp[1:])
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(np.where(ramp_valid, ramp, np.nan), q, axis=0).astype(np.float32)
    return ramp_valid & np.isfinite(thresholds.reshape(1, -1)) & (ramp >= thresholds.reshape(1, -1))


def condition_masks(target: np.ndarray, base_mask: np.ndarray, peak_q: float, high_q: float, ramp_q: float) -> dict[str, np.ndarray]:
    return {
        "high_volume": quantile_condition_mask(target, base_mask, high_q),
        "peak": quantile_condition_mask(target, base_mask, peak_q),
        "ramp": ramp_condition_mask(target, base_mask, ramp_q),
    }


def top_pool(mask: np.ndarray, score: np.ndarray, pool_size: int) -> tuple[np.ndarray, np.ndarray]:
    flat_score = np.where(mask & np.isfinite(score), score, -np.inf).reshape(-1)
    valid = np.flatnonzero(np.isfinite(flat_score) & (flat_score > -np.inf))
    if valid.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if valid.size > pool_size:
        local = np.argpartition(flat_score[valid], -pool_size)[-pool_size:]
        valid = valid[local]
    valid = valid[np.argsort(flat_score[valid])[::-1]]
    return np.divmod(valid, mask.shape[1])


def safe_gain(exact_error: float, improved_error: float) -> float:
    if exact_error <= 0 or not np.isfinite(exact_error) or not np.isfinite(improved_error):
        return float("nan")
    return (exact_error - improved_error) / exact_error


def best_case_match(
    condition: str,
    horizon: int,
    sample: int,
    node: int,
    pred_h: np.ndarray,
    target_h: np.ndarray,
    neighbors: np.ndarray,
    max_shift: int,
) -> CaseMatch | None:
    target_value = float(target_h[sample, node])
    exact_pred = float(pred_h[sample, node])
    if not np.isfinite(target_value) or not np.isfinite(exact_pred):
        return None
    exact_error = abs(exact_pred - target_value)
    if exact_error <= 1e-8:
        return None

    best_time = (0, exact_pred, exact_error)
    best_st = (0, node, exact_pred, exact_error)
    zero_spatial = (node, exact_pred, exact_error)
    num_samples = pred_h.shape[0]

    for delta in range(-max_shift, max_shift + 1):
        pred_sample = sample + delta
        if pred_sample < 0 or pred_sample >= num_samples:
            continue
        same_node_pred = float(pred_h[pred_sample, node])
        if np.isfinite(same_node_pred):
            err = abs(same_node_pred - target_value)
            if err < best_time[2] or (err == best_time[2] and abs(delta) < abs(best_time[0])):
                best_time = (delta, same_node_pred, err)

        candidate_pred = np.asarray(pred_h[pred_sample, neighbors], dtype=np.float32)
        finite = np.isfinite(candidate_pred)
        if not np.any(finite):
            continue
        candidate_err = np.abs(candidate_pred - target_value)
        candidate_err = np.where(finite, candidate_err, np.inf)
        best_idx = int(np.argmin(candidate_err))
        err = float(candidate_err[best_idx])
        best_node = int(neighbors[best_idx])
        best_pred = float(candidate_pred[best_idx])
        if delta == 0 and err < zero_spatial[2]:
            zero_spatial = (best_node, best_pred, err)
        if err < best_st[3] or (err == best_st[3] and abs(delta) < abs(best_st[0])):
            best_st = (delta, best_node, best_pred, err)

    time_gain = safe_gain(exact_error, best_time[2])
    spatial_gain = safe_gain(exact_error, zero_spatial[2])
    st_gain = safe_gain(exact_error, best_st[3])
    extra_time = st_gain - spatial_gain if np.isfinite(st_gain) and np.isfinite(spatial_gain) else float("nan")
    return CaseMatch(
        condition=condition,
        horizon=horizon,
        sample_index=sample,
        node=node,
        target_value=target_value,
        exact_pred=exact_pred,
        exact_error=exact_error,
        zero_spatial_node=zero_spatial[0],
        zero_spatial_pred=zero_spatial[1],
        zero_spatial_error=zero_spatial[2],
        best_time_delta=best_time[0],
        best_time_pred=best_time[1],
        best_time_error=best_time[2],
        best_st_delta=best_st[0],
        best_st_node=best_st[1],
        best_st_pred=best_st[2],
        best_st_error=best_st[3],
        time_shift_gain=time_gain,
        spatial_gain_at_zero=spatial_gain,
        st_shift_gain=st_gain,
        extra_time_gain_after_spatial=extra_time,
    )


def case_to_row(case: CaseMatch, frequency: int, model_name: str) -> dict:
    row = {
        "model": model_name,
        "condition": case.condition,
        "horizon": case.horizon,
        "sample_index": case.sample_index,
        "node": case.node,
        "target_value": case.target_value,
        "exact_pred": case.exact_pred,
        "exact_error": case.exact_error,
        "zero_spatial_node": case.zero_spatial_node,
        "zero_spatial_pred": case.zero_spatial_pred,
        "zero_spatial_error": case.zero_spatial_error,
        "best_time_delta_steps": case.best_time_delta,
        "best_time_delta_minutes": case.best_time_delta * frequency,
        "best_time_pred": case.best_time_pred,
        "best_time_error": case.best_time_error,
        "best_st_delta_steps": case.best_st_delta,
        "best_st_delta_minutes": case.best_st_delta * frequency,
        "best_st_node": case.best_st_node,
        "best_st_pred": case.best_st_pred,
        "best_st_error": case.best_st_error,
        "time_shift_gain": case.time_shift_gain,
        "spatial_gain_at_zero": case.spatial_gain_at_zero,
        "st_shift_gain": case.st_shift_gain,
        "extra_time_gain_after_spatial": case.extra_time_gain_after_spatial,
        "abs_ST_gain_MAE": case.exact_error - case.best_st_error,
        "abs_spatial_gain_MAE_at0": case.exact_error - case.zero_spatial_error,
        "abs_extra_time_gain_MAE": case.zero_spatial_error - case.best_st_error,
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shifted_series(pred_h: np.ndarray, node: int, sample_start: int, sample_end: int, delta: int) -> np.ndarray:
    values = np.full(sample_end - sample_start, np.nan, dtype=np.float32)
    num_samples = pred_h.shape[0]
    for out_idx, sample in enumerate(range(sample_start, sample_end)):
        pred_sample = sample + delta
        if 0 <= pred_sample < num_samples:
            values[out_idx] = pred_h[pred_sample, node]
    return values


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> list[Path]:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    paths = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    return paths


def plot_case(
    case: CaseMatch,
    model_name: str,
    pred_h: np.ndarray,
    target_h: np.ndarray,
    comparison_pred_by_name: dict[str, np.ndarray],
    output_dir: Path,
    window_before: int,
    window_after: int,
    frequency: int,
    plot_format: str,
) -> list[str]:
    sample_start = max(0, case.sample_index - window_before)
    sample_end = min(pred_h.shape[0], case.sample_index + window_after + 1)
    x = np.arange(sample_start, sample_end)
    center_x = case.sample_index

    target_node = np.asarray(target_h[sample_start:sample_end, case.node], dtype=np.float32)
    exact_pred = np.asarray(pred_h[sample_start:sample_end, case.node], dtype=np.float32)
    time_shift = shifted_series(pred_h, case.node, sample_start, sample_end, case.best_time_delta)
    st_shift = shifted_series(pred_h, case.best_st_node, sample_start, sample_end, case.best_st_delta)
    neighbor_target = np.asarray(target_h[sample_start:sample_end, case.best_st_node], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, target_node, color="black", linewidth=2.3, label=f"GT node {case.node}")
    ax.plot(x, exact_pred, color="#d62728", linewidth=1.8, alpha=0.86, label=f"{model_name} exact node {case.node}")
    ax.plot(
        x,
        time_shift,
        color="#1f77b4",
        linewidth=1.9,
        linestyle="--",
        alpha=0.92,
        label=f"time-shift same node, dt={case.best_time_delta}",
    )
    ax.plot(
        x,
        st_shift,
        color="#2ca02c",
        linewidth=1.9,
        linestyle="-.",
        alpha=0.92,
        label=f"ST match node {case.best_st_node}, dt={case.best_st_delta}",
    )
    comparison_colors = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for model_idx, (comparison_name, comparison_pred_h) in enumerate(comparison_pred_by_name.items()):
        comparison_exact = np.asarray(comparison_pred_h[sample_start:sample_end, case.node], dtype=np.float32)
        ax.plot(
            x,
            comparison_exact,
            color=comparison_colors[model_idx % len(comparison_colors)],
            linewidth=1.55,
            alpha=0.78,
            label=f"{comparison_name} exact node {case.node}",
        )
    if case.best_st_node != case.node:
        ax.plot(
            x,
            neighbor_target,
            color="#7f7f7f",
            linewidth=1.3,
            linestyle=":",
            alpha=0.82,
            label=f"GT matched node {case.best_st_node}",
        )
    ax.axvline(center_x, color="#444444", linewidth=1.0, alpha=0.55)
    ax.scatter([center_x], [case.target_value], color="black", s=32, zorder=5)
    ax.scatter([center_x], [case.exact_pred], color="#d62728", s=28, zorder=5)
    ax.scatter([center_x], [case.best_st_pred], color="#2ca02c", s=30, zorder=5)
    title = (
        f"{model_name} {case.condition} case: node {case.node}, H{case.horizon}, "
        f"sample {case.sample_index}, STGain={case.st_shift_gain:.3f}, ExtraTime={case.extra_time_gain_after_spatial:.3f}"
    )
    ax.set_title(title)
    ax.set_xlabel("test sample index")
    ax.set_ylabel("traffic flow")
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()

    stem = (
        f"joint_st_case_{model_name}_{case.condition}_node{case.node}_"
        f"match{case.best_st_node}_h{case.horizon}_sample{case.sample_index}"
    )
    written = [str(path) for path in save_figure(fig, output_dir, stem, plot_format)]
    plt.close(fig)

    rows = []
    for idx, sample in enumerate(range(sample_start, sample_end)):
        rows.append(
            {
                "sample_index": sample,
                "minutes_from_case": (sample - case.sample_index) * frequency,
                "target_node": float(target_node[idx]),
                "exact_pred_node": float(exact_pred[idx]),
                "time_shift_pred_same_node": float(time_shift[idx]) if np.isfinite(time_shift[idx]) else "",
                "st_shift_pred_matched_node": float(st_shift[idx]) if np.isfinite(st_shift[idx]) else "",
                "target_matched_node": float(neighbor_target[idx]) if np.isfinite(neighbor_target[idx]) else "",
                **{
                    f"{comparison_name}_exact_pred_node": (
                        float(comparison_pred_h[sample, case.node])
                        if 0 <= sample < comparison_pred_h.shape[0] and np.isfinite(comparison_pred_h[sample, case.node])
                        else ""
                    )
                    for comparison_name, comparison_pred_h in comparison_pred_by_name.items()
                },
            }
        )
    csv_path = output_dir / f"{stem}.csv"
    write_csv(csv_path, rows)
    written.append(str(csv_path))
    return written


def main() -> int:
    args = parse_args()
    model_name, run_dir = parse_run(args.run)
    comparison_runs = [parse_run(item) for item in args.comparison_run]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes, output_len, frequency, null_val = dataset_settings(args)
    horizons = sorted({h for h in args.horizons if 1 <= h <= output_len})
    if not horizons:
        raise ValueError(f"No valid horizons in {args.horizons}; output_len={output_len}")

    adj = load_adjacency(args.adj_path)
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}")
    neighbors_by_node = reach_lists(adj, directed=args.alignment_directed)
    pred = load_saved_array(run_dir / "predictions.npy", args.dtype, output_len, num_nodes, args.target_dim)
    target = load_saved_array(run_dir / "targets.npy", args.dtype, output_len, num_nodes, args.target_dim)
    comparison_pred_by_name_full = {
        name: load_saved_array(result_dir / "predictions.npy", args.dtype, output_len, num_nodes, args.target_dim)
        for name, result_dir in comparison_runs
    }

    cases: list[CaseMatch] = []
    for horizon in horizons:
        h_idx = horizon - 1
        pred_h = np.asarray(pred[:, h_idx, :], dtype=np.float32)
        target_h = np.asarray(target[:, h_idx, :], dtype=np.float32)
        base_mask = valid_target_mask(target_h, null_val)
        exact_error = np.abs(pred_h - target_h)
        masks = condition_masks(target_h, base_mask, args.peak_q, args.condition_high_q, args.condition_ramp_q)
        for condition in args.conditions:
            sample_idx, node_idx = top_pool(masks[condition], exact_error, args.pool_size)
            for sample, node in zip(sample_idx.tolist(), node_idx.tolist()):
                case = best_case_match(
                    condition,
                    horizon,
                    int(sample),
                    int(node),
                    pred_h,
                    target_h,
                    neighbors_by_node[int(node)],
                    int(args.max_shift),
                )
                if case is None:
                    continue
                if args.require_time_shift and case.best_st_delta == 0:
                    continue
                if args.require_spatial_change and case.best_st_node == case.node:
                    continue
                if args.require_zero_spatial_change and case.zero_spatial_node == case.node:
                    continue
                if not np.isfinite(case.extra_time_gain_after_spatial):
                    continue
                cases.append(case)

    cases.sort(key=lambda item: (item.rank_score(args.rank_by), item.exact_error), reverse=True)
    selected: list[CaseMatch] = []
    used_nodes: set[int] = set()
    for case in cases:
        if not args.allow_repeated_nodes and case.node in used_nodes:
            continue
        selected.append(case)
        used_nodes.add(case.node)
        if len(selected) >= args.top_cases:
            break

    if len(selected) < args.top_cases and not args.allow_repeated_nodes:
        for case in cases:
            if case in selected:
                continue
            selected.append(case)
            if len(selected) >= args.top_cases:
                break

    summary_rows = [case_to_row(case, frequency, model_name) for case in selected]
    write_csv(output_dir / "joint_st_case_summary.csv", summary_rows)
    written = [str(output_dir / "joint_st_case_summary.csv")]

    for case in selected:
        pred_h = np.asarray(pred[:, case.horizon - 1, :], dtype=np.float32)
        target_h = np.asarray(target[:, case.horizon - 1, :], dtype=np.float32)
        comparison_pred_by_name = {
            name: np.asarray(comparison_pred[:, case.horizon - 1, :], dtype=np.float32)
            for name, comparison_pred in comparison_pred_by_name_full.items()
        }
        written.extend(
            plot_case(
                case,
                model_name,
                pred_h,
                target_h,
                comparison_pred_by_name,
                output_dir,
                args.window_before,
                args.window_after,
                frequency,
                args.plot_format,
            )
        )

    manifest = {
        "dataset_name": args.dataset_name,
        "run": {"name": model_name, "result_dir": str(run_dir)},
        "comparison_runs": [{"name": name, "result_dir": str(result_dir)} for name, result_dir in comparison_runs],
        "adj_path": str(args.adj_path.expanduser().resolve()),
        "alignment_directed": bool(args.alignment_directed),
        "conditions": args.conditions,
        "horizons": horizons,
        "max_shift": args.max_shift,
        "top_cases": args.top_cases,
        "pool_size": args.pool_size,
        "require_time_shift": bool(args.require_time_shift),
        "require_spatial_change": bool(args.require_spatial_change),
        "require_zero_spatial_change": bool(args.require_zero_spatial_change),
        "rank_by": args.rank_by,
        "frequency_minutes": frequency,
        "written": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_cases": len(selected)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

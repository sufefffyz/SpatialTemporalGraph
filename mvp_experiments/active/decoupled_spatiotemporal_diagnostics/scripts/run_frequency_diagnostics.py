#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--moving-window", type=int, default=12)
    parser.add_argument("--peak-q", type=float, default=0.90)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12], help="1-based horizons.")
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


def load_dataset_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_nodes is not None and args.output_len is not None:
        return int(args.num_nodes), int(args.output_len)
    desc_path = dataset_dir(args) / "desc.json"
    if not desc_path.exists():
        missing = []
        if args.num_nodes is None:
            missing.append("--num-nodes")
        if args.output_len is None:
            missing.append("--output-len")
        raise FileNotFoundError(
            f"Dataset desc not found at {desc_path}; provide {' and '.join(missing)}."
        )
    desc = load_json(desc_path)
    num_nodes = int(args.num_nodes or desc["num_nodes"])
    output_len = int(args.output_len or desc["regular_settings"]["OUTPUT_LEN"])
    return num_nodes, output_len


def normalize_result_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "test_results":
        return path
    return path / "test_results"


def discover_runs(args: argparse.Namespace) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen: set[Path] = set()

    for item in args.run:
        name, path = parse_key_path(item)
        result_dir = normalize_result_dir(path)
        if result_dir in seen:
            continue
        seen.add(result_dir)
        runs.append(RunSpec(name=name or result_dir.parent.name, result_dir=result_dir))

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
            name = result_dir.parent.name
            if name in {"test_results", ""}:
                name = result_dir.parent.parent.name
            runs.append(RunSpec(name=name, result_dir=result_dir))

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


def metric_row(system: str, horizon: int, component: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float | int | str]:
    return {
        "system": system,
        "horizon": horizon,
        "component": component,
        "MAE": masked_mae(pred, target),
        "RMSE": masked_rmse(pred, target),
        "WAPE": masked_wape(pred, target),
        "W1": wasserstein_1d(pred, target),
    }


def quantile_peak_mask(target: np.ndarray, q: float) -> np.ndarray:
    thresholds = np.nanquantile(target, q, axis=0).astype(np.float32)
    return target >= thresholds.reshape(1, -1)


def peak_rows(system: str, horizon: int, pred: np.ndarray, target: np.ndarray, q: float) -> list[dict[str, float | int | str]]:
    peak = quantile_peak_mask(target, q)
    normal = ~peak
    rows = []
    for split_name, mask in [("normal", normal), (f"peak_q{q:.2f}", peak)]:
        rows.append(
            {
                "system": system,
                "horizon": horizon,
                "window_type": split_name,
                "MAE": masked_mae(pred, target, mask),
                "RMSE": masked_rmse(pred, target, mask),
                "WAPE": masked_wape(pred, target, mask),
                "W1": wasserstein_1d(pred, target, mask),
                "valid_count": int(np.sum(np.isfinite(pred) & np.isfinite(target) & mask)),
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


def spatial_rows(
    system: str,
    horizon: int,
    component: str,
    residual: np.ndarray,
    edges: np.ndarray,
    adj: np.ndarray,
) -> dict[str, float | int | str]:
    if len(edges) == 0:
        return {
            "system": system,
            "horizon": horizon,
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


def build_markdown(report: dict, component_rows: list[dict], peak_metric_rows: list[dict], spatial_metric_rows: list[dict]) -> str:
    lines = [
        "# Decoupled Spatiotemporal Diagnostic Summary",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Runs analyzed: {report['num_runs']}",
        f"- Moving-average window: {report['moving_window']}",
        f"- Horizons: {', '.join(str(h) for h in report['horizons'])}",
        "",
        "## Runs",
        "",
    ]
    for run in report["runs"]:
        lines.append(f"- `{run['name']}`: `{run['result_dir']}` shape={run.get('shape', 'unknown')}")

    lines.extend(["", "## Component Metrics", ""])
    if component_rows:
        lines.append("| System | Horizon | Component | MAE | RMSE | WAPE | W1 |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for row in component_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['component']} | "
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

    if spatial_metric_rows:
        lines.extend(["", "## Spatial Residual Metrics", ""])
        lines.append("| System | Horizon | Component | Edge Corr | Dirichlet | Edges |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for row in spatial_metric_rows[:80]:
            lines.append(
                f"| {row['system']} | {row['horizon']} | {row['component']} | "
                f"{row['edge_residual_corr']:.4g} | {row['residual_dirichlet']:.4g} | {row['num_edges']} |"
            )
    else:
        lines.extend(["", "## Spatial Residual Metrics", "", "Skipped; no matching adjacency was provided."])

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes, output_len = load_dataset_shape(args)
    horizons = sorted({h for h in args.horizons if 1 <= h <= output_len})
    if not horizons:
        raise ValueError(f"No valid horizons in {args.horizons}; output_len={output_len}")

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

    component_rows: list[dict] = []
    peak_metric_rows: list[dict] = []
    distribution_rows: list[dict] = []
    spatial_metric_rows: list[dict] = []
    run_reports: list[dict] = []

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

        for horizon in horizons:
            h_idx = horizon - 1
            pred_h = pred[:, h_idx, :]
            target_h = target[:, h_idx, :]

            pred_low = centered_moving_average(pred_h, args.moving_window)
            target_low = centered_moving_average(target_h, args.moving_window)
            pred_high = pred_h - pred_low
            target_high = target_h - target_low

            components = {
                "full": (pred_h, target_h),
                "low": (pred_low, target_low),
                "high": (pred_high, target_high),
            }
            for component, (pred_component, target_component) in components.items():
                row = metric_row(run.name, horizon, component, pred_component, target_component)
                component_rows.append(row)
                distribution_rows.append(
                    {
                        "system": run.name,
                        "horizon": horizon,
                        "component": component,
                        "W1": row["W1"],
                    }
                )

            peak_metric_rows.extend(peak_rows(run.name, horizon, pred_h, target_h, args.peak_q))

            if adj is not None:
                residual_components = {
                    "full": target_h - pred_h,
                    "low": target_low - pred_low,
                    "high": target_high - pred_high,
                }
                for component, residual in residual_components.items():
                    spatial_metric_rows.append(spatial_rows(run.name, horizon, component, residual, edges, adj))

    report = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir(args)),
        "num_nodes": num_nodes,
        "output_len": output_len,
        "horizons": horizons,
        "moving_window": args.moving_window,
        "peak_q": args.peak_q,
        "num_runs": len(run_reports),
        "runs": run_reports,
    }

    write_csv(args.output_dir / "component_metrics.csv", component_rows)
    write_csv(args.output_dir / "peak_window_metrics.csv", peak_metric_rows)
    write_csv(args.output_dir / "distribution_metrics.csv", distribution_rows)
    write_csv(args.output_dir / "spatial_residual_metrics.csv", spatial_metric_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    (args.output_dir / "diagnostic_summary.md").write_text(
        build_markdown(report, component_rows, peak_metric_rows, spatial_metric_rows),
        encoding="utf-8",
    )

    log(json.dumps({"output_dir": str(args.output_dir), "num_runs": len(run_reports)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
EPS = 5e-5


@dataclass
class RunSpec:
    name: str
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BasicTS saved test_results across H1-H12 and draw model comparison plots."
    )
    parser.add_argument("--dataset-name", default="SD")
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
    parser.add_argument("--include", action="append", default=[], help="Keep paths containing this substring.")
    parser.add_argument("--exclude", action="append", default=[], help="Drop paths containing this substring.")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--null-val", default=None, help="Override dataset NULL_VAL. Use 'nan' for NaN.")
    parser.add_argument("--batch-size", type=int, default=64, help="Match BasicTS test batch size for RMSE/WAPE.")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def parse_run(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path).expanduser()
    return None, Path(value).expanduser()


def normalize_result_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "test_results":
        return path
    return path / "test_results"


def dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    return (BASICTS_ROOT / "datasets" / args.dataset_name).resolve()


def load_desc(args: argparse.Namespace) -> dict:
    path = dataset_dir(args) / "desc.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def dataset_shape_and_null(args: argparse.Namespace) -> tuple[int, int, float]:
    desc = load_desc(args)
    regular = desc.get("regular_settings", {})
    num_nodes = int(args.num_nodes or desc.get("num_nodes", 0))
    output_len = int(args.output_len or regular.get("OUTPUT_LEN", 0))
    if not num_nodes or not output_len:
        raise FileNotFoundError("Dataset desc is missing; provide --num-nodes and --output-len.")
    if args.null_val is None:
        null_val = regular.get("NULL_VAL", math.nan)
    elif args.null_val.lower() == "nan":
        null_val = math.nan
    else:
        null_val = float(args.null_val)
    return num_nodes, output_len, float(null_val)


def canonical_model_name(result_dir: Path) -> str:
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
        return variant_dir.parent.name
    return run_dir.name


def discover_runs(args: argparse.Namespace) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen: set[Path] = set()
    for item in args.run:
        name, path = parse_run(item)
        result_dir = normalize_result_dir(path)
        if result_dir in seen:
            continue
        seen.add(result_dir)
        runs.append(RunSpec(name=name or canonical_model_name(result_dir), result_dir=result_dir))

    for root in args.search_root:
        root = root.expanduser()
        if not root.exists():
            log(f"[warn] missing search root: {root}")
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
            runs.append(RunSpec(name=canonical_model_name(result_dir), result_dir=result_dir))

    runs.sort(key=lambda run: run.name.lower())
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
            f"Cannot infer raw BasicTS memmap shape for {path}: bytes={size}, denominator={denom}."
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
        raise ValueError(f"Expected [samples, horizon, nodes], got {arr.shape} at {path}")
    if arr.shape[1] != output_len or arr.shape[2] != num_nodes:
        raise ValueError(f"Shape mismatch at {path}: got {arr.shape}, expected [*, {output_len}, {num_nodes}]")
    return np.asarray(arr, dtype=np.float32)


def base_valid_mask(target: np.ndarray, null_val: float) -> np.ndarray:
    if math.isnan(null_val):
        return np.isfinite(target)
    return np.isfinite(target) & ~np.isclose(target, null_val, atol=EPS, rtol=0.0)


def normalized_mask(mask: np.ndarray) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    mean = float(np.mean(mask_f))
    if mean <= 0:
        return np.zeros_like(mask_f, dtype=np.float32)
    return np.nan_to_num(mask_f / mean)


def basicts_mae(pred: np.ndarray, target: np.ndarray, null_val: float) -> float:
    mask = normalized_mask(base_valid_mask(target, null_val))
    loss = np.abs(pred - target) * mask
    return float(np.mean(np.nan_to_num(loss)))


def basicts_mape(pred: np.ndarray, target: np.ndarray, null_val: float) -> float:
    mask = base_valid_mask(target, null_val) & ~np.isclose(target, 0.0, atol=EPS, rtol=0.0)
    mask = normalized_mask(mask)
    loss = np.abs((pred - target) / target) * mask
    return float(np.mean(np.nan_to_num(loss, posinf=0.0, neginf=0.0)))


def basicts_rmse(pred: np.ndarray, target: np.ndarray, null_val: float) -> float:
    mask = normalized_mask(base_valid_mask(target, null_val))
    loss = ((pred - target) ** 2) * mask
    return float(np.sqrt(np.mean(np.nan_to_num(loss))))


def basicts_wape(pred: np.ndarray, target: np.ndarray, null_val: float) -> float:
    mask = base_valid_mask(target, null_val).astype(np.float32)
    pred_masked = np.nan_to_num(pred * mask)
    target_masked = np.nan_to_num(target * mask)
    numerator = np.sum(np.abs(pred_masked - target_masked), axis=1)
    denominator = np.sum(np.abs(target_masked), axis=1) + EPS
    return float(np.mean(numerator / denominator))


def weighted_batch_metric(
    pred: np.ndarray,
    target: np.ndarray,
    null_val: float,
    batch_size: int,
    metric_func,
) -> tuple[float, int]:
    values = []
    weights = []
    n = pred.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred_b = pred[start:end]
        target_b = target[start:end]
        weight = int(np.sum(base_valid_mask(target_b, null_val)))
        if weight <= 0:
            continue
        values.append(metric_func(pred_b, target_b, null_val))
        weights.append(weight)
    if not weights:
        return float("nan"), 0
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64))), int(
        np.sum(weights)
    )


def horizon_metrics(system: str, result_dir: Path, pred: np.ndarray, target: np.ndarray, horizons: list[int], null_val: float, batch_size: int) -> list[dict]:
    rows = []
    total_count = pred.shape[0] * pred.shape[2]
    for horizon in horizons:
        h_idx = horizon - 1
        pred_h = pred[:, h_idx, :]
        target_h = target[:, h_idx, :]
        mae, valid_count = weighted_batch_metric(pred_h, target_h, null_val, batch_size, basicts_mae)
        rmse, _ = weighted_batch_metric(pred_h, target_h, null_val, batch_size, basicts_rmse)
        mape, _ = weighted_batch_metric(pred_h, target_h, null_val, batch_size, basicts_mape)
        wape, _ = weighted_batch_metric(pred_h, target_h, null_val, batch_size, basicts_wape)
        rows.append(
            {
                "system": system,
                "horizon": horizon,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "WAPE": wape,
                "valid_count": valid_count,
                "total_count": total_count,
                "result_dir": str(result_dir),
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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def build_average_rows(rows: list[dict]) -> list[dict]:
    systems = sorted({row["system"] for row in rows})
    out = []
    for system in systems:
        sub = [row for row in rows if row["system"] == system]
        out.append(
            {
                "system": system,
                "MAE": float(np.mean([row["MAE"] for row in sub])),
                "RMSE": float(np.mean([row["RMSE"] for row in sub])),
                "MAPE": float(np.mean([row["MAPE"] for row in sub])),
                "WAPE": float(np.mean([row["WAPE"] for row in sub])),
            }
        )
    out.sort(key=lambda row: row["MAE"])
    return out


def pivot_metric(rows: list[dict], metric: str, ordered_systems: list[str], horizons: list[int]) -> list[dict]:
    by_key = {(row["system"], row["horizon"]): row for row in rows}
    out = []
    for system in ordered_systems:
        values = []
        item = {"system": system}
        for horizon in horizons:
            value = by_key[(system, horizon)][metric]
            item[f"H{horizon}"] = value
            values.append(value)
        item["Avg"] = float(np.mean(values))
        out.append(item)
    return out


def write_wide_csv(output_dir: Path, rows: list[dict], averages: list[dict], horizons: list[int]) -> None:
    ordered = [row["system"] for row in averages]
    for metric in ["MAE", "RMSE", "MAPE", "WAPE"]:
        wide = pivot_metric(rows, metric, ordered, horizons)
        write_csv(output_dir / f"horizon_{metric.lower()}_wide.csv", wide)


def markdown_metric_table(rows: list[dict], metric: str, ordered_systems: list[str], horizons: list[int]) -> list[str]:
    wide = pivot_metric(rows, metric, ordered_systems, horizons)
    header = ["System", *[f"H{h}" for h in horizons], "Avg"]
    lines = [
        f"### {metric}",
        "",
        "| " + " | ".join(header) + " |",
        "|---" + "|---:" * (len(header) - 1) + "|",
    ]
    for row in wide:
        values = [row["system"], *[fmt(row[f"H{h}"]) for h in horizons], fmt(row["Avg"])]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def write_markdown(path: Path, rows: list[dict], averages: list[dict], horizons: list[int], report: dict) -> None:
    ordered = [row["system"] for row in averages]
    lines = [
        "# SD Baseline H1-H12 Test Metrics",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Runs: {report['num_runs']}",
        f"- Horizons: {', '.join(f'H{h}' for h in horizons)}",
        f"- Metric style: BasicTS-style masked metrics, `NULL_VAL={report['null_val']}`, batch size {report['batch_size']}.",
        "",
        "## Average Across H1-H12",
        "",
        "| System | MAE | RMSE | MAPE | WAPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in averages:
        lines.append(
            f"| {row['system']} | {fmt(row['MAE'])} | {fmt(row['RMSE'])} | {fmt(row['MAPE'])} | {fmt(row['WAPE'])} |"
        )
    lines.extend(["", "## Horizon Tables", ""])
    for metric in ["MAE", "RMSE", "MAPE", "WAPE"]:
        lines.extend(markdown_metric_table(rows, metric, ordered, horizons))
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> None:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    for suffix in formats:
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.04)


def plot_lines(rows: list[dict], averages: list[dict], horizons: list[int], output_dir: Path, plot_format: str) -> None:
    ordered = [row["system"] for row in averages]
    colors = plt.cm.tab10.colors
    for metric in ["MAE", "RMSE", "MAPE", "WAPE"]:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        by_key = {(row["system"], row["horizon"]): row for row in rows}
        for idx, system in enumerate(ordered):
            values = [by_key[(system, horizon)][metric] for horizon in horizons]
            ax.plot(horizons, values, marker="o", linewidth=1.8, markersize=3.5, label=system, color=colors[idx % len(colors)])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(metric)
        ax.set_xticks(horizons)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        save_figure(fig, output_dir, f"{metric.lower()}_by_horizon", plot_format)
        plt.close(fig)


def plot_summary_bar(averages: list[dict], output_dir: Path, plot_format: str) -> None:
    systems = [row["system"] for row in averages]
    metrics = ["MAE", "RMSE", "MAPE"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(systems)))
    for ax, metric in zip(axes, metrics):
        values = [row[metric] for row in averages]
        ax.barh(systems, values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.grid(True, axis="x", alpha=0.25)
        if metric != "MAE":
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "average_metric_bars", plot_format)
    plt.close(fig)


def plot_heatmap(rows: list[dict], averages: list[dict], horizons: list[int], output_dir: Path, plot_format: str) -> None:
    systems = [row["system"] for row in averages]
    metrics = ["MAE", "RMSE", "MAPE", "WAPE"]
    by_key = {(row["system"], row["horizon"]): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True, sharey=True)
    for ax, metric in zip(axes.ravel(), metrics):
        matrix = np.asarray([[by_key[(system, horizon)][metric] for horizon in horizons] for system in systems], dtype=float)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(horizons)), [f"H{h}" for h in horizons], rotation=45)
        ax.set_yticks(np.arange(len(systems)), systems)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "horizon_metric_heatmaps", plot_format)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes, output_len, null_val = dataset_shape_and_null(args)
    horizons = sorted({h for h in args.horizons if 1 <= h <= output_len})
    if not horizons:
        raise ValueError(f"No valid horizons from {args.horizons}; output_len={output_len}.")
    runs = discover_runs(args)
    if not runs:
        raise FileNotFoundError("No BasicTS test_results directories were found.")

    rows: list[dict] = []
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
            raise ValueError(f"Shape mismatch for {run.name}: {pred.shape} vs {target.shape}")
        rows.extend(horizon_metrics(run.name, run.result_dir, pred, target, horizons, null_val, args.batch_size))
        run_reports.append({"name": run.name, "result_dir": str(run.result_dir), "shape": list(pred.shape)})

    averages = build_average_rows(rows)
    report = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir(args)),
        "num_runs": len(run_reports),
        "runs": run_reports,
        "horizons": horizons,
        "null_val": "nan" if math.isnan(null_val) else null_val,
        "batch_size": args.batch_size,
    }

    write_csv(args.output_dir / "horizon_test_metrics.csv", rows)
    write_csv(args.output_dir / "average_test_metrics.csv", averages)
    write_wide_csv(args.output_dir, rows, averages, horizons)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "horizon_test_metrics.md", rows, averages, horizons, report)
    plot_lines(rows, averages, horizons, args.output_dir, args.plot_format)
    plot_summary_bar(averages, args.output_dir, args.plot_format)
    plot_heatmap(rows, averages, horizons, args.output_dir, args.plot_format)
    log(json.dumps({"output_dir": str(args.output_dir), "num_runs": len(run_reports)}, indent=2))


if __name__ == "__main__":
    main()

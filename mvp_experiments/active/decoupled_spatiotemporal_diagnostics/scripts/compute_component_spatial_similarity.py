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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
COMPONENTS = ["full", "low", "high"]


@dataclass
class RunSpec:
    name: str
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure graph-spatial similarity of full/low/high target or prediction signals."
    )
    parser.add_argument("--dataset-name", default="SD")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="Run as name=/path/to/ckpt_or_test_results.")
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--models", nargs="+", default=[], help="Optional prediction model labels to keep.")
    parser.add_argument("--include-predictions", action="store_true", help="Also measure model prediction components.")
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument(
        "--decomp-methods",
        nargs="+",
        default=["moving_average", "fft_lowpass"],
        choices=["moving_average", "fft_lowpass"],
    )
    parser.add_argument("--moving-window", type=int, default=12)
    parser.add_argument("--fft-cutoff-period", type=float, default=None)
    parser.add_argument("--adj-path", type=Path, required=True)
    parser.add_argument("--max-edge-pairs", type=int, default=2000)
    parser.add_argument("--random-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    return parser.parse_args()


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
        raise FileNotFoundError(f"Dataset desc not found at {desc_path}; provide --num-nodes and --output-len.")
    desc = load_json(desc_path)
    return int(args.num_nodes or desc["num_nodes"]), int(args.output_len or desc["regular_settings"]["OUTPUT_LEN"])


def parse_run(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path).expanduser()
    return None, Path(value).expanduser()


def normalize_result_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path if path.name == "test_results" else path / "test_results"


def canonical_model_name(result_dir: Path) -> str:
    parts = result_dir.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx + 1 < len(parts):
            return {"STGCNChebGraphConv": "STGCN"}.get(parts[idx + 1], parts[idx + 1])
    return result_dir.parent.name


def discover_runs(args: argparse.Namespace) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen: set[Path] = set()
    for item in args.run:
        name, path = parse_run(item)
        result_dir = normalize_result_dir(path)
        if result_dir in seen:
            continue
        seen.add(result_dir)
        runs.append(RunSpec(name or canonical_model_name(result_dir), result_dir))

    for root in args.search_root:
        root = root.expanduser()
        if not root.exists():
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
            runs.append(RunSpec(canonical_model_name(result_dir), result_dir))

    if args.models:
        wanted = {name: idx for idx, name in enumerate(args.models)}
        runs = [run for run in runs if run.name in wanted]
        runs.sort(key=lambda run: wanted[run.name])
    else:
        runs.sort(key=lambda run: run.name.lower())
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
        raise ValueError(f"Cannot infer shape for {path}: bytes={size}, denominator={denom}")
    samples = size // denom
    return np.memmap(path, dtype=dtype, mode="r", shape=(samples, output_len, num_nodes, target_dim))


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
    return np.asarray(arr, dtype=np.float32)


def centered_moving_average(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(series, dtype=np.float32).copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(series, ((left, right), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0, dtype=np.float64)
    cumsum = np.vstack([np.zeros((1, padded.shape[1]), dtype=np.float64), cumsum])
    return ((cumsum[window:] - cumsum[:-window]) / float(window)).astype(np.float32)


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
    freqs = np.fft.rfftfreq(series.shape[0], d=1.0)
    keep = freqs <= 1.0 / float(cutoff_period) + 1e-12
    spectrum = np.fft.rfft(fill_nonfinite_for_fft(series), axis=0)
    spectrum[~keep, :] = 0.0
    low = np.fft.irfft(spectrum, n=series.shape[0], axis=0).astype(np.float32)
    low[~np.isfinite(series)] = np.nan
    return low


def decompose(series: np.ndarray, method: str, moving_window: int, fft_cutoff_period: float) -> dict[str, np.ndarray]:
    if method == "moving_average":
        low = centered_moving_average(series, moving_window)
    elif method == "fft_lowpass":
        low = fft_lowpass(series, fft_cutoff_period)
    else:
        raise ValueError(f"Unknown method: {method}")
    return {"full": series, "low": low, "high": series - low}


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
    rows, cols = np.where(np.asarray(adj) > 0)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    if len(rows) > max_edges:
        idx = np.linspace(0, len(rows) - 1, max_edges, dtype=int)
        rows, cols = rows[idx], cols[idx]
    return np.stack([rows, cols], axis=1) if len(rows) else np.empty((0, 2), dtype=np.int64)


def sample_random_pairs(num_nodes: int, num_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=num_pairs, endpoint=False)
    dst = rng.integers(0, num_nodes - 1, size=num_pairs, endpoint=False)
    dst = np.where(dst >= src, dst + 1, dst)
    return np.stack([src, dst], axis=1)


def pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return float("nan")
    xv = x[valid] - np.mean(x[valid])
    yv = y[valid] - np.mean(y[valid])
    denom = math.sqrt(float(np.sum(xv * xv) * np.sum(yv * yv)))
    return float(np.sum(xv * yv) / denom) if denom > 0 else float("nan")


def pair_metrics(series: np.ndarray, pairs: np.ndarray, adj: np.ndarray | None = None) -> tuple[float, float, int]:
    corrs = []
    diffs = []
    weights = []
    for src, dst in pairs:
        corrs.append(pair_corr(series[:, src], series[:, dst]))
        diff = series[:, src] - series[:, dst]
        valid = np.isfinite(diff)
        if np.any(valid):
            diffs.append(float(np.mean(diff[valid] * diff[valid])))
            weights.append(float(adj[src, dst]) if adj is not None else 1.0)
    finite_corrs = np.asarray([value for value in corrs if np.isfinite(value)], dtype=np.float64)
    if diffs:
        weight_arr = np.asarray(weights, dtype=np.float64)
        diff_value = float(np.average(np.asarray(diffs), weights=weight_arr)) if np.sum(weight_arr) > 0 else float(np.mean(diffs))
    else:
        diff_value = float("nan")
    return float(np.mean(finite_corrs)) if finite_corrs.size else float("nan"), diff_value, int(len(diffs))


def component_row(
    signal_name: str,
    horizon: int,
    method: str,
    component: str,
    series: np.ndarray,
    edges: np.ndarray,
    random_pairs: np.ndarray,
    adj: np.ndarray,
) -> dict:
    edge_corr, edge_diff, edge_count = pair_metrics(series, edges, adj)
    random_corr, random_diff, random_count = pair_metrics(series, random_pairs, None)
    values = series[np.isfinite(series)]
    return {
        "signal": signal_name,
        "horizon": horizon,
        "decomposition_method": method,
        "component": component,
        "edge_corr": edge_corr,
        "random_pair_corr": random_corr,
        "edge_corr_lift": edge_corr - random_corr if np.isfinite(edge_corr) and np.isfinite(random_corr) else float("nan"),
        "edge_dirichlet": edge_diff,
        "random_pair_dirichlet": random_diff,
        "dirichlet_ratio_edge_over_random": edge_diff / random_diff if random_diff and np.isfinite(edge_diff) else float("nan"),
        "component_std": float(np.std(values)) if values.size else float("nan"),
        "component_rms": float(np.sqrt(np.mean(values * values))) if values.size else float("nan"),
        "num_edge_pairs": edge_count,
        "num_random_pairs": random_count,
    }


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


def average_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["signal"], row["decomposition_method"], row["component"]), []).append(row)
    metric_keys = [
        "edge_corr",
        "random_pair_corr",
        "edge_corr_lift",
        "edge_dirichlet",
        "random_pair_dirichlet",
        "dirichlet_ratio_edge_over_random",
        "component_std",
        "component_rms",
    ]
    output = []
    for (signal, method, component), items in sorted(groups.items()):
        row = {"signal": signal, "decomposition_method": method, "component": component, "num_horizons": len(items)}
        for key in metric_keys:
            vals = np.asarray([float(item[key]) for item in items if np.isfinite(float(item[key]))], dtype=np.float64)
            row[f"avg_{key}"] = float(np.mean(vals)) if vals.size else float("nan")
        output.append(row)
    return output


def save_fig(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> None:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    for suffix in formats:
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.04)


def plot_target_summary(summary_rows: list[dict], output_dir: Path, plot_format: str) -> None:
    rows = [row for row in summary_rows if row["signal"] == "target" and row["component"] in {"low", "high"}]
    if not rows:
        return
    methods = []
    for row in rows:
        if row["decomposition_method"] not in methods:
            methods.append(row["decomposition_method"])
    components = ["low", "high"]
    labels = [f"{method}\n{component}" for method in methods for component in components]
    lookup = {(row["decomposition_method"], row["component"]): row for row in rows}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    plots = [
        ("avg_edge_corr", "Edge corr"),
        ("avg_edge_corr_lift", "Edge corr lift vs random"),
        ("avg_dirichlet_ratio_edge_over_random", "Dirichlet edge/random"),
    ]
    colors = {"low": "#1f77b4", "high": "#d62728"}
    for ax, (key, title) in zip(axes, plots):
        values = [lookup.get((method, component), {}).get(key, float("nan")) for method in methods for component in components]
        bar_colors = [colors[component] for _method in methods for component in components]
        ax.bar(np.arange(len(labels)), values, color=bar_colors)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, output_dir, "target_low_high_spatial_similarity", plot_format)
    plt.close(fig)


def fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def write_report(path: Path, summary_rows: list[dict], report: dict) -> None:
    target_rows = [
        row for row in summary_rows if row["signal"] == "target" and row["component"] in {"full", "low", "high"}
    ]
    lines = [
        "# Component Spatial Similarity Report",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Horizons: {', '.join(f'H{h}' for h in report['horizons'])}",
        f"- Decomposition methods: {', '.join(report['decomposition_methods'])}",
        f"- Edge pairs: {report['num_edge_pairs']}",
        f"- Random pairs: {report['num_random_pairs']}",
        "",
        "Spatial meaning is stronger when `edge_corr_lift > 0` and `dirichlet_ratio_edge_over_random < 1`.",
        "",
        "## Target Signal Average Across Horizons",
        "",
        "| Method | Component | Edge Corr | Random Corr | Corr Lift | Edge Dirichlet | Random Dirichlet | Edge/Random Dirichlet | Std | RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"full": 0, "low": 1, "high": 2}
    for row in sorted(target_rows, key=lambda item: (item["decomposition_method"], order.get(item["component"], 9))):
        lines.append(
            "| {method} | {component} | {edge_corr} | {random_corr} | {lift} | {edge_dir} | {random_dir} | {ratio} | {std} | {rms} |".format(
                method=row["decomposition_method"],
                component=row["component"],
                edge_corr=fmt(float(row["avg_edge_corr"])),
                random_corr=fmt(float(row["avg_random_pair_corr"])),
                lift=fmt(float(row["avg_edge_corr_lift"])),
                edge_dir=fmt(float(row["avg_edge_dirichlet"])),
                random_dir=fmt(float(row["avg_random_pair_dirichlet"])),
                ratio=fmt(float(row["avg_dirichlet_ratio_edge_over_random"])),
                std=fmt(float(row["avg_component_std"])),
                rms=fmt(float(row["avg_component_rms"])),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes, output_len = load_dataset_shape(args)
    horizons = sorted({h for h in args.horizons if 1 <= h <= output_len})
    methods = list(dict.fromkeys(args.decomp_methods))
    fft_cutoff_period = float(args.fft_cutoff_period or args.moving_window)

    runs = discover_runs(args)
    if not runs:
        raise FileNotFoundError("No BasicTS test_results directories were found.")
    adj = load_adjacency(args.adj_path)
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}")
    edges = sample_edges(adj, args.max_edge_pairs)
    random_pairs = sample_random_pairs(num_nodes, args.random_pairs, args.seed)

    target = load_saved_array(runs[0].result_dir / "targets.npy", args.dtype, output_len, num_nodes, args.target_dim)
    signals: list[tuple[str, np.ndarray]] = [("target", target)]
    if args.include_predictions:
        for run in runs:
            pred = load_saved_array(run.result_dir / "predictions.npy", args.dtype, output_len, num_nodes, args.target_dim)
            signals.append((run.name, pred))

    rows: list[dict] = []
    for signal_name, arr in signals:
        print(f"[signal] {signal_name}", flush=True)
        for horizon in horizons:
            series = arr[:, horizon - 1, :]
            for method in methods:
                components = decompose(series, method, args.moving_window, fft_cutoff_period)
                for component in COMPONENTS:
                    rows.append(component_row(signal_name, horizon, method, component, components[component], edges, random_pairs, adj))

    summary_rows = average_rows(rows)
    report = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir(args)),
        "horizons": horizons,
        "decomposition_methods": methods,
        "moving_window": args.moving_window,
        "fft_cutoff_period": fft_cutoff_period,
        "num_nodes": num_nodes,
        "output_len": output_len,
        "signals": [name for name, _arr in signals],
        "num_edge_pairs": int(len(edges)),
        "num_random_pairs": int(len(random_pairs)),
        "adj_path": str(args.adj_path),
    }

    write_csv(args.output_dir / "component_spatial_similarity_by_horizon.csv", rows)
    write_csv(args.output_dir / "component_spatial_similarity_summary.csv", summary_rows)
    (args.output_dir / "component_spatial_similarity_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_report(args.output_dir / "component_spatial_similarity_report.md", summary_rows, report)
    plot_target_summary(summary_rows, args.output_dir, args.plot_format)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "signals": len(signals)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"


@dataclass
class RunSpec:
    name: str
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a selectable BasicTS test_results time-series slice with ground truth, "
            "model predictions, and the current low-frequency component."
        )
    )
    parser.add_argument("--dataset-name", default="SD")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="Run as name=/path/to/ckpt_or_test_results.")
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--include", action="append", default=[], help="Keep discovered paths containing token.")
    parser.add_argument("--exclude", action="append", default=[], help="Drop discovered paths containing token.")
    parser.add_argument("--models", nargs="+", default=[], help="Optional model labels to keep, in display order.")
    parser.add_argument("--node", nargs="+", type=int, required=True, help="Node indices to plot.")
    parser.add_argument("--horizon", nargs="+", type=int, default=[12], help="1-based forecast horizons.")
    parser.add_argument("--start-sample", type=int, default=0, help="Start index on saved test_results sample axis.")
    parser.add_argument(
        "--start-target-index",
        type=int,
        default=None,
        help="Global dataset target time index to start from. Overrides --start-sample.",
    )
    parser.add_argument(
        "--start-datetime",
        default=None,
        help="Datetime to start from, e.g. '2024-01-02 08:00'. Requires --dataset-start-datetime.",
    )
    parser.add_argument(
        "--dataset-start-datetime",
        default=None,
        help="Datetime corresponding to global dataset index 0. Enables datetime x-axis and --start-datetime.",
    )
    parser.add_argument("--steps", type=int, default=96, help="Number of target time steps to show.")
    parser.add_argument("--low-window", type=int, default=12, help="Moving-average window for low-frequency component.")
    parser.add_argument(
        "--low-method",
        choices=["centered", "trailing"],
        default="centered",
        help="Low-frequency moving average style. Diagnostic default is centered.",
    )
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    parser.add_argument(
        "--x-axis",
        choices=["sample", "target-index", "datetime"],
        default="target-index",
        help="X-axis style. Datetime requires --dataset-start-datetime.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    return (BASICTS_ROOT / "datasets" / args.dataset_name).resolve()


def dataset_desc(args: argparse.Namespace) -> dict:
    path = dataset_dir(args) / "desc.json"
    if not path.exists():
        return {}
    return load_json(path)


def dataset_settings(args: argparse.Namespace) -> tuple[int, int, int, int, int]:
    desc = dataset_desc(args)
    regular = desc.get("regular_settings", {})
    num_nodes = int(args.num_nodes or desc.get("num_nodes", 0))
    input_len = int(regular.get("INPUT_LEN", 12))
    output_len = int(args.output_len or regular.get("OUTPUT_LEN", 0))
    frequency = int(desc.get("frequency (minutes)", 1))
    total_len = int(desc.get("num_time_steps", desc.get("shape", [0])[0] if desc else 0))
    if not num_nodes or not output_len:
        raise FileNotFoundError("Dataset desc is missing; provide --num-nodes and --output-len.")
    return num_nodes, input_len, output_len, frequency, total_len


def test_start_index(args: argparse.Namespace) -> int:
    desc = dataset_desc(args)
    regular = desc.get("regular_settings", {})
    ratio = regular.get("TRAIN_VAL_TEST_RATIO", [0.6, 0.2, 0.2])
    total_len = int(desc.get("num_time_steps", desc.get("shape", [0])[0] if desc else 0))
    if not total_len:
        return 0
    valid_len = int(total_len * float(ratio[1]))
    test_len = int(total_len * float(ratio[2]))
    train_len = total_len - valid_len - test_len
    return train_len + valid_len


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


def canonical_model_name(result_dir: Path) -> str:
    parts = result_dir.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx + 1 < len(parts):
            aliases = {"STGCNChebGraphConv": "STGCN"}
            return aliases.get(parts[idx + 1], parts[idx + 1])
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


def moving_average_1d(series: np.ndarray, window: int, method: str) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    if window <= 1:
        return series.copy()
    if method == "trailing":
        left = window - 1
        right = 0
    else:
        left = window // 2
        right = window - 1 - left
    padded = np.pad(series, (left, right), mode="edge")
    cumsum = np.cumsum(padded, dtype=np.float64)
    cumsum = np.concatenate([[0.0], cumsum])
    return ((cumsum[window:] - cumsum[:-window]) / float(window)).astype(np.float32)


def target_index_to_sample(args: argparse.Namespace, horizon: int, input_len: int, frequency: int) -> int:
    if args.start_datetime:
        if not args.dataset_start_datetime:
            raise ValueError("--start-datetime requires --dataset-start-datetime")
        start_dt = datetime.fromisoformat(args.start_datetime)
        dataset_dt = datetime.fromisoformat(args.dataset_start_datetime)
        delta_minutes = (start_dt - dataset_dt).total_seconds() / 60.0
        if delta_minutes < 0:
            raise ValueError("--start-datetime is before --dataset-start-datetime")
        target_index = int(round(delta_minutes / frequency))
    elif args.start_target_index is not None:
        target_index = int(args.start_target_index)
    else:
        return int(args.start_sample)
    return target_index - test_start_index(args) - input_len - (horizon - 1)


def x_values(
    args: argparse.Namespace,
    sample_start: int,
    steps: int,
    horizon: int,
    input_len: int,
    frequency: int,
) -> tuple[list, str]:
    sample_indices = np.arange(sample_start, sample_start + steps)
    target_indices = test_start_index(args) + sample_indices + input_len + (horizon - 1)
    if args.x_axis == "sample":
        return sample_indices.tolist(), "test sample index"
    if args.x_axis == "datetime":
        if not args.dataset_start_datetime:
            raise ValueError("--x-axis datetime requires --dataset-start-datetime")
        dataset_dt = datetime.fromisoformat(args.dataset_start_datetime)
        return [dataset_dt + timedelta(minutes=int(idx) * frequency) for idx in target_indices], "target datetime"
    return target_indices.tolist(), "target global index"


def save_series_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> list[Path]:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    paths = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes, input_len, output_len, frequency, total_len = dataset_settings(args)
    runs = discover_runs(args)
    if not runs:
        raise FileNotFoundError("No test_results runs found. Pass --run or --search-root.")

    horizons = sorted({h for h in args.horizon if 1 <= h <= output_len})
    nodes = sorted({node for node in args.node if 0 <= node < num_nodes})
    if not horizons:
        raise ValueError(f"No valid horizon in {args.horizon}; output_len={output_len}")
    if not nodes:
        raise ValueError(f"No valid node in {args.node}; num_nodes={num_nodes}")

    target_arr = load_saved_array(runs[0].result_dir / "targets.npy", args.dtype, output_len, num_nodes, args.target_dim)
    pred_by_run = {
        run.name: load_saved_array(run.result_dir / "predictions.npy", args.dtype, output_len, num_nodes, args.target_dim)
        for run in runs
    }
    num_samples = int(target_arr.shape[0])
    colors = plt.cm.tab10.colors
    written: list[str] = []

    for horizon in horizons:
        sample_start = target_index_to_sample(args, horizon, input_len, frequency)
        if sample_start < 0:
            raise ValueError(f"Selected start resolves to negative sample index {sample_start} for H{horizon}.")
        if sample_start + args.steps > num_samples:
            raise ValueError(
                f"Slice [{sample_start}, {sample_start + args.steps}) exceeds available samples {num_samples} for H{horizon}."
            )
        sample_end = sample_start + args.steps
        x, xlabel = x_values(args, sample_start, args.steps, horizon, input_len, frequency)

        for node in nodes:
            target_full = np.asarray(target_arr[:, horizon - 1, node], dtype=np.float32)
            target_low = moving_average_1d(target_full, args.low_window, args.low_method)
            target_slice = target_full[sample_start:sample_end]
            low_slice = target_low[sample_start:sample_end]

            fig, ax = plt.subplots(figsize=(11.5, 4.8))
            ax.plot(x, target_slice, color="black", linewidth=2.2, label="Ground truth")
            ax.plot(
                x,
                low_slice,
                color="black",
                linewidth=2.0,
                linestyle="--",
                alpha=0.72,
                label=f"GT low ({args.low_method}, w={args.low_window})",
            )
            csv_rows = []
            for idx in range(args.steps):
                row = {
                    "sample_index": sample_start + idx,
                    "target_global_index": test_start_index(args) + sample_start + idx + input_len + (horizon - 1),
                    "ground_truth": float(target_slice[idx]),
                    "ground_truth_low": float(low_slice[idx]),
                }
                if args.dataset_start_datetime:
                    dataset_dt = datetime.fromisoformat(args.dataset_start_datetime)
                    row["target_datetime"] = (
                        dataset_dt + timedelta(minutes=int(row["target_global_index"]) * frequency)
                    ).isoformat(sep=" ")
                csv_rows.append(row)

            for model_idx, (name, pred_arr) in enumerate(pred_by_run.items()):
                pred_slice = np.asarray(pred_arr[sample_start:sample_end, horizon - 1, node], dtype=np.float32)
                ax.plot(
                    x,
                    pred_slice,
                    linewidth=1.5,
                    alpha=0.88,
                    color=colors[model_idx % len(colors)],
                    label=name,
                )
                for idx, value in enumerate(pred_slice):
                    csv_rows[idx][name] = float(value)

            ax.set_title(
                f"{args.dataset_name} node {node} H{horizon}: samples {sample_start}-{sample_end - 1}"
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("traffic flow")
            ax.grid(True, axis="y", alpha=0.24)
            ax.legend(ncol=3, fontsize=8, frameon=False)
            fig.autofmt_xdate()
            fig.tight_layout()

            stem = f"prediction_slice_node{node}_h{horizon}_sample{sample_start}_steps{args.steps}"
            paths = save_figure(fig, output_dir, stem, args.plot_format)
            plt.close(fig)
            save_series_csv(output_dir / f"{stem}.csv", csv_rows)
            written.extend(str(path) for path in paths)
            written.append(str(output_dir / f"{stem}.csv"))
            log(f"[write] {stem}")

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir(args)),
        "runs": [{"name": run.name, "result_dir": str(run.result_dir)} for run in runs],
        "nodes": nodes,
        "horizons": horizons,
        "steps": args.steps,
        "low_window": args.low_window,
        "low_method": args.low_method,
        "test_start_index": test_start_index(args),
        "input_len": input_len,
        "output_len": output_len,
        "frequency_minutes": frequency,
        "total_len": total_len,
        "written": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(json.dumps({"output_dir": str(output_dir), "figures_or_csv": len(written)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
class RunSpec:
    name: str
    result_dir: Path


@dataclass
class PatchCandidate:
    rank: int
    model: str
    target_node: int
    source_node: int
    horizon: int
    delta: int
    patch_mae: float
    center_abs_error: float
    source_start_sample: int
    source_end_sample: int
    target_start_sample: int
    target_end_sample: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For a fixed node/time event, plot per-model top-k spatiotemporal patch references."
    )
    parser.add_argument("--dataset-name", default="SD")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--run", action="append", required=True, help="Run as name=/path/to/test_results.")
    parser.add_argument("--adj-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--patch-before", type=int, default=5)
    parser.add_argument("--patch-after", type=int, default=6)
    parser.add_argument("--context-before", type=int, default=48)
    parser.add_argument("--context-after", type=int, default=48)
    parser.add_argument(
        "--rank-by",
        choices=["center_abs_error", "patch_mae"],
        default="center_abs_error",
        help="How to pick top-k candidate curves. center_abs_error matches the original point-wise ShiftGain rule.",
    )
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alignment-directed", action="store_true")
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    return parser.parse_args()


def dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    return (BASICTS_ROOT / "datasets" / args.dataset_name).resolve()


def dataset_desc(args: argparse.Namespace) -> dict:
    path = dataset_dir(args) / "desc.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset desc not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_settings(args: argparse.Namespace) -> tuple[int, int, int]:
    desc = dataset_desc(args)
    regular = desc.get("regular_settings", {})
    return int(desc["num_nodes"]), int(regular["OUTPUT_LEN"]), int(desc.get("frequency (minutes)", 1))


def parse_run(value: str) -> RunSpec:
    if "=" in value:
        name, path = value.split("=", 1)
        return RunSpec(name.strip(), normalize_result_dir(Path(path)))
    path = normalize_result_dir(Path(value))
    return RunSpec(canonical_model_name(path), path)


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


def one_hop_neighbors(adj: np.ndarray, node: int, directed: bool) -> np.ndarray:
    base = np.asarray(adj) > 0
    if not directed:
        base = base | base.T
    neighbors = set(np.flatnonzero(base[node]).tolist()) | {node}
    return np.asarray(sorted(neighbors), dtype=np.int64)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> list[Path]:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    paths = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    return paths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(candidates: list[PatchCandidate]) -> list[dict]:
    rows = []
    for item in candidates:
        rows.append(
            {
                "rank": item.rank,
                "model": item.model,
                "target_node": item.target_node,
                "source_node": item.source_node,
                "horizon": item.horizon,
                "delta_steps": item.delta,
                "patch_mae": item.patch_mae,
                "center_abs_error": item.center_abs_error,
                "target_patch": f"[{item.target_start_sample},{item.target_end_sample})",
                "source_patch": f"[{item.source_start_sample},{item.source_end_sample})",
            }
        )
    return rows


def find_top_patches(
    model_name: str,
    pred_h: np.ndarray,
    target_h: np.ndarray,
    node: int,
    neighbors: np.ndarray,
    sample_index: int,
    horizon: int,
    max_shift: int,
    patch_before: int,
    patch_after: int,
    top_k: int,
    rank_by: str,
) -> list[PatchCandidate]:
    target_start = sample_index - patch_before
    target_end = sample_index + patch_after + 1
    if target_start < 0 or target_end > target_h.shape[0]:
        raise ValueError(f"Target patch [{target_start},{target_end}) is outside sample range [0,{target_h.shape[0]}).")
    target_patch = np.asarray(target_h[target_start:target_end, node], dtype=np.float32)
    if not np.all(np.isfinite(target_patch)):
        raise ValueError("Target patch contains non-finite values.")

    candidates: list[PatchCandidate] = []
    for delta in range(-max_shift, max_shift + 1):
        source_start = target_start + delta
        source_end = target_end + delta
        if source_start < 0 or source_end > pred_h.shape[0]:
            continue
        for source_node in neighbors.tolist():
            pred_patch = np.asarray(pred_h[source_start:source_end, source_node], dtype=np.float32)
            if not np.all(np.isfinite(pred_patch)):
                continue
            patch_mae = float(np.mean(np.abs(pred_patch - target_patch)))
            center_pred = float(pred_h[sample_index + delta, source_node]) if 0 <= sample_index + delta < pred_h.shape[0] else float("nan")
            center_target = float(target_h[sample_index, node])
            center_abs_error = abs(center_pred - center_target) if np.isfinite(center_pred) else float("nan")
            candidates.append(
                PatchCandidate(
                    rank=0,
                    model=model_name,
                    target_node=node,
                    source_node=int(source_node),
                    horizon=horizon,
                    delta=delta,
                    patch_mae=patch_mae,
                    center_abs_error=center_abs_error,
                    source_start_sample=source_start,
                    source_end_sample=source_end,
                    target_start_sample=target_start,
                    target_end_sample=target_end,
                )
            )
    if rank_by == "patch_mae":
        candidates.sort(key=lambda item: (item.patch_mae, item.center_abs_error, abs(item.delta)))
    else:
        candidates.sort(key=lambda item: (item.center_abs_error, item.patch_mae, abs(item.delta)))
    selected = candidates[:top_k]
    for idx, item in enumerate(selected, start=1):
        item.rank = idx
    return selected


def aligned_patch_series(pred_h: np.ndarray, candidate: PatchCandidate) -> tuple[np.ndarray, np.ndarray]:
    target_x = np.arange(candidate.target_start_sample, candidate.target_end_sample)
    values = np.asarray(
        pred_h[candidate.source_start_sample : candidate.source_end_sample, candidate.source_node],
        dtype=np.float32,
    )
    return target_x, values


def aligned_context_series(
    arr_h: np.ndarray,
    node: int,
    context_start: int,
    context_end: int,
    delta: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_context = np.arange(context_start, context_end)
    values = np.full(x_context.shape, np.nan, dtype=np.float32)
    for idx, sample in enumerate(x_context.tolist()):
        source_sample = sample + delta
        if 0 <= source_sample < arr_h.shape[0]:
            values[idx] = arr_h[source_sample, node]
    return x_context, values


def save_auxiliary_series(
    output_dir: Path,
    stem: str,
    x: np.ndarray,
    series: dict[str, np.ndarray],
) -> str:
    rows = []
    for idx, sample in enumerate(x.tolist()):
        row = {"target_aligned_sample_index": int(sample)}
        for name, values in series.items():
            value = float(values[idx]) if np.isfinite(values[idx]) else ""
            row[name] = value
        rows.append(row)
    path = output_dir / f"{stem}.csv"
    write_csv(path, rows)
    return str(path)


def plot_neighbor_predictions_vs_truths(
    model_name: str,
    pred_h: np.ndarray,
    target_h: np.ndarray,
    candidates: list[PatchCandidate],
    sample_index: int,
    output_dir: Path,
    context_before: int,
    context_after: int,
    plot_format: str,
) -> list[str]:
    if not candidates:
        return []
    primary = candidates[0]
    context_start = max(0, sample_index - context_before)
    context_end = min(target_h.shape[0], sample_index + context_after + 1)
    x = np.arange(context_start, context_end)

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.axvspan(primary.target_start_sample, primary.target_end_sample - 1, color="#f2e6c9", alpha=0.55)
    colors = plt.cm.tab10.colors
    series: dict[str, np.ndarray] = {}
    for idx, candidate in enumerate(candidates):
        color = colors[idx % len(colors)]
        _, neighbor_pred = aligned_context_series(pred_h, candidate.source_node, context_start, context_end, candidate.delta)
        _, neighbor_true = aligned_context_series(target_h, candidate.source_node, context_start, context_end, candidate.delta)
        label_base = f"#{candidate.rank} node {candidate.source_node}, dt={candidate.delta}"
        ax.plot(x, neighbor_pred, color=color, linewidth=1.75, alpha=0.82, label=f"pred {label_base}")
        ax.plot(x, neighbor_true, color=color, linewidth=1.55, linestyle="--", alpha=0.78, label=f"GT {label_base}")
        series[f"rank{candidate.rank}_node{candidate.source_node}_dt{candidate.delta}_prediction"] = neighbor_pred
        series[f"rank{candidate.rank}_node{candidate.source_node}_dt{candidate.delta}_ground_truth"] = neighbor_true
    ax.axvline(sample_index, color="#444444", linewidth=1.0, alpha=0.75)
    ax.set_title(
        f"{model_name}: top-{len(candidates)} source-node predictions vs source-node truths "
        f"(target {primary.target_node}, H{primary.horizon}, sample {sample_index})"
    )
    ax.set_xlabel("target-aligned test sample index")
    ax.set_ylabel("traffic flow")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(ncol=4, fontsize=7.2, frameon=False)
    fig.tight_layout()
    stem = (
        f"top8_neighbor_pred_vs_truth_{model_name}_target{primary.target_node}_"
        f"h{primary.horizon}_sample{sample_index}"
    )
    written = [str(path) for path in save_figure(fig, output_dir, stem, plot_format)]
    plt.close(fig)
    written.append(
        save_auxiliary_series(
            output_dir,
            stem,
            x,
            series,
        )
    )
    return written


def plot_target_truth_vs_neighbor_truths(
    model_name: str,
    target_h: np.ndarray,
    candidates: list[PatchCandidate],
    sample_index: int,
    output_dir: Path,
    context_before: int,
    context_after: int,
    plot_format: str,
) -> list[str]:
    if not candidates:
        return []
    primary = candidates[0]
    context_start = max(0, sample_index - context_before)
    context_end = min(target_h.shape[0], sample_index + context_after + 1)
    x = np.arange(context_start, context_end)
    target_true = np.asarray(target_h[context_start:context_end, primary.target_node], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.axvspan(primary.target_start_sample, primary.target_end_sample - 1, color="#f2e6c9", alpha=0.55)
    ax.plot(x, target_true, color="black", linewidth=2.4, label=f"GT target node {primary.target_node}")
    colors = plt.cm.tab10.colors
    series: dict[str, np.ndarray] = {"target_ground_truth": target_true}
    for idx, candidate in enumerate(candidates):
        _, neighbor_true = aligned_context_series(target_h, candidate.source_node, context_start, context_end, candidate.delta)
        label = f"#{candidate.rank} GT node {candidate.source_node}, dt={candidate.delta}"
        ax.plot(
            x,
            neighbor_true,
            color=colors[idx % len(colors)],
            linewidth=1.65,
            alpha=0.78,
            label=label,
        )
        series[f"rank{candidate.rank}_node{candidate.source_node}_dt{candidate.delta}_ground_truth"] = neighbor_true
    ax.axvline(sample_index, color="#444444", linewidth=1.0, alpha=0.75)
    ax.set_title(
        f"{model_name}: target truth vs top-{len(candidates)} source-node truths "
        f"(target {primary.target_node}, H{primary.horizon}, sample {sample_index})"
    )
    ax.set_xlabel("target-aligned test sample index")
    ax.set_ylabel("traffic flow")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout()
    stem = (
        f"top8_target_truth_vs_neighbor_truth_{model_name}_target{primary.target_node}_"
        f"h{primary.horizon}_sample{sample_index}"
    )
    written = [str(path) for path in save_figure(fig, output_dir, stem, plot_format)]
    plt.close(fig)
    written.append(
        save_auxiliary_series(
            output_dir,
            stem,
            x,
            series,
        )
    )
    return written


def plot_model_patches(
    model_name: str,
    pred_h: np.ndarray,
    target_h: np.ndarray,
    node: int,
    sample_index: int,
    candidates: list[PatchCandidate],
    output_dir: Path,
    context_before: int,
    context_after: int,
    plot_format: str,
) -> list[str]:
    context_start = max(0, sample_index - context_before)
    context_end = min(target_h.shape[0], sample_index + context_after + 1)
    target_patch_start = candidates[0].target_start_sample
    target_patch_end = candidates[0].target_end_sample
    x_context = np.arange(context_start, context_end)
    target_context = np.asarray(target_h[context_start:context_end, node], dtype=np.float32)
    exact_context = np.asarray(pred_h[context_start:context_end, node], dtype=np.float32)

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2])
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    ax.axvspan(target_patch_start, target_patch_end - 1, color="#f2e6c9", alpha=0.55, label="target patch span")
    ax.plot(x_context, target_context, color="black", linewidth=2.5, label=f"GT target node {node}")
    ax.plot(x_context, exact_context, color="#d62728", linewidth=1.8, alpha=0.86, label=f"{model_name} exact node {node}")
    colors = plt.cm.tab10.colors
    for idx, candidate in enumerate(candidates):
        x_patch, values = aligned_context_series(
            pred_h,
            candidate.source_node,
            context_start,
            context_end,
            candidate.delta,
        )
        label = f"#{candidate.rank} src node {candidate.source_node}, dt={candidate.delta}, MAE={candidate.patch_mae:.1f}"
        ax.plot(
            x_patch,
            values,
            color=colors[idx % len(colors)],
            linewidth=2.0 if idx < 3 else 1.5,
            alpha=0.9 if idx < 3 else 0.68,
            label=label,
        )
    ax.axvline(sample_index, color="#444444", linewidth=1.0, alpha=0.75)
    ax.scatter([sample_index], [target_h[sample_index, node]], color="black", s=32, zorder=5)
    ax.set_title(
        f"{model_name}: top-{len(candidates)} one-hop/time-shift patch references "
        f"for node {node}, H{candidates[0].horizon}, sample {sample_index}"
    )
    ax.set_xlabel("target-aligned test sample index")
    ax.set_ylabel("traffic flow")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(ncol=2, fontsize=8, frameon=False)

    table_rows = [
        [
            item.rank,
            item.source_node,
            item.delta,
            f"{item.patch_mae:.2f}",
            f"{item.center_abs_error:.2f}",
            f"[{item.source_start_sample},{item.source_end_sample})",
        ]
        for item in candidates
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["rank", "src node", "dt", "patch MAE", "center abs err", "source patch"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    fig.tight_layout()

    stem = f"patch_reference_{model_name}_node{node}_h{candidates[0].horizon}_sample{sample_index}"
    written = [str(path) for path in save_figure(fig, output_dir, stem, plot_format)]
    plt.close(fig)
    return written


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes, output_len, frequency = dataset_settings(args)
    if not (0 <= args.node < num_nodes):
        raise ValueError(f"--node {args.node} outside [0,{num_nodes}).")
    if not (1 <= args.horizon <= output_len):
        raise ValueError(f"--horizon {args.horizon} outside [1,{output_len}].")

    adj = load_adjacency(args.adj_path)
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}")
    neighbors = one_hop_neighbors(adj, args.node, directed=args.alignment_directed)
    runs = [parse_run(item) for item in args.run]
    written: list[str] = []
    all_candidate_rows: list[dict] = []

    for run in runs:
        pred = load_saved_array(run.result_dir / "predictions.npy", args.dtype, output_len, num_nodes, args.target_dim)
        target = load_saved_array(run.result_dir / "targets.npy", args.dtype, output_len, num_nodes, args.target_dim)
        pred_h = np.asarray(pred[:, args.horizon - 1, :], dtype=np.float32)
        target_h = np.asarray(target[:, args.horizon - 1, :], dtype=np.float32)
        candidates = find_top_patches(
            run.name,
            pred_h,
            target_h,
            args.node,
            neighbors,
            args.sample_index,
            args.horizon,
            args.max_shift,
            args.patch_before,
            args.patch_after,
            args.top_k,
            args.rank_by,
        )
        rows = candidate_rows(candidates)
        all_candidate_rows.extend(rows)
        per_model_csv = output_dir / f"patch_reference_{run.name}_node{args.node}_h{args.horizon}_sample{args.sample_index}_top{args.top_k}.csv"
        write_csv(per_model_csv, rows)
        written.append(str(per_model_csv))
        written.extend(
            plot_model_patches(
                run.name,
                pred_h,
                target_h,
                args.node,
                args.sample_index,
                candidates,
                output_dir,
                args.context_before,
                args.context_after,
                args.plot_format,
            )
        )
        if candidates:
            written.extend(
                plot_neighbor_predictions_vs_truths(
                    run.name,
                    pred_h,
                    target_h,
                    candidates,
                    args.sample_index,
                    output_dir,
                    args.context_before,
                    args.context_after,
                    args.plot_format,
                )
            )
            written.extend(
                plot_target_truth_vs_neighbor_truths(
                    run.name,
                    target_h,
                    candidates,
                    args.sample_index,
                    output_dir,
                    args.context_before,
                    args.context_after,
                    args.plot_format,
                )
            )

    write_csv(output_dir / "patch_reference_all_models_topk.csv", all_candidate_rows)
    written.append(str(output_dir / "patch_reference_all_models_topk.csv"))
    manifest = {
        "dataset_name": args.dataset_name,
        "node": args.node,
        "sample_index": args.sample_index,
        "horizon": args.horizon,
        "max_shift": args.max_shift,
        "top_k": args.top_k,
        "patch_before": args.patch_before,
        "patch_after": args.patch_after,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "rank_by": args.rank_by,
        "frequency_minutes": frequency,
        "adj_path": str(args.adj_path.expanduser().resolve()),
        "alignment_directed": bool(args.alignment_directed),
        "neighbors_count": int(neighbors.size),
        "runs": [{"name": run.name, "result_dir": str(run.result_dir)} for run in runs],
        "written": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_models": len(runs), "neighbors_count": int(neighbors.size)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

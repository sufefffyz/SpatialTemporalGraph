#!/usr/bin/env python3
"""Plot per-road MAE/RMSE diagnostics from BasicTS test_results memmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basicts-dir",
        type=Path,
        default=Path("/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS"),
    )
    parser.add_argument("--run-tag", default="xuancheng_sparse_100ep_20260525")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["XCHENG_5MIN_FLOW", "XCHENG_5MIN_STOCK"],
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Defaults to <basicts-dir>/checkpoints/SparseGraphWaveNet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/yuzhang_fei/code/SpatialTemporalGraph/"
            "mvp_experiments/active/xuancheng_cityflow_pipeline/"
            "results/road_error_diagnostics_xuancheng_sparse_100ep_20260525"
        ),
    )
    return parser.parse_args()


def load_meta(meta_path: Path) -> list[dict[str, str]]:
    with meta_path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_test_results(checkpoint_root: Path, dataset: str, run_tag: str) -> Path:
    matches = sorted(checkpoint_root.glob(f"{dataset}_*{run_tag}/*/test_results"))
    if not matches:
        raise FileNotFoundError(
            f"No test_results found for dataset={dataset}, run_tag={run_tag} "
            f"under {checkpoint_root}"
        )
    return matches[-1]


def load_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def infer_test_shape(result_dir: Path, desc: dict) -> tuple[int, int, int, int]:
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    num_nodes = int(desc["num_nodes"])
    num_features = int(desc["num_features"])
    bytes_per_float = np.dtype("float32").itemsize
    size = (result_dir / "targets.npy").stat().st_size
    denom = output_len * num_nodes * num_features * bytes_per_float
    if size % denom != 0:
        raise ValueError(f"Cannot infer sample count from {result_dir / 'targets.npy'}")
    return (size // denom, output_len, num_nodes, num_features)


def dataset_label(name: str) -> str:
    return name.replace("XCHENG_", "").replace("_", " ")


def compute_road_metrics(
    basicts_dir: Path,
    checkpoint_root: Path,
    dataset: str,
    run_tag: str,
    output_dir: Path,
) -> dict[str, np.ndarray | str | Path]:
    dataset_dir = basicts_dir / "datasets" / dataset
    desc = json.loads((dataset_dir / "desc.json").read_text())
    meta = load_meta(dataset_dir / "meta.csv")
    result_dir = find_test_results(checkpoint_root, dataset, run_tag)
    shape = infer_test_shape(result_dir, desc)
    pred = load_memmap(result_dir / "predictions.npy", shape)
    target = load_memmap(result_dir / "targets.npy", shape)

    error = np.asarray(pred - target, dtype=np.float64)
    abs_error = np.abs(error)
    road_mae = abs_error.mean(axis=(0, 1, 3))
    road_rmse = np.sqrt((error**2).mean(axis=(0, 1, 3)))
    target_arr = np.asarray(target, dtype=np.float64)
    road_target_mean = target_arr.mean(axis=(0, 1, 3))
    road_target_max = target_arr.max(axis=(0, 1, 3))
    road_target_zero_rate = (target_arr == 0).mean(axis=(0, 1, 3))
    road_abs_p95 = np.quantile(abs_error, 0.95, axis=(0, 1, 3))
    road_abs_p99 = np.quantile(abs_error, 0.99, axis=(0, 1, 3))
    road_abs_max = abs_error.max(axis=(0, 1, 3))

    csv_path = output_dir / f"{dataset.lower()}_per_road_errors.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_index",
                "road_id",
                "lane_count",
                "length_m",
                "speed_limit_kmh",
                "mae",
                "rmse",
                "abs_error_p95",
                "abs_error_p99",
                "abs_error_max",
                "target_mean",
                "target_max",
                "target_zero_rate",
            ]
        )
        for i, row in enumerate(meta):
            writer.writerow(
                [
                    i,
                    row.get("road_id", ""),
                    row.get("lane_count", ""),
                    row.get("length_m", ""),
                    row.get("speed_limit_kmh", ""),
                    road_mae[i],
                    road_rmse[i],
                    road_abs_p95[i],
                    road_abs_p99[i],
                    road_abs_max[i],
                    road_target_mean[i],
                    road_target_max[i],
                    road_target_zero_rate[i],
                ]
            )

    top_idx = np.argsort(-road_rmse)[:20]
    top_path = output_dir / f"{dataset.lower()}_top20_rmse_roads.md"
    with top_path.open("w") as f:
        f.write(f"# Top-20 RMSE Roads: {dataset}\n\n")
        f.write("| rank | node | road_id | length_m | MAE | RMSE | target_mean | target_max |\n")
        f.write("|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for rank, i in enumerate(top_idx, 1):
            row = meta[int(i)]
            f.write(
                f"| {rank} | {int(i)} | {row.get('road_id', '')} | "
                f"{float(row.get('length_m', 'nan')):.1f} | "
                f"{road_mae[i]:.4g} | {road_rmse[i]:.4g} | "
                f"{road_target_mean[i]:.4g} | {road_target_max[i]:.4g} |\n"
            )

    return {
        "dataset": dataset,
        "label": dataset_label(dataset),
        "csv_path": csv_path,
        "top_path": top_path,
        "mae": road_mae,
        "rmse": road_rmse,
        "target_mean": road_target_mean,
        "target_max": road_target_max,
        "target_zero_rate": road_target_zero_rate,
        "meta": meta,
        "shape": np.array(shape),
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_distribution(results: list[dict[str, np.ndarray | str | Path]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(5.6 * len(results), 7.0))
    if len(results) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, res in enumerate(results):
        mae = res["mae"]
        rmse = res["rmse"]
        target_max = res["target_max"]
        label = str(res["label"])

        ax = axes[0, col]
        bins = 60
        ax.hist(mae, bins=bins, alpha=0.65, label="MAE", color="#4C78A8")
        ax.hist(rmse, bins=bins, alpha=0.55, label="RMSE", color="#F58518")
        ax.set_yscale("log")
        ax.set_xlabel("Per-road error")
        ax.set_ylabel("# roads (log)")
        ax.set_title(f"{label}: road error histogram")
        ax.legend(frameon=False)

        ax = axes[1, col]
        color = np.log10(np.asarray(target_max) + 1.0)
        sc = ax.scatter(mae, rmse, c=color, s=16, alpha=0.75, cmap="viridis", linewidths=0)
        ax.set_xlabel("Per-road MAE")
        ax.set_ylabel("Per-road RMSE")
        ax.set_title(f"{label}: MAE vs. RMSE by road")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("log10(max target + 1)")

        for i in np.argsort(-rmse)[:3]:
            road_id = res["meta"][int(i)].get("road_id", str(i))
            ax.annotate(str(road_id), (mae[i], rmse[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"xuancheng_sparsegwnet_5min_per_road_error_distribution.{ext}")
    plt.close(fig)


def plot_ranked(results: list[dict[str, np.ndarray | str | Path]], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(9.0, 3.4 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        mae = np.asarray(res["mae"])
        rmse = np.asarray(res["rmse"])
        order = np.argsort(rmse)
        x = np.arange(len(order))
        ax.plot(x, rmse[order], label="RMSE", color="#F58518", linewidth=1.6)
        ax.plot(x, mae[order], label="MAE", color="#4C78A8", linewidth=1.4)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("Roads sorted by RMSE")
        ax.set_ylabel("Error (symlog)")
        ax.set_title(f"{res['label']}: per-road errors ranked by RMSE")
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"xuancheng_sparsegwnet_5min_per_road_error_ranked.{ext}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_style()
    checkpoint_root = args.checkpoint_root or args.basicts_dir / "checkpoints" / "SparseGraphWaveNet"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        compute_road_metrics(args.basicts_dir, checkpoint_root, dataset, args.run_tag, args.output_dir)
        for dataset in args.datasets
    ]
    plot_distribution(results, args.output_dir)
    plot_ranked(results, args.output_dir)
    print(f"Saved road-error diagnostics to {args.output_dir}")
    for res in results:
        print(res["dataset"], res["csv_path"], res["top_path"])


if __name__ == "__main__":
    main()

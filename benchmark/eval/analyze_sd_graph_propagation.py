#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from basicts.utils.adjacent_matrix_norm import calculate_transition_matrix

DEFAULT_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "eval" / "sd_graph_propagation_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the predictive value of GWNet-style graph propagation features on old SD."
    )
    parser.add_argument("--sd-dir", type=Path, default=DEFAULT_SD_DIR, help="Old SD dataset directory.")
    parser.add_argument("--sd-phys-dir", type=Path, default=DEFAULT_SD_PHYS_DIR, help="Old SD_phys dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV/PNG/MD.")
    parser.add_argument("--max-order", type=int, default=1, help="Maximum propagation order K. Default: 1")
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["x_only", "pfpb", "x_pfpb"],
        choices=["x_only", "pfpb", "x_pfpb"],
        help="Feature groups to analyze.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        help="Ridge alphas for linear probe selection on validation set.",
    )
    parser.add_argument("--max-train-samples", type=int, default=20000, help="Cap number of train windows for probe fitting.")
    parser.add_argument("--max-eval-samples", type=int, default=5000, help="Cap number of val/test windows for probe evaluation.")
    return parser.parse_args()


def resolve_existing_path(path: str | Path, desc: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pkl(path: Path):
    import pickle

    with path.open("rb") as fp:
        return pickle.load(fp)


def unwrap_adj(payload) -> np.ndarray:
    if isinstance(payload, tuple) and len(payload) == 3:
        payload = payload[2]
    elif isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    return np.asarray(payload, dtype=np.float32)


def strip_self_loops(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32).copy()
    np.fill_diagonal(adj, 0.0)
    return adj


def load_flow(dataset_dir: Path) -> tuple[np.ndarray, dict]:
    desc = load_json(dataset_dir / "desc.json")
    shape = tuple(desc["shape"])
    data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    flow = np.asarray(data[..., 0]).copy()
    return flow, desc


def build_graphs(sd_dir: Path, sd_phys_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    distthre_adj = strip_self_loops(unwrap_adj(load_pkl(sd_dir / "adj_mx.pkl")))
    phys_dir_adj = strip_self_loops(unwrap_adj(load_pkl(sd_phys_dir / "adj_mx.pkl")))

    graphs = {
        "distthre": distthre_adj,
        "physical_dir": phys_dir_adj,
    }
    supports = {}
    for graph_name, adj in graphs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            supports[graph_name] = [
                np.asarray(calculate_transition_matrix(adj).T, dtype=np.float32),
                np.asarray(calculate_transition_matrix(adj.T).T, dtype=np.float32),
            ]
    return graphs, supports


def build_windows(flow: np.ndarray, input_len: int, output_len: int, train_val_test_ratio: list[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    total_len = len(flow)
    valid_len = int(total_len * train_val_test_ratio[1])
    test_len = int(total_len * train_val_test_ratio[2])
    train_len = total_len - valid_len - test_len

    def make_segment(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for idx in range(len(segment) - input_len - output_len + 1):
            xs.append(segment[idx : idx + input_len])
            ys.append(segment[idx + input_len : idx + input_len + output_len])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    train_seg = flow[:train_len]
    val_seg = flow[train_len : train_len + valid_len]
    test_seg = flow[train_len + valid_len :]

    return {
        "train": make_segment(train_seg),
        "val": make_segment(val_seg),
        "test": make_segment(test_seg),
    }


def subsample_windows(x: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_samples:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_samples, dtype=int)
    return x[idx], y[idx]


def apply_support(x: np.ndarray, support: np.ndarray) -> np.ndarray:
    # x: [B, L, N]
    return np.einsum("bln,nm->blm", x, support, optimize=True)


def build_feature_blocks(x: np.ndarray, supports: list[np.ndarray], max_order: int) -> dict[str, np.ndarray]:
    blocks = {"x": x}
    pf = x
    pb = x
    for order in range(1, max_order + 1):
        pf = apply_support(pf, supports[0])
        pb = apply_support(pb, supports[1])
        blocks[f"pf_{order}"] = pf
        blocks[f"pb_{order}"] = pb
    return blocks


def combine_feature_blocks(blocks: dict[str, np.ndarray], feature_mode: str, max_order: int) -> np.ndarray:
    if feature_mode == "x_only":
        parts = [blocks["x"]]
    elif feature_mode == "pfpb":
        parts = []
        for order in range(1, max_order + 1):
            parts.extend([blocks[f"pf_{order}"], blocks[f"pb_{order}"]])
    elif feature_mode == "x_pfpb":
        parts = [blocks["x"]]
        for order in range(1, max_order + 1):
            parts.extend([blocks[f"pf_{order}"], blocks[f"pb_{order}"]])
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return np.stack(parts, axis=-1)  # [B, L, N, F]


def flatten_for_probe(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # features: [B, L, N, F], targets: [B, H, N]
    bsz, seq_len, num_nodes, num_feat = features.shape
    horizon = targets.shape[1]
    x = np.transpose(features, (0, 2, 1, 3)).reshape(bsz * num_nodes, seq_len * num_feat)
    y = np.transpose(targets, (0, 2, 1)).reshape(bsz * num_nodes, horizon)
    return x.astype(np.float32), y.astype(np.float32)


def standardize_features(train_x: np.ndarray, other_x: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_std = (train_x - mean) / std
    others_std = [(x - mean) / std for x in other_x]
    return train_std, others_std, mean, std


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = alpha * np.eye(xtx.shape[0], dtype=train_x.dtype)
    reg[-1, -1] = 0.0  # do not regularize bias
    xty = x_aug.T @ train_y
    weights = np.linalg.solve(xtx + reg, xty)
    return weights[:-1], weights[-1]


def predict_ridge(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight + bias


def compute_metrics(pred: np.ndarray, target: np.ndarray, null_val: float) -> dict[str, float]:
    if math.isnan(null_val):
        mask = ~np.isnan(target)
    else:
        mask = ~np.isclose(target, null_val, atol=5e-5, rtol=0.0)
    mape_mask = mask & ~np.isclose(target, 0.0, atol=5e-5, rtol=0.0)

    mae = np.abs(pred - target)[mask]
    rmse = ((pred - target) ** 2)[mask]
    mape = np.abs((pred - target) / target)[mape_mask]
    return {
        "MAE": float(np.mean(mae)) if mae.size else float("nan"),
        "RMSE": float(np.sqrt(np.mean(rmse))) if rmse.size else float("nan"),
        "MAPE": float(np.mean(mape)) if mape.size else float("nan"),
    }


def mean_neighbor_gap(block: np.ndarray, adj: np.ndarray) -> float:
    # block: [B, L, N]
    edges = np.argwhere(adj > 0)
    if len(edges) == 0:
        return float("nan")
    diffs = []
    for src_idx, dst_idx in edges:
        diffs.append(np.abs(block[:, :, int(src_idx)] - block[:, :, int(dst_idx)]))
    return float(np.mean(np.stack(diffs, axis=0)))


def block_statistics(block: np.ndarray, adj: np.ndarray, graph_name: str, block_name: str) -> dict[str, float | str]:
    values = block.reshape(-1)
    return {
        "graph": graph_name,
        "block": block_name,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "mean_neighbor_gap": mean_neighbor_gap(block, adj),
    }


def feature_distribution_plot(plot_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=plot_df, x="block", y="value", hue="graph")
    plt.title("Feature Value Distribution after Graph Propagation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def smoothness_plot(stats_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stats_df, x="block", y="mean_neighbor_gap", hue="graph")
    plt.title("Neighbor Difference after Graph Propagation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    sd_dir = resolve_existing_path(args.sd_dir, "SD dataset dir")
    sd_phys_dir = resolve_existing_path(args.sd_phys_dir, "SD_phys dataset dir")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    flow, desc = load_flow(sd_dir)
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    split_ratio = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    null_val = float(desc["regular_settings"].get("NULL_VAL", 0.0))

    graphs, supports = build_graphs(sd_dir, sd_phys_dir)
    windows = build_windows(flow, input_len=input_len, output_len=output_len, train_val_test_ratio=split_ratio)
    train_x, train_y = subsample_windows(*windows["train"], max_samples=args.max_train_samples)
    val_x, val_y = subsample_windows(*windows["val"], max_samples=args.max_eval_samples)
    test_x, test_y = subsample_windows(*windows["test"], max_samples=args.max_eval_samples)

    feature_stats_rows = []
    plot_rows = []
    probe_rows = []

    # X-only baseline once
    x_only_train_feat = np.expand_dims(train_x, axis=-1)
    x_only_val_feat = np.expand_dims(val_x, axis=-1)
    x_only_test_feat = np.expand_dims(test_x, axis=-1)

    for feature_mode in args.feature_modes:
        if feature_mode == "x_only":
            train_flat, train_target = flatten_for_probe(x_only_train_feat, train_y)
            val_flat, val_target = flatten_for_probe(x_only_val_feat, val_y)
            test_flat, test_target = flatten_for_probe(x_only_test_feat, test_y)

            train_std, [val_std, test_std], _, _ = standardize_features(train_flat, [val_flat, test_flat])

            best_alpha = None
            best_val_mae = float("inf")
            best_weight = None
            best_bias = None
            for alpha in args.alphas:
                weight, bias = fit_ridge(train_std, train_target, alpha)
                pred_val = predict_ridge(val_std, weight, bias)
                val_metrics = compute_metrics(pred_val, val_target, null_val)
                if val_metrics["MAE"] < best_val_mae:
                    best_val_mae = val_metrics["MAE"]
                    best_alpha = alpha
                    best_weight = weight
                    best_bias = bias

            pred_test = predict_ridge(test_std, best_weight, best_bias)
            test_metrics = compute_metrics(pred_test, test_target, null_val)
            probe_rows.append(
                {
                    "graph": "none",
                    "feature_mode": feature_mode,
                    "max_order": 0,
                    "best_alpha": best_alpha,
                    "val_MAE": best_val_mae,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )
            continue

        for graph_name, adj in graphs.items():
            train_blocks = build_feature_blocks(train_x, supports[graph_name], args.max_order)
            val_blocks = build_feature_blocks(val_x, supports[graph_name], args.max_order)
            test_blocks = build_feature_blocks(test_x, supports[graph_name], args.max_order)

            for block_name, block_value in train_blocks.items():
                feature_stats_rows.append(block_statistics(block_value, adj, graph_name, block_name))
                sample = block_value.reshape(-1)
                if sample.size > 50000:
                    sample = sample[:: max(1, sample.size // 50000)]
                plot_rows.extend(
                    {"graph": graph_name, "block": block_name, "value": float(value)}
                    for value in sample
                )

            train_feat = combine_feature_blocks(train_blocks, feature_mode, args.max_order)
            val_feat = combine_feature_blocks(val_blocks, feature_mode, args.max_order)
            test_feat = combine_feature_blocks(test_blocks, feature_mode, args.max_order)

            train_flat, train_target = flatten_for_probe(train_feat, train_y)
            val_flat, val_target = flatten_for_probe(val_feat, val_y)
            test_flat, test_target = flatten_for_probe(test_feat, test_y)

            train_std, [val_std, test_std], _, _ = standardize_features(train_flat, [val_flat, test_flat])

            best_alpha = None
            best_val_mae = float("inf")
            best_weight = None
            best_bias = None
            for alpha in args.alphas:
                weight, bias = fit_ridge(train_std, train_target, alpha)
                pred_val = predict_ridge(val_std, weight, bias)
                val_metrics = compute_metrics(pred_val, val_target, null_val)
                if val_metrics["MAE"] < best_val_mae:
                    best_val_mae = val_metrics["MAE"]
                    best_alpha = alpha
                    best_weight = weight
                    best_bias = bias

            pred_test = predict_ridge(test_std, best_weight, best_bias)
            test_metrics = compute_metrics(pred_test, test_target, null_val)
            probe_rows.append(
                {
                    "graph": graph_name,
                    "feature_mode": feature_mode,
                    "max_order": args.max_order,
                    "best_alpha": best_alpha,
                    "val_MAE": best_val_mae,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

    feature_stats_df = pd.DataFrame(feature_stats_rows)
    probe_df = pd.DataFrame(probe_rows)
    plot_df = pd.DataFrame(plot_rows)

    feature_stats_df.to_csv(output_dir / "feature_block_stats.csv", index=False)
    probe_df.to_csv(output_dir / "linear_probe_results.csv", index=False)
    plot_df.to_csv(output_dir / "feature_distribution_sample.csv", index=False)

    if not plot_df.empty:
        feature_distribution_plot(plot_df, output_dir / "feature_distribution_boxplot.png")
    if not feature_stats_df.empty:
        smoothness_plot(feature_stats_df, output_dir / "feature_neighbor_gap.png")

    summary = {
        "dataset": "SD",
        "input_len": input_len,
        "output_len": output_len,
        "max_order": args.max_order,
        "feature_modes": args.feature_modes,
        "graphs": list(graphs.keys()),
        "train_windows": int(len(train_x)),
        "val_windows": int(len(val_x)),
        "test_windows": int(len(test_x)),
        "outputs": {
            "feature_stats_csv": str(output_dir / "feature_block_stats.csv"),
            "linear_probe_csv": str(output_dir / "linear_probe_results.csv"),
            "feature_distribution_csv": str(output_dir / "feature_distribution_sample.csv"),
            "feature_distribution_png": str(output_dir / "feature_distribution_boxplot.png"),
            "neighbor_gap_png": str(output_dir / "feature_neighbor_gap.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

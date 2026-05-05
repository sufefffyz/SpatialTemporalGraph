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
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from basicts.utils.adjacent_matrix_norm import calculate_transition_matrix

DEFAULT_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "eval" / "sd_graph_propagation_analysis"


def log(message: str) -> None:
    print(message, flush=True)


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
    parser.add_argument(
        "--add-self-loop",
        action="store_true",
        help="Add identity matrix before building forward/backward transition matrices.",
    )
    parser.add_argument(
        "--stats-window-samples",
        type=int,
        default=512,
        help="Maximum number of windows used to build node-level feature matrices for expensive diagnostics.",
    )
    parser.add_argument(
        "--mad-pairs",
        type=int,
        default=5000,
        help="Maximum number of edge/non-edge pairs sampled for MADGap diagnostics.",
    )
    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        default=["pca"],
        choices=["pca", "tsne"],
        help="Low-dimensional embedding methods for node signal visualization.",
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


def build_graphs(sd_dir: Path, sd_phys_dir: Path, add_self_loop: bool = False) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    distthre_adj = strip_self_loops(unwrap_adj(load_pkl(sd_dir / "adj_mx.pkl")))
    phys_dir_adj = strip_self_loops(unwrap_adj(load_pkl(sd_phys_dir / "adj_mx.pkl")))

    if add_self_loop:
        distthre_adj = distthre_adj + np.eye(distthre_adj.shape[0], dtype=np.float32)
        phys_dir_adj = phys_dir_adj + np.eye(phys_dir_adj.shape[0], dtype=np.float32)

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

    mae = np.abs(pred[mask] - target[mask])
    rmse = (pred[mask] - target[mask]) ** 2
    mape = np.abs((pred[mape_mask] - target[mape_mask]) / target[mape_mask])
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


def node_feature_matrix(block: np.ndarray, max_windows: int) -> np.ndarray:
    num_windows = min(block.shape[0], max_windows)
    matrix = np.transpose(block[:num_windows], (2, 0, 1)).reshape(block.shape[2], num_windows * block.shape[1])
    return matrix.astype(np.float32)


def variance_across_nodes(block: np.ndarray) -> float:
    return float(np.var(block, axis=2).mean())


def effective_rank(matrix: np.ndarray) -> float:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def directed_dirichlet_energy(block: np.ndarray, adj: np.ndarray) -> float:
    edges = np.argwhere(adj > 0)
    if len(edges) == 0:
        return float("nan")
    energy_terms = []
    for src_idx, dst_idx in edges:
        diff = block[:, :, int(src_idx)] - block[:, :, int(dst_idx)]
        weight = float(adj[int(src_idx), int(dst_idx)])
        energy_terms.append(weight * (diff ** 2))
    return float(np.mean(np.stack(energy_terms, axis=0)))


def cosine_distances_for_pairs(node_matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    x = node_matrix[pairs[:, 0]]
    y = node_matrix[pairs[:, 1]]
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    x_norm[x_norm < 1e-8] = 1.0
    y_norm[y_norm < 1e-8] = 1.0
    sim = np.sum((x / x_norm) * (y / y_norm), axis=1)
    return 1.0 - sim


def mad_gap(node_matrix: np.ndarray, adj: np.ndarray, max_pairs: int, rng: np.random.Generator) -> tuple[float, float, float]:
    num_nodes = adj.shape[0]
    edge_pairs = np.argwhere(adj > 0)
    if len(edge_pairs) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(edge_pairs) > max_pairs:
        edge_pairs = edge_pairs[rng.choice(len(edge_pairs), size=max_pairs, replace=False)]

    non_edge_pairs = []
    target_count = len(edge_pairs)
    attempts = 0
    seen = set()
    while len(non_edge_pairs) < target_count and attempts < target_count * 20:
        i = int(rng.integers(0, num_nodes))
        j = int(rng.integers(0, num_nodes))
        attempts += 1
        if i == j:
            continue
        if adj[i, j] > 0:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        non_edge_pairs.append(key)

    if not non_edge_pairs:
        return float("nan"), float("nan"), float("nan")

    edge_d = cosine_distances_for_pairs(node_matrix, edge_pairs)
    non_edge_d = cosine_distances_for_pairs(node_matrix, np.asarray(non_edge_pairs, dtype=int))
    edge_mean = float(np.mean(edge_d))
    non_edge_mean = float(np.mean(non_edge_d))
    return edge_mean, non_edge_mean, float(non_edge_mean - edge_mean)


def self_retention(node_matrix: np.ndarray, base_matrix: np.ndarray) -> float:
    x = node_matrix.reshape(-1)
    y = base_matrix.reshape(-1)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def block_statistics(
    block: np.ndarray,
    adj: np.ndarray,
    graph_name: str,
    block_name: str,
    base_block: np.ndarray,
    paired_block: np.ndarray | None,
    stats_window_samples: int,
    max_mad_pairs: int,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    values = block.reshape(-1)
    node_matrix = node_feature_matrix(block, stats_window_samples)
    base_matrix = node_feature_matrix(base_block, stats_window_samples)
    var_ratio = variance_across_nodes(block) / max(variance_across_nodes(base_block), 1e-12)
    edge_mad, non_edge_mad, madgap = mad_gap(node_matrix, adj, max_mad_pairs, rng)
    directional_gap = float("nan")
    if paired_block is not None:
        paired_matrix = node_feature_matrix(paired_block, stats_window_samples)
        directional_gap = float(
            np.linalg.norm(node_matrix - paired_matrix) / max(np.linalg.norm(base_matrix), 1e-12)
        )

    return {
        "graph": graph_name,
        "block": block_name,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "mean_neighbor_gap": mean_neighbor_gap(block, adj),
        "var_nodes": variance_across_nodes(block),
        "var_ratio": float(var_ratio),
        "effective_rank": effective_rank(node_matrix),
        "dirichlet_energy": directed_dirichlet_energy(block, adj),
        "mad_edge": edge_mad,
        "mad_non_edge": non_edge_mad,
        "mad_gap": madgap,
        "self_retention": self_retention(node_matrix, base_matrix),
        "directional_gap": directional_gap,
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


def metric_bar_plot(stats_df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stats_df, x="block", y=metric, hue="graph")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def node_embedding_plot(
    embedding_df: pd.DataFrame,
    method: str,
    output_path: Path,
) -> None:
    blocks = list(dict.fromkeys(embedding_df["block"].tolist()))
    fig, axes = plt.subplots(len(blocks), 2, figsize=(12, 4 * len(blocks)))
    if len(blocks) == 1:
        axes = np.asarray([axes])
    for row_idx, block in enumerate(blocks):
        for col_idx, graph_name in enumerate(["distthre", "physical_dir"]):
            ax = axes[row_idx, col_idx]
            sub = embedding_df[(embedding_df["block"] == block) & (embedding_df["graph"] == graph_name)]
            if sub.empty:
                ax.axis("off")
                continue
            sc = ax.scatter(
                sub["dim1"],
                sub["dim2"],
                c=sub["color_value"],
                s=12,
                cmap="viridis",
                alpha=0.8,
            )
            ax.set_title(f"{method.upper()} | {graph_name} | {block}")
            ax.set_xlabel("dim1")
            ax.set_ylabel("dim2")
        fig.colorbar(sc, ax=axes[row_idx, :], fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_embedding(node_matrix: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(node_matrix)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, node_matrix.shape[0] // 20))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(node_matrix)
    raise ValueError(f"Unsupported embedding method: {method}")


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

    graphs, supports = build_graphs(sd_dir, sd_phys_dir, add_self_loop=args.add_self_loop)
    windows = build_windows(flow, input_len=input_len, output_len=output_len, train_val_test_ratio=split_ratio)
    train_x, train_y = subsample_windows(*windows["train"], max_samples=args.max_train_samples)
    val_x, val_y = subsample_windows(*windows["val"], max_samples=args.max_eval_samples)
    test_x, test_y = subsample_windows(*windows["test"], max_samples=args.max_eval_samples)

    log(f"Dataset: SD | input_len={input_len} | output_len={output_len}")
    log(
        "Window counts after subsampling: "
        f"train={len(train_x)}, val={len(val_x)}, test={len(test_x)}"
    )
    log(
        "Running feature modes: "
        + ", ".join(args.feature_modes)
        + f" | max_order={args.max_order} | self_loop={args.add_self_loop}"
    )

    feature_stats_rows = []
    plot_rows = []
    probe_rows = []
    embedding_rows = []
    rng = np.random.default_rng(42)

    # X-only baseline once
    x_only_train_feat = np.expand_dims(train_x, axis=-1)
    x_only_val_feat = np.expand_dims(val_x, axis=-1)
    x_only_test_feat = np.expand_dims(test_x, axis=-1)

    for feature_mode in args.feature_modes:
        log("")
        log("=" * 80)
        log(f"Feature mode: {feature_mode}")
        log("=" * 80)
        if feature_mode == "x_only":
            train_flat, train_target = flatten_for_probe(x_only_train_feat, train_y)
            val_flat, val_target = flatten_for_probe(x_only_val_feat, val_y)
            test_flat, test_target = flatten_for_probe(x_only_test_feat, test_y)

            train_std, [val_std, test_std], _, _ = standardize_features(train_flat, [val_flat, test_flat])

            best_alpha = None
            best_val_mae = float("inf")
            best_weight = None
            best_bias = None
            for alpha in tqdm(args.alphas, desc=f"{feature_mode} alphas", leave=False):
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
            log(
                f"{feature_mode}: best_alpha={best_alpha}, "
                f"val_MAE={best_val_mae:.4f}, "
                f"test_MAE={test_metrics['MAE']:.4f}, "
                f"test_RMSE={test_metrics['RMSE']:.4f}, "
                f"test_MAPE={test_metrics['MAPE']:.4f}"
            )
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
            log(f"Graph: {graph_name}")
            train_blocks = build_feature_blocks(train_x, supports[graph_name], args.max_order)
            val_blocks = build_feature_blocks(val_x, supports[graph_name], args.max_order)
            test_blocks = build_feature_blocks(test_x, supports[graph_name], args.max_order)

            for block_name, block_value in train_blocks.items():
                paired_block = None
                if block_name.startswith("pf_"):
                    paired_block = train_blocks.get(block_name.replace("pf_", "pb_"))
                elif block_name.startswith("pb_"):
                    paired_block = train_blocks.get(block_name.replace("pb_", "pf_"))

                feature_stats_rows.append(
                    block_statistics(
                        block_value,
                        adj,
                        graph_name,
                        block_name,
                        base_block=train_blocks["x"],
                        paired_block=paired_block,
                        stats_window_samples=args.stats_window_samples,
                        max_mad_pairs=args.mad_pairs,
                        rng=rng,
                    )
                )
                sample = block_value.reshape(-1)
                if sample.size > 50000:
                    sample = sample[:: max(1, sample.size // 50000)]
                plot_rows.extend(
                    {"graph": graph_name, "block": block_name, "value": float(value)}
                    for value in sample
                )

                if block_name in {"x", "pf_1", "pb_1", f"pf_{args.max_order}", f"pb_{args.max_order}"}:
                    node_matrix = node_feature_matrix(block_value, args.stats_window_samples)
                    color_values = (adj > 0).sum(axis=1).astype(float)
                    for method in args.embedding_methods:
                        embedding = compute_embedding(node_matrix, method)
                        for node_idx in range(embedding.shape[0]):
                            embedding_rows.append(
                                {
                                    "graph": graph_name,
                                    "block": block_name,
                                    "method": method,
                                    "node_index": node_idx,
                                    "dim1": float(embedding[node_idx, 0]),
                                    "dim2": float(embedding[node_idx, 1]),
                                    "color_value": float(color_values[node_idx]),
                                }
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
            for alpha in tqdm(args.alphas, desc=f"{feature_mode}/{graph_name} alphas", leave=False):
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
            log(
                f"{feature_mode}/{graph_name}: best_alpha={best_alpha}, "
                f"val_MAE={best_val_mae:.4f}, "
                f"test_MAE={test_metrics['MAE']:.4f}, "
                f"test_RMSE={test_metrics['RMSE']:.4f}, "
                f"test_MAPE={test_metrics['MAPE']:.4f}"
            )
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
    embedding_df = pd.DataFrame(embedding_rows)

    feature_stats_df.to_csv(output_dir / "feature_block_stats.csv", index=False)
    probe_df.to_csv(output_dir / "linear_probe_results.csv", index=False)
    plot_df.to_csv(output_dir / "feature_distribution_sample.csv", index=False)
    if not embedding_df.empty:
        embedding_df.to_csv(output_dir / "node_embeddings.csv", index=False)

    if not plot_df.empty:
        feature_distribution_plot(plot_df, output_dir / "feature_distribution_boxplot.png")
    if not feature_stats_df.empty:
        smoothness_plot(feature_stats_df, output_dir / "feature_neighbor_gap.png")
        metric_bar_plot(feature_stats_df, "var_ratio", "Node-Variance Retention after Propagation", output_dir / "feature_var_ratio.png")
        metric_bar_plot(feature_stats_df, "effective_rank", "Effective Rank after Propagation", output_dir / "feature_effective_rank.png")
        metric_bar_plot(feature_stats_df, "dirichlet_energy", "Directed Dirichlet Energy after Propagation", output_dir / "feature_dirichlet_energy.png")
        metric_bar_plot(feature_stats_df, "mad_gap", "MADGap after Propagation", output_dir / "feature_mad_gap.png")
    if not embedding_df.empty:
        for method in embedding_df["method"].unique():
            node_embedding_plot(
                embedding_df[embedding_df["method"] == method].copy(),
                method,
                output_dir / f"node_embedding_{method}.png",
            )

    summary = {
        "dataset": "SD",
        "input_len": input_len,
        "output_len": output_len,
        "max_order": args.max_order,
        "add_self_loop": args.add_self_loop,
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
            "var_ratio_png": str(output_dir / "feature_var_ratio.png"),
            "effective_rank_png": str(output_dir / "feature_effective_rank.png"),
            "dirichlet_energy_png": str(output_dir / "feature_dirichlet_energy.png"),
            "mad_gap_png": str(output_dir / "feature_mad_gap.png"),
            "embedding_csv": str(output_dir / "node_embeddings.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("")
    log(f"Outputs written to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

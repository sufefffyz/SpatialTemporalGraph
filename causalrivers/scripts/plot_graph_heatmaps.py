#!/usr/bin/env python3
"""Plot true-graph and learned-graph heatmaps with aligned node order."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aligned heatmaps for a true graph and a learned graph."
    )
    parser.add_argument(
        "--true-graph-pickle",
        required=True,
        type=Path,
        help="Path to the ground-truth graph pickle, e.g. product/rivers_east_germany.p",
    )
    parser.add_argument(
        "--learned-graph",
        required=True,
        type=Path,
        help="Path to learned_graphs/final.npz or another learned graph snapshot.",
    )
    parser.add_argument(
        "--adj-mx-pkl",
        required=True,
        type=Path,
        help="BasicTS adj_mx.pkl used for node-order alignment.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save the heatmap figures.",
    )
    parser.add_argument(
        "--title-prefix",
        default="rivers",
        help="Prefix used in plot titles and output filenames.",
    )
    parser.add_argument(
        "--show-node-labels",
        action="store_true",
        help="Show node labels on both axes. Off by default for readability.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--pair-cmap",
        default="viridis",
        help="Colormap used for the side-by-side comparison with a shared colorbar.",
    )
    parser.add_argument(
        "--pair-scale-mode",
        choices=["normalize_each", "raw"],
        default="normalize_each",
        help=(
            "How to scale the side-by-side comparison. "
            "'normalize_each' rescales both matrices to [0, 1] before plotting for structural comparison; "
            "'raw' uses the shared raw value range."
        ),
    )
    return parser.parse_args()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _extract_networkx_graph(obj: Any) -> nx.Graph:
    if isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return obj
    if isinstance(obj, dict):
        for key in ("graph", "G", "digraph"):
            if key in obj:
                return _extract_networkx_graph(obj[key])
        for value in obj.values():
            if isinstance(value, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return value
    if isinstance(obj, (list, tuple)):
        for value in obj:
            if isinstance(value, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return value
    raise TypeError(f"Could not extract a networkx graph from {type(obj)!r}")


def _normalize_node_id(node: Any) -> Any:
    if isinstance(node, (int, np.integer)):
        return int(node)
    text = str(node).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _load_node_order_and_adj(adj_mx_pkl: Path) -> tuple[list[Any], np.ndarray]:
    payload = _load_pickle(adj_mx_pkl)
    if isinstance(payload, tuple) and len(payload) == 3:
        node_ids, _node_to_ind, adj = payload
        return [_normalize_node_id(node_id) for node_id in node_ids], np.asarray(adj, dtype=np.float32)
    adj = np.asarray(payload, dtype=np.float32)
    node_ids = list(range(adj.shape[0]))
    return node_ids, adj


def _build_true_adjacency(graph: nx.Graph, node_order: list[Any]) -> np.ndarray:
    adjacency = np.zeros((len(node_order), len(node_order)), dtype=np.float32)
    node_to_idx = {_normalize_node_id(node): idx for idx, node in enumerate(node_order)}
    for source, target in graph.edges():
        source_key = _normalize_node_id(source)
        target_key = _normalize_node_id(target)
        if source_key in node_to_idx and target_key in node_to_idx:
            adjacency[node_to_idx[source_key], node_to_idx[target_key]] = 1.0
    return adjacency


def _load_learned_adjacency(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "adj" not in data:
            raise KeyError(f"{path} does not contain an 'adj' array.")
        return np.asarray(data["adj"], dtype=np.float32)


def _plot_single_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    dpi: int,
    tick_labels: list[str] | None,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    size = max(8, min(18, 0.18 * matrix.shape[0]))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Cause")
    ax.set_ylabel("Effect")
    if tick_labels is not None:
        positions = np.arange(len(tick_labels))
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _minmax_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    min_val = float(np.nanmin(matrix))
    max_val = float(np.nanmax(matrix))
    if np.isclose(min_val, max_val):
        return np.zeros_like(matrix, dtype=np.float32)
    return (matrix - min_val) / (max_val - min_val)


def _plot_pair(
    true_adj: np.ndarray,
    learned_adj: np.ndarray,
    output_path: Path,
    title_prefix: str,
    dpi: int,
    tick_labels: list[str] | None,
    cmap: str,
    scale_mode: str,
) -> None:
    size = max(8, min(18, 0.18 * true_adj.shape[0]))
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, size), constrained_layout=True)

    if scale_mode == "normalize_each":
        true_plot = _minmax_normalize(true_adj)
        learned_plot = _minmax_normalize(learned_adj)
        shared_vmin = 0.0
        shared_vmax = 1.0
        title_suffix = "normalized"
    else:
        true_plot = true_adj
        learned_plot = learned_adj
        shared_vmin = float(min(np.nanmin(true_adj), np.nanmin(learned_adj)))
        shared_vmax = float(max(np.nanmax(true_adj), np.nanmax(learned_adj)))
        title_suffix = "raw"

    left = axes[0].imshow(
        true_plot,
        cmap=cmap,
        aspect="auto",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[0].set_title(f"{title_prefix}: true graph ({title_suffix})")
    axes[0].set_xlabel("Cause")
    axes[0].set_ylabel("Effect")

    right = axes[1].imshow(
        learned_plot,
        cmap=cmap,
        aspect="auto",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[1].set_title(f"{title_prefix}: learned graph ({title_suffix})")
    axes[1].set_xlabel("Cause")
    axes[1].set_ylabel("Effect")

    for ax in axes:
        if tick_labels is not None:
            positions = np.arange(len(tick_labels))
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
            ax.set_yticks(positions)
            ax.set_yticklabels(tick_labels, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(right, ax=axes, fraction=0.03, pad=0.02)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    true_graph_pickle = args.true_graph_pickle.expanduser().resolve()
    learned_graph_path = args.learned_graph.expanduser().resolve()
    adj_mx_pkl = args.adj_mx_pkl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    node_order, _prior_adj = _load_node_order_and_adj(adj_mx_pkl)
    graph_obj = _load_pickle(true_graph_pickle)
    graph = _extract_networkx_graph(graph_obj)
    true_adj = _build_true_adjacency(graph, node_order)
    learned_adj = _load_learned_adjacency(learned_graph_path)

    if learned_adj.shape != true_adj.shape:
        raise ValueError(
            f"Shape mismatch: true adjacency is {true_adj.shape}, learned adjacency is {learned_adj.shape}."
        )

    tick_labels = [str(node) for node in node_order] if args.show_node_labels else None
    prefix = args.title_prefix

    true_out = output_dir / f"{prefix}_true_heatmap.png"
    learned_out = output_dir / f"{prefix}_learned_heatmap.png"
    pair_out = output_dir / f"{prefix}_true_vs_learned_heatmap.png"

    _plot_single_heatmap(
        true_adj,
        true_out,
        title=f"{prefix}: true graph",
        dpi=args.dpi,
        tick_labels=tick_labels,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_single_heatmap(
        learned_adj,
        learned_out,
        title=f"{prefix}: learned graph",
        dpi=args.dpi,
        tick_labels=tick_labels,
        cmap="viridis",
    )
    _plot_pair(
        true_adj,
        learned_adj,
        pair_out,
        title_prefix=prefix,
        dpi=args.dpi,
        tick_labels=tick_labels,
        cmap=args.pair_cmap,
        scale_mode=args.pair_scale_mode,
    )

    print(f"Saved true heatmap   : {true_out}")
    print(f"Saved learned heatmap: {learned_out}")
    print(f"Saved combined figure: {pair_out}")
    print(f"num_nodes={len(node_order)} true_edges={int(np.count_nonzero(true_adj))}")
    print(
        "learned_stats="
        f"min={float(np.nanmin(learned_adj)):.6g} "
        f"max={float(np.nanmax(learned_adj)):.6g} "
        f"mean={float(np.nanmean(learned_adj)):.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

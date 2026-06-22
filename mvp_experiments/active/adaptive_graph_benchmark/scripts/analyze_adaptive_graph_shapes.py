#!/usr/bin/env python3
"""Analyze adaptive graph shapes from trained GWNet/AGCRN/MTGNN checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "mvp_experiments"
    / "active"
    / "adaptive_graph_benchmark"
    / "outputs"
    / "adaptive_graph_shape_analysis"
)


MODEL_SPECS = {
    "GWNet": {
        "variants": {
            "original": "GraphWaveNetAdaptiveImportance",
            "frozen_random_adaptive_from_scratch": "GraphWaveNetFrozenRandomAdaptiveImportance",
            "learned_no_relu": "GraphWaveNetLearnedNoReluAdaptiveImportance",
        },
    },
    "AGCRN": {
        "variants": {
            "original": "AGCRNAdaptiveImportanceOriginal",
            "identity_support": "AGCRNAdaptiveImportanceIdentitySupport",
            "frozen_random_embedding": "AGCRNAdaptiveImportanceFrozenRandomEmbedding",
            "graph_no_relu": "AGCRNAdaptiveImportanceGraphNoRelu",
        },
    },
    "MTGNN": {
        "variants": {
            "original": "MTGNNAdaptiveImportanceOriginal",
            "frozen_random_graph": "MTGNNAdaptiveImportanceFrozenRandomGraph",
            "no_relu_score": "MTGNNAdaptiveImportanceNoReluScore",
        },
    },
}


ACTUAL_USE_RELU = {
    ("GWNet", "original"): True,
    ("GWNet", "frozen_random_adaptive_from_scratch"): True,
    ("GWNet", "learned_no_relu"): False,
    ("AGCRN", "original"): True,
    ("AGCRN", "frozen_random_embedding"): True,
    ("AGCRN", "graph_no_relu"): False,
    ("MTGNN", "original"): True,
    ("MTGNN", "frozen_random_graph"): True,
    ("MTGNN", "no_relu_score"): False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["gwnet", "agcrn-mtgnn", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS04", "PEMS07", "SD"])
    parser.add_argument("--checkpoint-root", type=Path, default=BASICTS_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def selected_models(scope: str) -> list[str]:
    if scope == "gwnet":
        return ["GWNet"]
    if scope == "agcrn-mtgnn":
        return ["AGCRN", "MTGNN"]
    return ["GWNet", "AGCRN", "MTGNN"]


def checkpoint_patterns(checkpoint_root: Path, dataset: str, variant: str, model_name: str) -> list[str]:
    exact = (
        checkpoint_root
        / model_name
        / f"{dataset}_full_{variant}*det0_cudnndet0_100_*"
        / "*"
        / f"{model_name}_best_val_MAE.pt"
    )
    fallback = (
        checkpoint_root
        / model_name
        / f"{dataset}_full_{variant}*det0_cudnndet0*"
        / "*"
        / f"{model_name}_best_val_MAE.pt"
    )
    return [str(exact), str(fallback)]


def find_checkpoint(checkpoint_root: Path, dataset: str, variant: str, model_name: str) -> tuple[Path | None, str]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    patterns = checkpoint_patterns(checkpoint_root, dataset, variant, model_name)
    for pattern in patterns:
        for path in checkpoint_root.parent.glob(Path(pattern).relative_to(checkpoint_root.parent).as_posix()):
            if path not in seen:
                candidates.append(path)
                seen.add(path)
        if candidates:
            break
    if not candidates:
        return None, patterns[0]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1], patterns[0]


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def quantiles(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().flatten().float()
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float32)
    vals = torch.quantile(flat, qs)
    return {
        "min": float(flat.min()),
        "q01": float(vals[0]),
        "q05": float(vals[1]),
        "q50": float(vals[2]),
        "q95": float(vals[3]),
        "q99": float(vals[4]),
        "max": float(flat.max()),
    }


def raw_stats(raw: torch.Tensor) -> dict[str, float]:
    flat = raw.detach().flatten().float()
    stats = quantiles(flat)
    stats.update(
        {
            "raw_mean": float(flat.mean()),
            "raw_std": float(flat.std(unbiased=False)),
            "raw_neg_frac": float((flat < 0).float().mean()),
            "raw_zero_frac": float((flat == 0).float().mean()),
        }
    )
    return stats


def graph_stats(adj: torch.Tensor, eps: float, topks: tuple[int, ...] = (1, 5, 10, 20, 64)) -> dict[str, float]:
    a = adj.detach().float()
    n = a.shape[0]
    abs_a = a.abs()
    nonzero = abs_a > eps
    row_abs_sum = abs_a.sum(dim=1).clamp_min(eps)
    prob_abs = abs_a / row_abs_sum[:, None]
    entropy = -(prob_abs.clamp_min(eps) * prob_abs.clamp_min(eps).log()).sum(dim=1)
    eff_degree = row_abs_sum.pow(2) / abs_a.pow(2).sum(dim=1).clamp_min(eps)
    neg_abs_mass = torch.where(a < -eps, abs_a, torch.zeros_like(abs_a)).sum(dim=1) / row_abs_sum
    diag_mass = abs_a.diag() / row_abs_sum

    stats: dict[str, float] = {
        "num_nodes": float(n),
        "density": float(nonzero.float().mean()),
        "nonzero_per_row_mean": float(nonzero.float().sum(dim=1).mean()),
        "nonzero_per_row_min": float(nonzero.float().sum(dim=1).min()),
        "nonzero_per_row_max": float(nonzero.float().sum(dim=1).max()),
        "entropy_abs_mean": float(entropy.mean()),
        "entropy_abs_norm_mean": float((entropy / math.log(max(n, 2))).mean()),
        "effective_degree_mean": float(eff_degree.mean()),
        "effective_degree_median": float(eff_degree.median()),
        "effective_degree_q10": float(torch.quantile(eff_degree, 0.10)),
        "effective_degree_q90": float(torch.quantile(eff_degree, 0.90)),
        "diag_abs_mass_mean": float(diag_mass.mean()),
        "signed_row_sum_mean": float(a.sum(dim=1).mean()),
        "abs_row_sum_mean": float(row_abs_sum.mean()),
        "negative_edge_frac": float(((a < -eps) & nonzero).float().sum() / nonzero.float().sum().clamp_min(1.0)),
        "negative_abs_mass_frac_mean": float(neg_abs_mass.mean()),
        "asymmetry_abs_ratio": float((a - a.T).abs().sum() / abs_a.sum().clamp_min(eps)),
    }
    sorted_prob, _ = torch.sort(prob_abs, dim=1, descending=True)
    for k in topks:
        kk = min(k, n)
        stats[f"top{kk}_abs_mass_mean"] = float(sorted_prob[:, :kk].sum(dim=1).mean())
    return stats


def softmax_graph(raw: torch.Tensor, use_relu: bool) -> torch.Tensor:
    logits = F.relu(raw) if use_relu else raw
    return torch.softmax(logits, dim=1)


def mtgnn_topk_graph(score: torch.Tensor, use_relu: bool, k: int) -> torch.Tensor:
    adj = F.relu(score) if use_relu else score
    kk = min(k, adj.shape[0])
    mask = torch.zeros_like(adj)
    _, topk_idx = adj.topk(kk, dim=1)
    mask.scatter_(1, topk_idx, 1.0)
    return adj * mask


def mtgnn_score(state: dict[str, torch.Tensor], alpha: float = 3.0) -> torch.Tensor:
    emb1 = state["gc.emb1.weight"]
    emb2 = state["gc.emb2.weight"]
    w1, b1 = state["gc.lin1.weight"], state["gc.lin1.bias"]
    w2, b2 = state["gc.lin2.weight"], state["gc.lin2.bias"]
    nodevec1 = torch.tanh(alpha * F.linear(emb1, w1, b1))
    nodevec2 = torch.tanh(alpha * F.linear(emb2, w2, b2))
    anti = nodevec1 @ nodevec2.T - nodevec2 @ nodevec1.T
    return torch.tanh(alpha * anti)


def gwnet_records(state: dict[str, torch.Tensor], variant: str) -> list[dict[str, Any]]:
    raw = state["nodevec1"] @ state["nodevec2"]
    return graph_records("GWNet", variant, raw, actual_use_relu=ACTUAL_USE_RELU[("GWNet", variant)])


def agcrn_records(state: dict[str, torch.Tensor], variant: str) -> list[dict[str, Any]]:
    emb = state["node_embeddings"]
    raw = emb @ emb.T
    if variant == "identity_support":
        identity = torch.eye(raw.shape[0])
        return [
            {
                "graph_role": "actual",
                "formula": "identity",
                "use_relu": "",
                **raw_stats(raw),
                **graph_stats(identity, eps=1e-12),
            }
        ]
    return graph_records("AGCRN", variant, raw, actual_use_relu=ACTUAL_USE_RELU[("AGCRN", variant)])


def mtgnn_records(state: dict[str, torch.Tensor], variant: str, k: int) -> list[dict[str, Any]]:
    score = mtgnn_score(state)
    actual_use_relu = ACTUAL_USE_RELU[("MTGNN", variant)]
    records = []
    for use_relu, role in [(actual_use_relu, "actual"), (not actual_use_relu, "counterfactual")]:
        adj = mtgnn_topk_graph(score, use_relu=use_relu, k=k)
        records.append(
            {
                "graph_role": role,
                "formula": "topk_relu" if use_relu else "topk_no_relu",
                "use_relu": str(use_relu),
                **raw_stats(score),
                **graph_stats(adj, eps=1e-12),
            }
        )
    return records


def graph_records(model: str, variant: str, raw: torch.Tensor, actual_use_relu: bool) -> list[dict[str, Any]]:
    records = []
    for use_relu, role in [(actual_use_relu, "actual"), (not actual_use_relu, "counterfactual")]:
        adj = softmax_graph(raw, use_relu=use_relu)
        records.append(
            {
                "graph_role": role,
                "formula": "softmax_relu" if use_relu else "softmax_no_relu",
                "use_relu": str(use_relu),
                **raw_stats(raw),
                **graph_stats(adj, eps=1e-12),
            }
        )
    return records


def topk_indices(adj: torch.Tensor, k: int) -> torch.Tensor:
    kk = min(k, adj.shape[1])
    return adj.abs().topk(kk, dim=1).indices


def row_topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    ia = topk_indices(a, k)
    ib = topk_indices(b, k)
    overlaps = []
    for row in range(ia.shape[0]):
        overlaps.append(len(set(ia[row].tolist()) & set(ib[row].tolist())) / ia.shape[1])
    return float(sum(overlaps) / len(overlaps))


def actual_adj_for_record(model: str, variant: str, state: dict[str, torch.Tensor], k: int) -> torch.Tensor | None:
    if model == "GWNet":
        raw = state["nodevec1"] @ state["nodevec2"]
        return softmax_graph(raw, ACTUAL_USE_RELU[("GWNet", variant)])
    if model == "AGCRN":
        raw = state["node_embeddings"] @ state["node_embeddings"].T
        if variant == "identity_support":
            return torch.eye(raw.shape[0])
        return softmax_graph(raw, ACTUAL_USE_RELU[("AGCRN", variant)])
    if model == "MTGNN":
        score = mtgnn_score(state)
        return mtgnn_topk_graph(score, ACTUAL_USE_RELU[("MTGNN", variant)], k)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> int:
    run_name = args.run_name or "shape_analysis"
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    shape_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    actual_adjs: dict[tuple[str, str, str], torch.Tensor] = {}

    for model in selected_models(args.scope):
        for dataset in args.datasets:
            for variant, model_name in MODEL_SPECS[model]["variants"].items():
                checkpoint, pattern = find_checkpoint(args.checkpoint_root.resolve(), dataset, variant, model_name)
                if checkpoint is None:
                    missing_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "variant": variant,
                            "model_name": model_name,
                            "pattern": pattern,
                            "reason": "checkpoint_not_found",
                        }
                    )
                    continue
                state = load_state_dict(checkpoint)
                if model == "GWNet":
                    records = gwnet_records(state, variant)
                elif model == "AGCRN":
                    records = agcrn_records(state, variant)
                elif model == "MTGNN":
                    records = mtgnn_records(state, variant, args.topk)
                else:
                    records = []
                actual_adj = actual_adj_for_record(model, variant, state, args.topk)
                if actual_adj is not None:
                    actual_adjs[(dataset, model, variant)] = actual_adj
                for record in records:
                    shape_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "variant": variant,
                            "model_name": model_name,
                            "checkpoint": str(checkpoint.resolve()),
                            **record,
                        }
                    )

    overlap_rows: list[dict[str, Any]] = []
    for (dataset, model, variant), adj in actual_adjs.items():
        if variant == "original":
            continue
        base = actual_adjs.get((dataset, model, "original"))
        if base is None or base.shape != adj.shape:
            continue
        overlap_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "reference": "original",
                f"top{args.topk}_row_overlap": row_topk_overlap(base, adj, args.topk),
            }
        )

    fieldnames = sorted({key for row in shape_rows for key in row})
    write_csv(run_dir / "graph_shape_stats.csv", shape_rows, fieldnames)
    write_csv(
        run_dir / "missing.csv",
        missing_rows,
        ["dataset", "model", "variant", "model_name", "pattern", "reason"],
    )
    write_csv(
        run_dir / "edge_overlap_vs_original.csv",
        overlap_rows,
        ["dataset", "model", "variant", "reference", f"top{args.topk}_row_overlap"],
    )
    with (run_dir / "graph_shape_stats.jsonl").open("w") as f:
        for row in shape_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(run_dir)
    print(f"shape_rows={len(shape_rows)} missing={len(missing_rows)} overlaps={len(overlap_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(analyze(parse_args()))

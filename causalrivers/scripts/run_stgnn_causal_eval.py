#!/usr/bin/env python3
"""Convenience wrapper for evaluating BasicTS learned graphs with causalrivers.

Examples
--------
Traffic 15min, AGCRN, 2-hop labels:

    python scripts/run_stgnn_causal_eval.py \
        --dataset TRAFFIC_VOLUME_15MIN \
        --model AGCRN \
        --strategy 2_hop \
        --n-vars 3

Rivers 6h, GWNET, random labels:

    python scripts/run_stgnn_causal_eval.py \
        --dataset RIVERS_EAST_GERMANY_6H \
        --model GWNET \
        --strategy random \
        --n-vars 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve dataset/model paths and run learned-graph causal evaluation."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="BasicTS dataset name, for example TRAFFIC_VOLUME_15MIN or RIVERS_EAST_GERMANY_6H.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["AGCRN", "D2STGNN", "GTS", "MTGNN", "GWNET"],
        help="STGNN model name.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Label strategy, for example 2_hop/random/confounder/close/root_cause.",
    )
    parser.add_argument("--n-vars", type=int, default=3, help="Label subgraph size.")
    parser.add_argument(
        "--label-tag",
        default="paperlike",
        help="Traffic label suffix. Ignored for rivers labels unless --label-path is overridden.",
    )
    parser.add_argument(
        "--region",
        default="east",
        help="Rivers region label stem. Default: east.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Checkpoint epoch count encoded in CKPT_SAVE_DIR.")
    parser.add_argument("--input-len", type=int, default=12, help="Checkpoint input length.")
    parser.add_argument("--output-len", type=int, default=12, help="Checkpoint output length.")
    parser.add_argument(
        "--basicts-root",
        type=Path,
        default=PROJECT_ROOT.parent / "BasicTS",
        help="Path to the BasicTS repository root.",
    )
    parser.add_argument(
        "--label-path",
        action="append",
        default=[],
        help="Optional explicit label path(s). When provided, automatic dataset-based resolution is skipped.",
    )
    parser.add_argument(
        "--learned-graph",
        type=Path,
        default=None,
        help="Optional explicit learned graph file or learned_graphs directory. Defaults to the checkpoint final snapshot.",
    )
    parser.add_argument(
        "--adj-mx-pkl",
        type=Path,
        default=None,
        help="Optional explicit adj_mx.pkl path. Defaults to BasicTS/datasets/<dataset>/adj_mx.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/learned_graph_eval/<dataset>_<model>_<strategy>_<nvars>.",
    )
    parser.add_argument("--all-snapshots", action="store_true", help="Evaluate the full learned_graphs directory instead of final.npz.")
    parser.add_argument("--restrict-to", type=int, default=-1, help="Optional label-graph index to evaluate.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel scorer jobs.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for scorer parallelism.")
    parser.add_argument("--keep-autoregressive", action="store_true", help="Keep self-loops during scoring.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    return parser.parse_args()


def model_dir_for(model: str) -> str:
    if model == "GWNET":
        return "GraphWaveNet"
    return model


def traffic_label_dataset_name(dataset: str) -> str:
    prefix = "TRAFFIC_VOLUME_"
    if not dataset.startswith(prefix):
        raise ValueError(f"Unsupported traffic dataset name: {dataset}")
    suffix = dataset[len(prefix) :]
    if suffix == "5MIN":
        return "city_traffic_m_volume__category__1_0"
    return f"city_traffic_m_volume__category__1_0_{suffix.lower()}"


def resolve_label_paths(args: argparse.Namespace) -> list[Path]:
    if args.label_path:
        return [Path(path).expanduser().resolve() for path in args.label_path]

    if args.dataset.startswith("TRAFFIC_VOLUME_"):
        dataset_name = traffic_label_dataset_name(args.dataset)
        label_group = f"{args.strategy}_{args.n_vars}"
        if args.label_tag:
            label_group = f"{label_group}_{args.label_tag}"
        label_path = (
            PROJECT_ROOT
            / "datasets"
            / f"traffic_{dataset_name}"
            / label_group
            / f"{dataset_name}.p"
        )
        return [label_path.resolve()]

    if args.dataset.startswith("RIVERS_EAST_GERMANY_"):
        label_path = PROJECT_ROOT / "datasets" / f"{args.strategy}_{args.n_vars}" / f"{args.region}.p"
        return [label_path.resolve()]

    raise ValueError(
        f"Unsupported dataset family for automatic label resolution: {args.dataset}. "
        "Pass --label-path explicitly if needed."
    )


def resolve_checkpoint_graph(args: argparse.Namespace) -> Path:
    if args.learned_graph is not None:
        return args.learned_graph.expanduser().resolve()

    ckpt_dir = (
        args.basicts_root.expanduser().resolve()
        / "checkpoints"
        / model_dir_for(args.model)
        / f"{args.dataset}_{args.epochs}_{args.input_len}_{args.output_len}"
    )
    if args.all_snapshots:
        return (ckpt_dir / "learned_graphs").resolve()
    return (ckpt_dir / "learned_graphs" / "final.npz").resolve()


def resolve_adj_mx_pkl(args: argparse.Namespace) -> Path:
    if args.adj_mx_pkl is not None:
        return args.adj_mx_pkl.expanduser().resolve()
    return (args.basicts_root.expanduser().resolve() / "datasets" / args.dataset / "adj_mx.pkl").resolve()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    run_name = f"{args.dataset.lower()}_{args.model.lower()}_{args.strategy}_{args.n_vars}"
    return (PROJECT_ROOT / "results" / "learned_graph_eval" / run_name).resolve()


def main() -> int:
    args = parse_args()

    eval_script = (PROJECT_ROOT / "scripts" / "evaluate_learned_graphs_against_labels.py").resolve()
    label_paths = resolve_label_paths(args)
    learned_graph = resolve_checkpoint_graph(args)
    adj_mx_pkl = resolve_adj_mx_pkl(args)
    output_dir = resolve_output_dir(args)

    command = [sys.executable, str(eval_script)]
    for label_path in label_paths:
        command.extend(["--label-path", str(label_path)])
    command.extend(
        [
            "--learned-graph",
            str(learned_graph),
            "--adj-mx-pkl",
            str(adj_mx_pkl),
            "--output-dir",
            str(output_dir),
            "--restrict-to",
            str(args.restrict_to),
            "--n-jobs",
            str(args.n_jobs),
            "--chunk-size",
            str(args.chunk_size),
        ]
    )
    if args.keep_autoregressive:
        command.append("--keep-autoregressive")

    print("Resolved evaluation command:")
    print(" ".join(command))

    missing_paths = [path for path in label_paths + [learned_graph, adj_mx_pkl] if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"The following required paths do not exist:\n{missing_text}")

    if args.dry_run:
        return 0

    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

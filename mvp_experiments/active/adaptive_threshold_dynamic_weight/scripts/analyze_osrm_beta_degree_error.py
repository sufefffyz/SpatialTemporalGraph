#!/usr/bin/env python3
"""Analyze OSRM beta graph degree distributions and node-level errors.

This script is designed for the BasicTS SD_OSRMGG_Bxxx sweep. It has two modes:

1. Build graph degree summaries and distribution plots from staged adj_mx.pkl files.
2. Stream test predictions from existing best-val checkpoints and accumulate
   per-node MAE/RMSE/MAPE without saving full prediction tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BETA_SPECS = [
    ("0.25", "beta0p25", "SD_OSRMGG_B025"),
    ("0.50", "beta0p50", "SD_OSRMGG_B050"),
    ("1.00", "beta1p00", "SD_OSRMGG_B100"),
    ("1.50", "beta1p50", "SD_OSRMGG_B150"),
    ("2.00", "beta2p00", "SD_OSRMGG_B200"),
]

MODEL_SPECS = {
    "gwnet_fixed": {
        "model_dir": "GraphWaveNet",
        "model_name": "GraphWaveNet",
        "config": "baselines/GWNet/SD_osrm_gaussian_global.py",
        "ckpt_suffix": "osrm_gaussian_global_fixed",
        "run_tag": "osrm_gaussian_global_20260513",
    },
    "gwnet_adaptive": {
        "model_dir": "GraphWaveNet",
        "model_name": "GraphWaveNet",
        "config": "baselines/GWNet/SD_osrm_gaussian_global_adaptive.py",
        "ckpt_suffix": "osrm_gaussian_global_adaptive",
        "run_tag": "osrm_gaussian_global_gwnet_adaptive_20260513",
    },
    "dcrnn_fixed": {
        "model_dir": "DCRNN",
        "model_name": "DCRNN",
        "config": "baselines/DCRNN/SD_osrm_gaussian_global.py",
        "ckpt_suffix": "osrm_gaussian_global_fixed",
        "run_tag": "osrm_gaussian_global_20260513",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch", "single"), default="batch")
    parser.add_argument("--basicts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--device-type", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--models", default="gwnet_fixed,gwnet_adaptive,dcrnn_fixed")
    parser.add_argument("--betas", default="0.25,0.50,1.00,1.50,2.00")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--degree-only", action="store_true")
    parser.add_argument("--reference-beta", default="1.00")
    parser.add_argument("--num-strata", type=int, default=10)
    parser.add_argument(
        "--reference-degree-col",
        choices=("out_degree", "in_degree", "total_degree"),
        default="out_degree",
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), default=None)
    parser.add_argument("--beta", choices=[item[0] for item in BETA_SPECS], default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def load_pkl(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def load_adj_payload(path: Path) -> tuple[list[str] | None, np.ndarray]:
    obj = load_pkl(path)
    node_ids = None
    if isinstance(obj, (tuple, list)):
        if len(obj) >= 3:
            node_ids = [str(item) for item in obj[0]]
            obj = obj[2]
        elif len(obj) == 1:
            obj = obj[0]
    return node_ids, np.asarray(obj)


def offdiag_support(adj: np.ndarray) -> np.ndarray:
    support = adj != 0
    np.fill_diagonal(support, False)
    return support


def quantile_dict(values: np.ndarray, prefix: str) -> dict[str, float]:
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    vals = np.percentile(values, qs)
    return {f"{prefix}_p{q:02d}": float(v) for q, v in zip(qs, vals)}


def build_degree_tables(basicts_dir: Path, output_dir: Path, betas: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    node_rows = []
    for beta, graph_tag, dataset_name in BETA_SPECS:
        if beta not in betas:
            continue
        adj_path = basicts_dir / "datasets" / dataset_name / "adj_mx.pkl"
        node_ids, adj = load_adj_payload(adj_path)
        support = offdiag_support(adj.copy())
        out_degree = support.sum(axis=1).astype(int)
        in_degree = support.sum(axis=0).astype(int)
        total_degree = out_degree + in_degree
        out_weight = np.where(support, adj, 0).sum(axis=1)
        in_weight = np.where(support, adj, 0).sum(axis=0)
        n = int(adj.shape[0])
        edge_count = int(support.sum())
        row = {
            "beta": beta,
            "graph_tag": graph_tag,
            "dataset": dataset_name,
            "num_nodes": n,
            "edge_count": edge_count,
            "avg_out_degree": edge_count / n,
            "density_no_diag": edge_count / (n * (n - 1)),
            "isolated_out_nodes": int((out_degree == 0).sum()),
            "isolated_in_nodes": int((in_degree == 0).sum()),
            "out_degree_std": float(out_degree.std()),
            "out_degree_cv": float(out_degree.std() / out_degree.mean()) if out_degree.mean() else math.nan,
            "in_degree_std": float(in_degree.std()),
            "in_degree_cv": float(in_degree.std() / in_degree.mean()) if in_degree.mean() else math.nan,
        }
        row.update(quantile_dict(out_degree, "out_degree"))
        row.update(quantile_dict(in_degree, "in_degree"))
        rows.append(row)
        for idx in range(n):
            node_rows.append(
                {
                    "beta": beta,
                    "graph_tag": graph_tag,
                    "dataset": dataset_name,
                    "node_index": idx,
                    "node_id": node_ids[idx] if node_ids is not None else str(idx),
                    "out_degree": int(out_degree[idx]),
                    "in_degree": int(in_degree[idx]),
                    "total_degree": int(total_degree[idx]),
                    "out_weight_sum": float(out_weight[idx]),
                    "in_weight_sum": float(in_weight[idx]),
                }
            )
    summary_df = pd.DataFrame(rows)
    node_df = pd.DataFrame(node_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "graph_degree_summary.csv", index=False)
    node_df.to_csv(output_dir / "degree_by_node.csv", index=False)
    return summary_df, node_df


def find_checkpoint(basicts_dir: Path, model_key: str, dataset_name: str) -> Path | None:
    spec = MODEL_SPECS[model_key]
    pattern = (
        basicts_dir
        / "checkpoints"
        / spec["model_dir"]
        / f"{dataset_name}_{spec['ckpt_suffix']}_100_12_12_{spec['run_tag']}"
        / "*"
        / f"{spec['model_name']}_best_val_MAE.pt"
    )
    matches = sorted(pattern.parent.parent.glob(f"*/{spec['model_name']}_best_val_MAE.pt"))
    return matches[-1] if matches else None


def run_single_metric_export(args: argparse.Namespace) -> None:
    if args.model_key is None or args.beta is None or args.checkpoint is None:
        raise ValueError("--mode single requires --model-key, --beta, and --checkpoint.")
    beta, graph_tag, dataset_name = next(item for item in BETA_SPECS if item[0] == args.beta)
    spec = MODEL_SPECS[args.model_key]

    os.environ["BASICTS_DATA_NAME"] = dataset_name
    os.environ["BASICTS_GRAPH_TAG"] = graph_tag
    os.environ["BASICTS_RUN_TAG"] = spec["run_tag"]
    os.environ.setdefault("WANDB_MODE", "disabled")

    sys.path.insert(0, str(args.basicts_dir))
    os.chdir(args.basicts_dir)

    import torch  # pylint: disable=import-outside-toplevel
    from easytorch.config import init_cfg  # pylint: disable=import-outside-toplevel
    from easytorch.device import set_device_type  # pylint: disable=import-outside-toplevel
    from easytorch.utils import set_visible_devices  # pylint: disable=import-outside-toplevel

    set_device_type(args.device_type)
    if args.device_type != "cpu":
        set_visible_devices(args.gpu)

    cfg = init_cfg(spec["config"], save=False)
    runner = cfg["RUNNER"](cfg)
    runner.init_logger(logger_name=f"node-metrics-{args.model_key}-{graph_tag}", log_file_name=None)
    runner.init_test(cfg)
    runner.load_model(str(args.checkpoint), strict=True)
    runner.model.eval()

    num_nodes = int(cfg["MODEL"]["PARAM"]["num_nodes"])
    null_val = float(cfg["METRICS"]["NULL_VAL"])
    sum_abs = np.zeros(num_nodes, dtype=np.float64)
    sum_sq = np.zeros(num_nodes, dtype=np.float64)
    sum_ape = np.zeros(num_nodes, dtype=np.float64)
    sum_target_abs = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)

    with torch.no_grad():
        for data in runner.test_data_loader:
            forward_return = runner.forward(data, epoch=None, iter_num=None, train=False)
            pred = forward_return["prediction"].detach().cpu().numpy()
            target = forward_return["target"].detach().cpu().numpy()
            if np.isnan(null_val):
                mask = ~np.isnan(target)
            else:
                mask = ~np.isclose(target, null_val, atol=5e-5, rtol=0.0)
            err = pred - target
            abs_err = np.abs(err)
            sq_err = err * err
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs(err / target)
            axes = (0, 1, 3)
            sum_abs += np.where(mask, abs_err, 0.0).sum(axis=axes)
            sum_sq += np.where(mask, sq_err, 0.0).sum(axis=axes)
            sum_ape += np.where(mask, ape, 0.0).sum(axis=axes)
            sum_target_abs += np.where(mask, np.abs(target), 0.0).sum(axis=axes)
            count += mask.sum(axis=axes)

    safe_count = np.maximum(count, 1.0)
    out = pd.DataFrame(
        {
            "model": args.model_key,
            "beta": beta,
            "graph_tag": graph_tag,
            "dataset": dataset_name,
            "node_index": np.arange(num_nodes),
            "valid_count": count,
            "mae": sum_abs / safe_count,
            "rmse": np.sqrt(sum_sq / safe_count),
            "mape": sum_ape / safe_count,
            "wape": sum_abs / (sum_target_abs + 5e-5),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"node_metrics_{args.model_key}_{graph_tag}.csv"
    out.to_csv(out_path, index=False)
    print(json.dumps({"path": str(out_path), "rows": int(len(out))}, indent=2))


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return math.nan
    if method == "pearson":
        from scipy.stats import pearsonr  # pylint: disable=import-outside-toplevel

        return float(pearsonr(x[mask], y[mask]).statistic)
    from scipy.stats import spearmanr  # pylint: disable=import-outside-toplevel

    return float(spearmanr(x[mask], y[mask]).statistic)


def make_degree_plots(output_dir: Path, degree_df: pd.DataFrame) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    plt.figure(figsize=(8, 5))
    for beta, group in degree_df.groupby("beta", sort=False):
        plt.hist(group["out_degree"], bins=30, alpha=0.35, density=True, label=f"beta {beta}")
    plt.xlabel("Out-degree")
    plt.ylabel("Node density")
    plt.title("OSRM beta graph out-degree distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "degree_distribution_out_degree.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    degree_df.boxplot(column="out_degree", by="beta", grid=False)
    plt.suptitle("")
    plt.title("Out-degree by beta")
    plt.xlabel("beta")
    plt.ylabel("Out-degree")
    plt.tight_layout()
    plt.savefig(output_dir / "degree_boxplot_out_degree.png", dpi=220)
    plt.close()


def make_error_plots(output_dir: Path, merged_df: pd.DataFrame) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    for model, model_df in merged_df.groupby("model", sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        for metric, ax in zip(["mae", "rmse", "mape"], axes):
            for beta, beta_df in model_df.groupby("beta", sort=False):
                beta_df = beta_df.copy()
                bins = min(5, beta_df["out_degree"].nunique())
                if bins < 2:
                    continue
                beta_df["degree_bin"] = pd.qcut(
                    beta_df["out_degree"].rank(method="first"),
                    q=bins,
                    labels=[f"Q{i + 1}" for i in range(bins)],
                )
                agg = beta_df.groupby("degree_bin", observed=True)[metric].mean()
                ax.plot(range(len(agg)), agg.values, marker="o", label=f"beta {beta}")
            ax.set_title(metric.upper())
            ax.set_xlabel("Within-beta out-degree quantile")
            ax.set_ylabel(f"Mean node {metric.upper()}")
            ax.set_xticks(range(5))
            ax.set_xticklabels([f"Q{i + 1}" for i in range(5)])
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Node error by out-degree quantile: {model}")
        fig.tight_layout()
        fig.savefig(output_dir / f"degree_error_binned_{model}.png", dpi=220)
        plt.close(fig)


def build_reference_degree_strata(
    output_dir: Path,
    degree_df: pd.DataFrame,
    model_keys: list[str],
    reference_beta: str,
    num_strata: int,
    reference_degree_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group nodes by beta=reference_beta degree deciles and track beta-wise errors.

    The important detail is that the node groups are fixed by the reference
    graph. Later beta values are evaluated on those same node sets, so changes
    reflect model/graph behavior rather than changing bin membership.
    """
    ref = degree_df[degree_df["beta"].astype(str) == str(reference_beta)].copy()
    if ref.empty:
        return pd.DataFrame(), pd.DataFrame()
    if num_strata < 2:
        raise ValueError("--num-strata must be at least 2.")

    ref = ref.sort_values([reference_degree_col, "node_index"], kind="mergesort").reset_index(drop=True)
    labels = [f"D{i + 1:02d}" for i in range(num_strata)]
    ref["ref_degree_stratum"] = pd.qcut(
        ref[reference_degree_col].rank(method="first"),
        q=num_strata,
        labels=labels,
    )
    ref_cols = [
        "node_index",
        "node_id",
        reference_degree_col,
        "ref_degree_stratum",
    ]
    ref_nodes = ref[ref_cols].rename(columns={reference_degree_col: "ref_degree"})
    ref_nodes.to_csv(
        output_dir / f"beta{reference_beta.replace('.', 'p')}_{reference_degree_col}_decile_nodes.csv",
        index=False,
    )

    metric_frames = []
    for model_key in model_keys:
        for path in sorted(output_dir.glob(f"node_metrics_{model_key}_beta*.csv")):
            metric_frames.append(pd.read_csv(path, dtype={"beta": str, "graph_tag": str, "dataset": str}))
    if not metric_frames:
        return pd.DataFrame(), pd.DataFrame()

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    merged = metrics_df.merge(ref_nodes, on="node_index", how="inner")
    rows = []
    for (model, beta, stratum), group in merged.groupby(
        ["model", "beta", "ref_degree_stratum"],
        sort=False,
        observed=True,
    ):
        rows.append(
            {
                "model": model,
                "beta": beta,
                "ref_beta": reference_beta,
                "ref_degree_signal": reference_degree_col,
                "ref_degree_stratum": str(stratum),
                "node_count": int(group["node_index"].nunique()),
                "ref_degree_min": float(group["ref_degree"].min()),
                "ref_degree_mean": float(group["ref_degree"].mean()),
                "ref_degree_max": float(group["ref_degree"].max()),
                "mae": float(group["mae"].mean()),
                "rmse": float(group["rmse"].mean()),
                "mape": float(group["mape"].mean()),
                "wape": float(group["wape"].mean()) if "wape" in group else math.nan,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, pd.DataFrame()

    beta_order = [item[0] for item in BETA_SPECS]
    summary["beta"] = pd.Categorical(summary["beta"], categories=beta_order, ordered=True)
    summary["ref_degree_stratum"] = pd.Categorical(summary["ref_degree_stratum"], categories=labels, ordered=True)
    summary = summary.sort_values(["model", "ref_degree_stratum", "beta"]).reset_index(drop=True)
    summary["beta"] = summary["beta"].astype(str)
    summary["ref_degree_stratum"] = summary["ref_degree_stratum"].astype(str)

    baseline = summary[summary["beta"] == str(reference_beta)][
        ["model", "ref_degree_stratum", "mae", "rmse", "mape", "wape"]
    ].rename(
        columns={
            "mae": "mae_ref_beta",
            "rmse": "rmse_ref_beta",
            "mape": "mape_ref_beta",
            "wape": "wape_ref_beta",
        }
    )
    delta = summary.merge(baseline, on=["model", "ref_degree_stratum"], how="left")
    for metric in ["mae", "rmse", "mape", "wape"]:
        delta[f"delta_{metric}_vs_ref_beta"] = delta[metric] - delta[f"{metric}_ref_beta"]
        denom = delta[f"{metric}_ref_beta"].replace(0, np.nan)
        delta[f"rel_delta_{metric}_vs_ref_beta"] = delta[f"delta_{metric}_vs_ref_beta"] / denom

    summary.to_csv(output_dir / "beta1_degree_decile_metric_summary.csv", index=False)
    delta.to_csv(output_dir / "beta1_degree_decile_metric_delta_vs_beta1.csv", index=False)
    make_reference_strata_plots(output_dir, summary, delta, labels)
    return summary, delta


def make_reference_strata_plots(
    output_dir: Path,
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    labels: list[str],
) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    beta_order = [item[0] for item in BETA_SPECS]
    for model, model_df in summary.groupby("model", sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        for metric, ax in zip(["mae", "rmse", "mape"], axes):
            for stratum, stratum_df in model_df.groupby("ref_degree_stratum", sort=False):
                stratum_df = stratum_df.copy()
                stratum_df["beta"] = pd.Categorical(stratum_df["beta"], categories=beta_order, ordered=True)
                stratum_df = stratum_df.sort_values("beta")
                ax.plot(stratum_df["beta"].astype(str), stratum_df[metric], marker="o", linewidth=1.2, label=str(stratum))
            ax.set_title(metric.upper())
            ax.set_xlabel("beta")
            ax.set_ylabel(f"Mean node {metric.upper()}")
        axes[0].legend(title="beta=1 degree decile", fontsize=7, ncol=2)
        fig.suptitle(f"Fixed beta=1.0 degree deciles across graph betas: {model}")
        fig.tight_layout()
        fig.savefig(output_dir / f"beta1_degree_decile_metric_curves_{model}.png", dpi=220)
        plt.close(fig)

        model_delta = delta[delta["model"] == model].copy()
        for metric in ["mae", "rmse", "mape"]:
            pivot = model_delta.pivot(
                index="ref_degree_stratum",
                columns="beta",
                values=f"delta_{metric}_vs_ref_beta",
            )
            pivot = pivot.reindex(index=labels, columns=beta_order)
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
            ax.set_xticks(range(len(beta_order)))
            ax.set_xticklabels(beta_order)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("beta")
            ax.set_ylabel("beta=1.0 degree decile")
            ax.set_title(f"Delta {metric.upper()} vs beta=1.0: {model}")
            fig.colorbar(im, ax=ax, label=f"Delta {metric.upper()}")
            fig.tight_layout()
            fig.savefig(output_dir / f"beta1_degree_decile_delta_heatmap_{model}_{metric}.png", dpi=220)
            plt.close(fig)


def build_correlations(
    output_dir: Path,
    degree_df: pd.DataFrame,
    model_keys: list[str],
    reference_beta: str,
    num_strata: int,
    reference_degree_col: str,
) -> pd.DataFrame:
    metric_frames = []
    for model_key in model_keys:
        for path in sorted(output_dir.glob(f"node_metrics_{model_key}_beta*.csv")):
            metric_frames.append(pd.read_csv(path, dtype={"beta": str, "graph_tag": str, "dataset": str}))
    if not metric_frames:
        return pd.DataFrame()
    degree_df = degree_df.copy()
    degree_df["beta"] = degree_df["beta"].astype(str)
    degree_df["graph_tag"] = degree_df["graph_tag"].astype(str)
    degree_df["dataset"] = degree_df["dataset"].astype(str)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    merged = metrics_df.merge(degree_df, on=["beta", "graph_tag", "dataset", "node_index"], how="left")
    merged.to_csv(output_dir / "degree_error_by_node.csv", index=False)

    rows = []
    for (model, beta), group in merged.groupby(["model", "beta"], sort=False):
        for degree_col in ["out_degree", "in_degree", "total_degree", "out_weight_sum", "in_weight_sum"]:
            for metric in ["mae", "rmse", "mape", "wape"]:
                x = group[degree_col].to_numpy(dtype=float)
                y = group[metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "model": model,
                        "beta": beta,
                        "degree_signal": degree_col,
                        "metric": metric,
                        "pearson": safe_corr(x, y, "pearson"),
                        "spearman": safe_corr(x, y, "spearman"),
                        "num_nodes": int(len(group)),
                    }
                )
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "degree_error_correlations.csv", index=False)
    make_error_plots(output_dir, merged)
    build_reference_degree_strata(
        output_dir=output_dir,
        degree_df=degree_df,
        model_keys=model_keys,
        reference_beta=reference_beta,
        num_strata=num_strata,
        reference_degree_col=reference_degree_col,
    )
    return corr_df


def batch(args: argparse.Namespace) -> None:
    betas = {item.strip() for item in args.betas.split(",") if item.strip()}
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df, degree_df = build_degree_tables(args.basicts_dir, args.output_dir, betas)
    make_degree_plots(args.output_dir, degree_df)

    if not args.degree_only:
        for model_key in model_keys:
            for beta, graph_tag, dataset_name in BETA_SPECS:
                if beta not in betas:
                    continue
                out_path = args.output_dir / f"node_metrics_{model_key}_{graph_tag}.csv"
                if args.skip_existing and out_path.exists():
                    print(f"skip existing {out_path}")
                    continue
                ckpt = find_checkpoint(args.basicts_dir, model_key, dataset_name)
                if ckpt is None:
                    print(f"missing checkpoint: {model_key} {graph_tag}", file=sys.stderr)
                    continue
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "single",
                    "--basicts-dir",
                    str(args.basicts_dir),
                    "--output-dir",
                    str(args.output_dir),
                    "--gpu",
                    args.gpu,
                    "--device-type",
                    args.device_type,
                    "--model-key",
                    model_key,
                    "--beta",
                    beta,
                    "--checkpoint",
                    str(ckpt),
                ]
                print("RUN", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)

    corr_df = build_correlations(
        args.output_dir,
        degree_df,
        model_keys,
        args.reference_beta,
        args.num_strata,
        args.reference_degree_col,
    )
    print("Wrote:")
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(path)
    print("Degree summary:")
    print(summary_df.to_string(index=False))
    if not corr_df.empty:
        print("Top out_degree correlations:")
        filt = corr_df[corr_df["degree_signal"] == "out_degree"].copy()
        filt["abs_spearman"] = filt["spearman"].abs()
        print(filt.sort_values("abs_spearman", ascending=False).head(20).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_single_metric_export(args)
    else:
        batch(args)


if __name__ == "__main__":
    main()

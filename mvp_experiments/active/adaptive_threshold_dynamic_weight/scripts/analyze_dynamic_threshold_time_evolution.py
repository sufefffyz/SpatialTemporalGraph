#!/usr/bin/env python3
"""Analyze time evolution of a trained DynamicThreshold graph.

This is a post-hoc analysis script. It does not train a model or change any
backbone setting; it only loads a best-val checkpoint and streams a BasicTS split
to measure node-wise radius thresholds and active degrees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _basic_ts_dir(repo_root: Path) -> Path:
    return repo_root / "BasicTS"


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _distance_scale_m(raw_distance_path: str | None, dist_norm: str) -> float:
    if not raw_distance_path:
        return 1.0
    raw_path = Path(raw_distance_path).expanduser()
    if not raw_path.exists():
        return 1.0
    dist = np.load(raw_path).astype("float64")
    positive = dist[np.isfinite(dist) & (dist > 0)]
    if positive.size == 0:
        return 1.0
    norm = str(dist_norm or "none").lower()
    if norm == "p95":
        return float(np.quantile(positive, 0.95))
    if norm == "median":
        return float(np.median(positive))
    if norm in {"none", "identity", "raw"}:
        return 1.0
    return float(np.max(positive))


def _clone_batch(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if hasattr(value, "clone") else value for key, value in data.items()}


def _prepare_runner(args: argparse.Namespace, basic_ts_dir: Path):
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["BASICTS_RUN_TAG"] = args.run_tag
    os.environ["BASICTS_SEED"] = str(args.seed)
    os.environ["DYNAMIC_GRAPH_MODE"] = args.dynamic_mode
    os.environ["DYNAMIC_GRAPH_WEIGHT_MODE"] = args.weight_mode
    os.environ["DYNAMIC_GRAPH_RADIUS_PARAM"] = args.radius_param
    os.environ["DYNAMIC_GRAPH_STATE_ACT"] = args.state_act
    os.environ["DYNAMIC_GRAPH_QUANTILE_TEMPERATURE"] = str(args.quantile_temperature)
    os.environ["DYNAMIC_GRAPH_DEGREE_MIN"] = str(args.degree_min)
    os.environ["DYNAMIC_GRAPH_DEGREE_MAX"] = str(args.degree_max)
    os.environ["DYNAMIC_GRAPH_DEGREE_INIT"] = str(args.degree_init)
    os.environ["DYNAMIC_GWNET_ADDAPTADJ"] = "1" if args.dynamic_addaptadj else "0"
    if args.data_name:
        os.environ["BASICTS_DATA_NAME"] = args.data_name
    if args.graph_tag:
        os.environ["BASICTS_GRAPH_TAG"] = args.graph_tag
    if args.dist_mtx:
        os.environ["DYNAMIC_GRAPH_DIST_MTX"] = args.dist_mtx
    if args.candidate_adj:
        os.environ["DYNAMIC_GRAPH_CANDIDATE_ADJ"] = args.candidate_adj
    if args.target_avg_degree is not None:
        os.environ["DYNAMIC_GRAPH_TARGET_AVG_DEGREE"] = str(args.target_avg_degree)

    sys.path.insert(0, str(basic_ts_dir))
    os.chdir(basic_ts_dir)

    from easytorch.config import init_cfg  # pylint: disable=import-outside-toplevel
    from easytorch.device import set_device_type  # pylint: disable=import-outside-toplevel
    from easytorch.utils import set_visible_devices  # pylint: disable=import-outside-toplevel

    set_device_type("gpu" if args.device.startswith("cuda") else "cpu")
    if args.device.startswith("cuda"):
        set_visible_devices(args.gpu)

    cfg = init_cfg(args.cfg, save=False)
    runner = cfg["RUNNER"](cfg)
    runner.init_logger(logger_name="analyze-dynamic-threshold-time", log_file_name=None)
    runner.init_test(cfg)
    if args.split == "train":
        cfg.TRAIN.DATA.SHUFFLE = False
        runner.test_data_loader = runner.build_train_data_loader(cfg)
    elif args.split == "valid":
        runner.test_data_loader = runner.build_val_data_loader(cfg)
    elif args.split != "test":
        raise ValueError(f"Unsupported split: {args.split}")
    runner.load_model(args.ckpt, strict=args.strict_load)
    runner.model.eval()
    return cfg, runner


def _time_index_info(cfg: Any, split: str) -> tuple[int, int, int, int, int]:
    desc = json.loads((Path("datasets") / cfg.DATASET.NAME / "desc.json").read_text(encoding="utf-8"))
    total_len = int(desc["num_time_steps"])
    ratio = desc["regular_settings"]["TRAIN_VAL_TEST_RATIO"]
    valid_len = int(total_len * ratio[1])
    test_len = int(total_len * ratio[2])
    train_len = total_len - valid_len - test_len
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    frequency = int(desc.get("frequency (minutes)", 15))
    if split == "train":
        split_start = 0
    elif split == "valid":
        split_start = train_len
    else:
        split_start = train_len + valid_len
    steps_per_day = int(round(24 * 60 / frequency))
    return split_start, input_len, frequency, steps_per_day, int(desc["num_nodes"])


def _scan_dynamic_graph(args: argparse.Namespace, cfg: Any, runner: Any) -> dict[str, pd.DataFrame | dict]:
    module = getattr(runner.model, "dynamic_support", None)
    if module is None:
        raise ValueError("Loaded model does not expose `dynamic_support`.")

    split_start, input_len, frequency, steps_per_day, num_nodes = _time_index_info(cfg, args.split)
    dist_norm = cfg.MODEL.PARAM.get("dynamic_graph", {}).get("dist_norm", "max")
    dist_path = cfg.MODEL.PARAM.get("dynamic_graph", {}).get("dist_mtx_path")
    scale_m = _distance_scale_m(args.raw_distance or dist_path, dist_norm)

    device = next(runner.model.parameters()).device
    src = module.candidate_src.to(device)
    sample_rows: list[dict[str, float | int]] = []
    node_radius_sum = np.zeros(num_nodes, dtype=np.float64)
    node_radius_sq_sum = np.zeros(num_nodes, dtype=np.float64)
    node_degree_sum = np.zeros(num_nodes, dtype=np.float64)
    node_degree_sq_sum = np.zeros(num_nodes, dtype=np.float64)
    tod_radius_sum = np.zeros((steps_per_day, num_nodes), dtype=np.float64)
    tod_degree_sum = np.zeros((steps_per_day, num_nodes), dtype=np.float64)
    tod_count = np.zeros(steps_per_day, dtype=np.float64)
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(runner.test_data_loader, desc="scan dynamic graph")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            graph_batch = runner.preprocessing(_clone_batch(batch))
            history = runner.to_running_device(graph_batch["inputs"])
            history = runner.select_input_features(history)
            radius = module._node_radius(history)  # pylint: disable=protected-access
            edge_weight = module._candidate_edge_weights(history)  # pylint: disable=protected-access
            active = (edge_weight > 0).float()
            src_batch = src.to(active.device)
            degree = edge_weight.new_zeros(active.shape[0], num_nodes)
            degree.scatter_add_(1, src_batch.unsqueeze(0).expand(active.shape[0], -1), active)

            radius_np = radius.detach().cpu().numpy().astype("float64")
            degree_np = degree.detach().cpu().numpy().astype("float64")
            batch_size = radius_np.shape[0]
            for local_idx in range(batch_size):
                sample_idx = sample_count + local_idx
                history_end_step = split_start + sample_idx + input_len - 1
                forecast_start_step = split_start + sample_idx + input_len
                tod_idx = history_end_step % steps_per_day
                radius_row = radius_np[local_idx]
                degree_row = degree_np[local_idx]
                sample_rows.append(
                    {
                        "sample_index": sample_idx,
                        "history_end_step": history_end_step,
                        "forecast_start_step": forecast_start_step,
                        "time_of_day_index": int(tod_idx),
                        "time_of_day_hour": float(tod_idx * frequency / 60.0),
                        "radius_norm_mean": float(np.mean(radius_row)),
                        "radius_norm_p10": float(np.percentile(radius_row, 10)),
                        "radius_norm_median": float(np.percentile(radius_row, 50)),
                        "radius_norm_p90": float(np.percentile(radius_row, 90)),
                        "radius_m_mean": float(np.mean(radius_row) * scale_m),
                        "radius_m_p10": float(np.percentile(radius_row, 10) * scale_m),
                        "radius_m_median": float(np.percentile(radius_row, 50) * scale_m),
                        "radius_m_p90": float(np.percentile(radius_row, 90) * scale_m),
                        "active_avg_out_degree": float(np.mean(degree_row)),
                        "active_edges": float(np.sum(degree_row)),
                        "degree_p10": float(np.percentile(degree_row, 10)),
                        "degree_median": float(np.percentile(degree_row, 50)),
                        "degree_p90": float(np.percentile(degree_row, 90)),
                    }
                )
                tod_radius_sum[tod_idx] += radius_row
                tod_degree_sum[tod_idx] += degree_row
                tod_count[tod_idx] += 1

            node_radius_sum += radius_np.sum(axis=0)
            node_radius_sq_sum += np.square(radius_np).sum(axis=0)
            node_degree_sum += degree_np.sum(axis=0)
            node_degree_sq_sum += np.square(degree_np).sum(axis=0)
            sample_count += batch_size

    sample_df = pd.DataFrame(sample_rows)
    denom = max(float(sample_count), 1.0)
    radius_mean = node_radius_sum / denom
    degree_mean = node_degree_sum / denom
    radius_std = np.sqrt(np.maximum(node_radius_sq_sum / denom - np.square(radius_mean), 0.0))
    degree_std = np.sqrt(np.maximum(node_degree_sq_sum / denom - np.square(degree_mean), 0.0))
    node_df = pd.DataFrame(
        {
            "node_index": np.arange(num_nodes),
            "radius_norm_mean": radius_mean,
            "radius_norm_std": radius_std,
            "radius_m_mean": radius_mean * scale_m,
            "radius_m_std": radius_std * scale_m,
            "active_out_degree_mean": degree_mean,
            "active_out_degree_std": degree_std,
        }
    )

    safe_tod_count = np.maximum(tod_count, 1.0)
    tod_radius_mean = tod_radius_sum / safe_tod_count[:, None]
    tod_degree_mean = tod_degree_sum / safe_tod_count[:, None]
    tod_rows = []
    for idx in range(steps_per_day):
        tod_rows.append(
            {
                "time_of_day_index": idx,
                "time_of_day_hour": float(idx * frequency / 60.0),
                "sample_count": int(tod_count[idx]),
                "radius_norm_mean": float(np.mean(tod_radius_mean[idx])),
                "radius_norm_p10_node": float(np.percentile(tod_radius_mean[idx], 10)),
                "radius_norm_median_node": float(np.percentile(tod_radius_mean[idx], 50)),
                "radius_norm_p90_node": float(np.percentile(tod_radius_mean[idx], 90)),
                "radius_m_mean": float(np.mean(tod_radius_mean[idx]) * scale_m),
                "radius_m_median_node": float(np.percentile(tod_radius_mean[idx], 50) * scale_m),
                "active_avg_out_degree": float(np.mean(tod_degree_mean[idx])),
                "degree_median_node": float(np.percentile(tod_degree_mean[idx], 50)),
                "degree_p90_node": float(np.percentile(tod_degree_mean[idx], 90)),
            }
        )
    tod_df = pd.DataFrame(tod_rows)

    node_tod_parts = []
    top_nodes = np.argsort(-node_df["radius_norm_std"].to_numpy())[: args.heatmap_nodes]
    for node in top_nodes:
        node_tod_parts.append(
            pd.DataFrame(
                {
                    "node_index": int(node),
                    "time_of_day_index": np.arange(steps_per_day),
                    "time_of_day_hour": np.arange(steps_per_day) * frequency / 60.0,
                    "radius_norm_mean": tod_radius_mean[:, node],
                    "radius_m_mean": tod_radius_mean[:, node] * scale_m,
                    "active_out_degree_mean": tod_degree_mean[:, node],
                }
            )
        )
    node_tod_df = pd.concat(node_tod_parts, ignore_index=True) if node_tod_parts else pd.DataFrame()

    summary = {
        "split": args.split,
        "samples": int(sample_count),
        "num_nodes": int(num_nodes),
        "frequency_minutes": int(frequency),
        "steps_per_day": int(steps_per_day),
        "distance_norm": str(dist_norm),
        "radius_meter_scale": float(scale_m),
        "radius_norm": _summary(node_df["radius_norm_mean"].to_numpy()),
        "radius_m": _summary(node_df["radius_m_mean"].to_numpy()),
        "active_out_degree": _summary(node_df["active_out_degree_mean"].to_numpy()),
        "time_of_day_radius_m_range": float(tod_df["radius_m_mean"].max() - tod_df["radius_m_mean"].min()),
        "time_of_day_degree_range": float(tod_df["active_avg_out_degree"].max() - tod_df["active_avg_out_degree"].min()),
        "top_radius_variation_nodes": [
            {
                "node_index": int(row.node_index),
                "radius_m_mean": float(row.radius_m_mean),
                "radius_m_std": float(row.radius_m_std),
                "active_out_degree_mean": float(row.active_out_degree_mean),
            }
            for row in node_df.sort_values("radius_m_std", ascending=False).head(20).itertuples()
        ],
    }
    return {
        "sample": sample_df,
        "node": node_df,
        "time_of_day": tod_df,
        "node_time_of_day": node_tod_df,
        "summary": summary,
    }


def _plot_outputs(output_dir: Path, tables: dict[str, pd.DataFrame | dict], smooth_window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    sample = tables["sample"]
    node = tables["node"]
    tod = tables["time_of_day"]
    node_tod = tables["node_time_of_day"]
    assert isinstance(sample, pd.DataFrame)
    assert isinstance(node, pd.DataFrame)
    assert isinstance(tod, pd.DataFrame)
    assert isinstance(node_tod, pd.DataFrame)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = sample["sample_index"].to_numpy()
    radius = _moving_average(sample["radius_m_mean"].to_numpy(), smooth_window)
    degree = _moving_average(sample["active_avg_out_degree"].to_numpy(), smooth_window)
    ax1.plot(x, radius, color="#2563eb", linewidth=1.2, label="mean radius (m)")
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Mean radius threshold (m)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(x, degree, color="#dc2626", linewidth=1.0, alpha=0.78, label="avg active degree")
    ax2.set_ylabel("Average active out-degree", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_radius_degree_over_test_samples.png", dpi=240)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(tod["time_of_day_hour"], tod["radius_m_mean"], marker="o", markersize=2.5, color="#2563eb")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean radius threshold (m)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(tod["time_of_day_hour"], tod["active_avg_out_degree"], marker="o", markersize=2.5, color="#dc2626")
    ax2.set_ylabel("Average active out-degree", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax1.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_radius_degree_by_time_of_day.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(node["active_out_degree_mean"], bins=35, alpha=0.82, color="#4c78a8")
    ax.set_xlabel("Mean active out-degree")
    ax.set_ylabel("Node count")
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_active_degree_distribution.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(
        node["radius_m_mean"],
        node["active_out_degree_mean"],
        c=node["radius_m_std"],
        s=18,
        alpha=0.72,
        cmap="viridis",
    )
    ax.set_xlabel("Mean radius threshold (m)")
    ax.set_ylabel("Mean active out-degree")
    fig.colorbar(sc, ax=ax, label="Radius std (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "node_radius_vs_active_degree.png", dpi=240)
    plt.close(fig)

    if not node_tod.empty:
        pivot = node_tod.pivot(index="node_index", columns="time_of_day_index", values="radius_m_mean")
        fig, ax = plt.subplots(figsize=(10, max(4, min(12, 0.16 * len(pivot)))))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        ax.set_xlabel("Time-of-day index")
        ax.set_ylabel("Top radius-varying nodes")
        ax.set_yticks(np.arange(len(pivot)))
        ax.set_yticklabels([str(idx) for idx in pivot.index], fontsize=6)
        fig.colorbar(image, ax=ax, label="Radius threshold (m)")
        fig.tight_layout()
        fig.savefig(output_dir / "node_radius_time_of_day_heatmap_topvarying.png", dpi=240)
        plt.close(fig)


def main() -> None:
    repo_root = _repo_root()
    basic_ts = _basic_ts_dir(repo_root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="baselines/GWNet/SD_dynamic_threshold.py")
    parser.add_argument("--ckpt", default=str(basic_ts / "checkpoints/DynamicThresholdGraphWaveNet/SD_dynamic_threshold_hard_exp_tanh_gwnet_100_12_12_addaptadj_dynamic_threshold_hard_addaptadj_20260522_gpu1/94d620cfd224e6d2cf7cd2c344b1a3cd/DynamicThresholdGraphWaveNet_best_val_MAE.pt"))
    parser.add_argument("--run-tag", default="dynamic_threshold_hard_addaptadj_20260522_gpu1")
    parser.add_argument("--output-dir", default=str(repo_root / "mvp_experiments/active/adaptive_threshold_dynamic_weight/results/gwnet_dynamic_threshold_time_evolution_20260529"))
    parser.add_argument("--raw-distance", default="/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/distance_matrices/SD/SD_osrm_shortest_distance_m.npy")
    parser.add_argument("--candidate-adj", default="")
    parser.add_argument("--target-avg-degree", type=float, default=None)
    parser.add_argument("--dynamic-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--weight-mode", choices=("binary", "gaussian"), default="binary")
    parser.add_argument("--radius-param", choices=("exp_tanh", "softplus", "quantile", "degree_quantile"), default="exp_tanh")
    parser.add_argument("--state-act", choices=("tanh", "identity"), default="tanh")
    parser.add_argument("--quantile-temperature", type=float, default=1.0)
    parser.add_argument("--degree-min", type=float, default=8.0)
    parser.add_argument("--degree-max", type=float, default=64.0)
    parser.add_argument("--degree-init", type=float, default=32.0)
    parser.add_argument("--data-name", default="")
    parser.add_argument("--graph-tag", default="")
    parser.add_argument("--dist-mtx", default="")
    parser.add_argument("--dynamic-addaptadj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--heatmap-nodes", type=int, default=60)
    parser.add_argument("--smooth-window", type=int, default=96)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    args.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, runner = _prepare_runner(args, basic_ts)
    tables = _scan_dynamic_graph(args, cfg, runner)

    tables["sample"].to_csv(out_dir / "sample_time_series.csv", index=False)
    tables["node"].to_csv(out_dir / "node_dynamic_threshold_summary.csv", index=False)
    tables["time_of_day"].to_csv(out_dir / "time_of_day_summary.csv", index=False)
    tables["node_time_of_day"].to_csv(out_dir / "node_time_of_day_summary_topvarying.csv", index=False)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(tables["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_outputs(out_dir, tables, args.smooth_window)
    print(json.dumps({"output_dir": str(out_dir), **tables["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

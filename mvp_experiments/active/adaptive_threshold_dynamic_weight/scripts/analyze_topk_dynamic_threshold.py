#!/usr/bin/env python3
"""Analyze fixed K64 priors versus dynamic-threshold variants on SD.

The script streams BasicTS test predictions from best-val checkpoints and avoids
saving full prediction tensors. For dynamic-threshold models it also measures
the actually active candidate edges on the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GRAPH_SPECS = {
    "osrmK64": {
        "fixed_dataset": "SD_OSRMTOPK_K064",
        "candidate_dataset": "SD_OSRMTOPK_K064",
        "distance_label": "OSRM K64",
    },
    "gspK64": {
        "fixed_dataset": "SD_GSPTOPK_K064",
        "candidate_dataset": "SD_GSPTOPK_K064",
        "distance_label": "GSP K64",
    },
}

VARIANT_SPECS = {
    "fixed": {
        "config": "baselines/GWNet/SD_topk_prior_largest_aligned.py",
        "model_name": "GraphWaveNet",
        "dataset_source": "fixed_dataset",
    },
    "dynamic": {
        "config": "baselines/GWNet/SD_dynamic_threshold_largest_aligned.py",
        "model_name": "DynamicThresholdGraphWaveNet",
        "dataset_source": "SD",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basicts-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graphs", default="osrmK64,gspK64")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device-type", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit test batches for quick smoke checks.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_adj_payload(path: Path) -> tuple[list[str] | None, np.ndarray]:
    with path.open("rb") as fp:
        obj: Any = pickle.load(fp, encoding="latin1")
    node_ids = None
    if isinstance(obj, (tuple, list)):
        if len(obj) >= 3:
            node_ids = [str(item) for item in obj[0]]
            obj = obj[2]
        elif len(obj) == 1:
            obj = obj[0]
    return node_ids, np.asarray(obj)


def load_support_stats(basicts_dir: Path, dataset: str, graph: str) -> pd.DataFrame:
    node_ids, adj = load_adj_payload(basicts_dir / "datasets" / dataset / "adj_mx.pkl")
    support = np.asarray(adj) > 0
    np.fill_diagonal(support, False)
    out_degree = support.sum(axis=1).astype(float)
    in_degree = support.sum(axis=0).astype(float)
    out_weight = np.where(support, adj, 0.0).sum(axis=1)
    in_weight = np.where(support, adj, 0.0).sum(axis=0)
    num_nodes = support.shape[0]
    return pd.DataFrame(
        {
            "graph": graph,
            "dataset": dataset,
            "node_index": np.arange(num_nodes),
            "node_id": node_ids if node_ids is not None else [str(i) for i in range(num_nodes)],
            "fixed_out_degree": out_degree,
            "fixed_in_degree": in_degree,
            "fixed_out_weight_sum": out_weight,
            "fixed_in_weight_sum": in_weight,
        }
    )


def quantile_row(values: np.ndarray, prefix: str) -> dict[str, float]:
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    vals = np.percentile(values, qs)
    return {f"{prefix}_p{q:02d}": float(v) for q, v in zip(qs, vals)}


def clone_batch(data: dict) -> dict:
    cloned = {}
    for key, value in data.items():
        cloned[key] = value.clone() if hasattr(value, "clone") else value
    return cloned


def setup_env(args: argparse.Namespace, graph: str, variant: str) -> None:
    graph_spec = GRAPH_SPECS[graph]
    variant_spec = VARIANT_SPECS[variant]
    for name in list(os.environ):
        if name.startswith("DYNAMIC_GRAPH_") or name.startswith("DYNAMIC_GWNET_"):
            os.environ.pop(name, None)
    dataset_name = graph_spec[variant_spec["dataset_source"]] if variant != "dynamic" else "SD"
    os.environ["BASICTS_DATA_NAME"] = dataset_name
    os.environ["BASICTS_GRAPH_TAG"] = graph
    os.environ["BASICTS_RUN_TAG"] = args.run_tag
    os.environ.setdefault("BASICTS_SEED", "2023")
    os.environ.setdefault("BASICTS_NUM_EPOCHS", "100")
    os.environ.setdefault("BASICTS_PATIENCE", "30")
    os.environ.setdefault("BASICTS_BATCH_SIZE", "64")
    os.environ["WANDB_MODE"] = "disabled"
    if variant == "dynamic":
        os.environ["DYNAMIC_GRAPH_MODE"] = "hard"
        os.environ["DYNAMIC_GRAPH_WEIGHT_MODE"] = "binary"
        os.environ["DYNAMIC_GRAPH_TARGET_AVG_DEGREE"] = "64"
        os.environ["DYNAMIC_GRAPH_CANDIDATE_ADJ"] = f"datasets/{graph_spec['candidate_dataset']}/adj_mx.pkl"
        os.environ["DYNAMIC_GWNET_ADDAPTADJ"] = "0"


def init_runner(args: argparse.Namespace, graph: str, variant: str):
    setup_env(args, graph, variant)
    sys.path.insert(0, str(args.basicts_dir))
    os.chdir(args.basicts_dir)

    from easytorch.config import init_cfg  # pylint: disable=import-outside-toplevel
    from easytorch.device import set_device_type  # pylint: disable=import-outside-toplevel
    from easytorch.utils import set_visible_devices  # pylint: disable=import-outside-toplevel

    set_device_type(args.device_type)
    if args.device_type != "cpu":
        set_visible_devices(args.gpu)

    cfg = init_cfg(VARIANT_SPECS[variant]["config"], save=False)
    runner = cfg["RUNNER"](cfg)
    runner.init_logger(logger_name=f"analyze-{graph}-{variant}", log_file_name=None)
    runner.init_test(cfg)
    return cfg, runner


def find_checkpoint(basicts_dir: Path, cfg: Any) -> Path | None:
    ckpt_root = basicts_dir / cfg.TRAIN.CKPT_SAVE_DIR
    model_name = str(cfg.MODEL.NAME)
    matches = sorted(ckpt_root.glob(f"*/{model_name}_best_val_MAE.pt"))
    return matches[-1] if matches else None


def update_metric_sums(sums: dict[str, np.ndarray], pred: np.ndarray, target: np.ndarray, null_val: float) -> None:
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
    sums["sum_abs"] += np.where(mask, abs_err, 0.0).sum(axis=axes)
    sums["sum_sq"] += np.where(mask, sq_err, 0.0).sum(axis=axes)
    sums["sum_ape"] += np.where(mask, ape, 0.0).sum(axis=axes)
    sums["sum_target_abs"] += np.where(mask, np.abs(target), 0.0).sum(axis=axes)
    sums["count"] += mask.sum(axis=axes)


def update_dynamic_sums(runner: Any, data: dict, dynamic: dict[str, np.ndarray]) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    module = runner.model
    dyn = getattr(module, "dynamic_support", None)
    if dyn is None:
        return
    graph_data = runner.preprocessing(clone_batch(data))
    history = runner.to_running_device(graph_data["inputs"])
    history = runner.select_input_features(history)
    with torch.no_grad():
        radius = dyn._node_radius(history)  # pylint: disable=protected-access
        edge_weight = dyn._candidate_edge_weights(history)  # pylint: disable=protected-access
        active = (edge_weight > 0).float()
        src = dyn.candidate_src.to(edge_weight.device)
        batch_size = active.shape[0]
        degree = edge_weight.new_zeros(batch_size, dyn.num_nodes)
        degree.scatter_add_(1, src.unsqueeze(0).expand(batch_size, -1), active)
    degree_np = degree.detach().cpu().numpy()
    radius_np = radius.detach().cpu().numpy()
    dynamic["degree_sum"] += degree_np.sum(axis=0)
    dynamic["degree_sq_sum"] += np.square(degree_np).sum(axis=0)
    dynamic["radius_sum"] += radius_np.sum(axis=0)
    dynamic["radius_sq_sum"] += np.square(radius_np).sum(axis=0)
    dynamic["samples"] += degree_np.shape[0]


def evaluate_variant(args: argparse.Namespace, graph: str, variant: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    out_metrics = args.output_dir / f"node_metrics_gwnet_{graph}_{variant}.csv"
    out_dynamic = args.output_dir / f"dynamic_degree_gwnet_{graph}_{variant}.csv"
    if args.skip_existing and out_metrics.exists() and (variant != "dynamic" or out_dynamic.exists()):
        metrics = pd.read_csv(out_metrics)
        dynamic = pd.read_csv(out_dynamic) if out_dynamic.exists() else None
        return metrics, dynamic

    cfg, runner = init_runner(args, graph, variant)
    checkpoint = find_checkpoint(args.basicts_dir, cfg)
    if checkpoint is None:
        print(json.dumps({"graph": graph, "variant": variant, "status": "missing_checkpoint"}))
        return None, None
    runner.load_model(str(checkpoint), strict=True)
    runner.model.eval()

    num_nodes = int(cfg.MODEL.PARAM.get("num_nodes", cfg.MODEL.PARAM.get("n_vertex")))
    sums = {
        "sum_abs": np.zeros(num_nodes, dtype=np.float64),
        "sum_sq": np.zeros(num_nodes, dtype=np.float64),
        "sum_ape": np.zeros(num_nodes, dtype=np.float64),
        "sum_target_abs": np.zeros(num_nodes, dtype=np.float64),
        "count": np.zeros(num_nodes, dtype=np.float64),
    }
    dynamic_sums = None
    if variant == "dynamic":
        dynamic_sums = {
            "degree_sum": np.zeros(num_nodes, dtype=np.float64),
            "degree_sq_sum": np.zeros(num_nodes, dtype=np.float64),
            "radius_sum": np.zeros(num_nodes, dtype=np.float64),
            "radius_sq_sum": np.zeros(num_nodes, dtype=np.float64),
            "samples": 0,
        }

    import torch  # pylint: disable=import-outside-toplevel

    with torch.no_grad():
        for batch_idx, data in enumerate(runner.test_data_loader):
            if args.sample_limit and batch_idx >= args.sample_limit:
                break
            if dynamic_sums is not None:
                update_dynamic_sums(runner, data, dynamic_sums)
            forward_return = runner.forward(clone_batch(data), epoch=None, iter_num=None, train=False)
            pred = forward_return["prediction"].detach().cpu().numpy()
            target = forward_return["target"].detach().cpu().numpy()
            update_metric_sums(sums, pred, target, float(cfg.METRICS.NULL_VAL))

    safe_count = np.maximum(sums["count"], 1.0)
    metrics = pd.DataFrame(
        {
            "model": "gwnet",
            "graph": graph,
            "variant": variant,
            "node_index": np.arange(num_nodes),
            "valid_count": sums["count"],
            "mae": sums["sum_abs"] / safe_count,
            "rmse": np.sqrt(sums["sum_sq"] / safe_count),
            "mape": sums["sum_ape"] / safe_count,
            "wape": sums["sum_abs"] / (sums["sum_target_abs"] + 5e-5),
            "checkpoint": str(checkpoint),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_metrics, index=False)

    dynamic_df = None
    if dynamic_sums is not None and dynamic_sums["samples"] > 0:
        samples = float(dynamic_sums["samples"])
        degree_mean = dynamic_sums["degree_sum"] / samples
        radius_mean = dynamic_sums["radius_sum"] / samples
        degree_var = dynamic_sums["degree_sq_sum"] / samples - np.square(degree_mean)
        radius_var = dynamic_sums["radius_sq_sum"] / samples - np.square(radius_mean)
        dynamic_df = pd.DataFrame(
            {
                "model": "gwnet",
                "graph": graph,
                "variant": variant,
                "node_index": np.arange(num_nodes),
                "dynamic_out_degree_mean": degree_mean,
                "dynamic_out_degree_std": np.sqrt(np.maximum(degree_var, 0.0)),
                "dynamic_radius_mean": radius_mean,
                "dynamic_radius_std": np.sqrt(np.maximum(radius_var, 0.0)),
                "num_samples": int(dynamic_sums["samples"]),
                "checkpoint": str(checkpoint),
            }
        )
        dynamic_df.to_csv(out_dynamic, index=False)

    print(json.dumps({"graph": graph, "variant": variant, "status": "ok", "checkpoint": str(checkpoint)}))
    return metrics, dynamic_df


def load_meta(basicts_dir: Path) -> pd.DataFrame:
    path = basicts_dir / "datasets" / "SD" / "meta.csv"
    if not path.exists():
        return pd.DataFrame()
    meta = pd.read_csv(path)
    meta = meta.rename(columns={"ID": "sensor_id", "Lat": "lat", "Lng": "lng"})
    meta["node_index"] = np.arange(len(meta))
    return meta


def summarize_graphs(args: argparse.Namespace, graphs: list[str]) -> pd.DataFrame:
    rows = []
    for graph in graphs:
        fixed = load_support_stats(args.basicts_dir, GRAPH_SPECS[graph]["fixed_dataset"], graph)
        row = {
            "graph": graph,
            "dataset": GRAPH_SPECS[graph]["fixed_dataset"],
            "edge_count": int(fixed["fixed_out_degree"].sum()),
            "avg_out_degree": float(fixed["fixed_out_degree"].mean()),
            "isolated_out_nodes": int((fixed["fixed_out_degree"] == 0).sum()),
            "out_degree_std": float(fixed["fixed_out_degree"].std(ddof=0)),
        }
        row.update(quantile_row(fixed["fixed_out_degree"].to_numpy(), "out_degree"))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "topk_graph_degree_summary.csv", index=False)
    return out


def build_pair_tables(args: argparse.Namespace, graph: str) -> pd.DataFrame | None:
    fixed_path = args.output_dir / f"node_metrics_gwnet_{graph}_fixed.csv"
    dyn_path = args.output_dir / f"node_metrics_gwnet_{graph}_dynamic.csv"
    if not fixed_path.exists() or not dyn_path.exists():
        return None
    fixed = pd.read_csv(fixed_path)
    dynamic = pd.read_csv(dyn_path)
    support = load_support_stats(args.basicts_dir, GRAPH_SPECS[graph]["fixed_dataset"], graph)
    dyn_degree_path = args.output_dir / f"dynamic_degree_gwnet_{graph}_dynamic.csv"
    dyn_degree = pd.read_csv(dyn_degree_path) if dyn_degree_path.exists() else pd.DataFrame({"node_index": []})
    meta = load_meta(args.basicts_dir)

    merged = fixed.merge(dynamic, on=["node_index"], suffixes=("_fixed", "_dynamic"))
    for metric in ["mae", "rmse", "mape", "wape"]:
        merged[f"delta_{metric}_dynamic_minus_fixed"] = merged[f"{metric}_dynamic"] - merged[f"{metric}_fixed"]
        denom = merged[f"{metric}_fixed"].replace(0, np.nan)
        merged[f"rel_delta_{metric}_dynamic_minus_fixed"] = merged[f"delta_{metric}_dynamic_minus_fixed"] / denom
    merged = merged.merge(support, on="node_index", how="left")
    if not dyn_degree.empty:
        merged = merged.merge(dyn_degree.drop(columns=["model", "variant", "checkpoint"], errors="ignore"), on=["node_index", "graph"], how="left")
        merged["delta_out_degree_dynamic_minus_fixed"] = (
            merged["dynamic_out_degree_mean"] - merged["fixed_out_degree"]
        )
    if not meta.empty:
        merged = merged.merge(meta, on="node_index", how="left")
    out_path = args.output_dir / f"delta_node_metrics_gwnet_{graph}_dynamic_vs_fixed.csv"
    merged.to_csv(out_path, index=False)
    return merged


def make_plots(args: argparse.Namespace, graph: str, merged: pd.DataFrame) -> None:
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    label = GRAPH_SPECS[graph]["distance_label"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(merged["fixed_out_degree"], bins=30, alpha=0.45, label="fixed candidate degree")
    if "dynamic_out_degree_mean" in merged:
        ax.hist(merged["dynamic_out_degree_mean"], bins=30, alpha=0.45, label="dynamic active degree")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("Node count")
    ax.set_title(f"{label}: fixed vs dynamic degree distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"degree_distribution_gwnet_{graph}_dynamic_vs_fixed.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(merged["mae_fixed"], bins=35, alpha=0.45, label="fixed")
    ax.hist(merged["mae_dynamic"], bins=35, alpha=0.45, label="dynamic")
    ax.set_xlabel("Node MAE")
    ax.set_ylabel("Node count")
    ax.set_title(f"{label}: node MAE distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"mae_distribution_gwnet_{graph}_dynamic_vs_fixed.png", dpi=240)
    plt.close(fig)

    delta = merged["delta_mae_dynamic_minus_fixed"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.hist(delta, bins=40, color="#4c78a8", alpha=0.82)
    ax.set_xlabel("Delta node MAE (dynamic - fixed)")
    ax.set_ylabel("Node count")
    ax.set_title(f"{label}: MAE delta distribution")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"mae_delta_distribution_gwnet_{graph}_dynamic_vs_fixed.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    color = np.where(delta <= 0, "#2ca25f", "#de2d26")
    x_col = "dynamic_out_degree_mean" if "dynamic_out_degree_mean" in merged else "fixed_out_degree"
    ax.scatter(merged[x_col], delta, c=color, alpha=0.68, s=18, linewidths=0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Delta node MAE (dynamic - fixed)")
    ax.set_title(f"{label}: degree vs MAE delta")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"degree_vs_mae_delta_gwnet_{graph}_dynamic_vs_fixed.png", dpi=240)
    plt.close(fig)


def color_for_delta(value: float, limit: float) -> str:
    if not math.isfinite(value):
        return "#999999"
    if value <= -limit:
        return "#1a9850"
    if value < -limit / 3:
        return "#66bd63"
    if value <= limit / 3:
        return "#f7f7f7"
    if value < limit:
        return "#f46d43"
    return "#d73027"


def bin_for_delta(value: float, limit: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    if value <= -limit:
        return "strong_improve"
    if value < -limit / 3:
        return "mild_improve"
    if value <= limit / 3:
        return "near_tie"
    if value < limit:
        return "mild_worse"
    return "strong_worse"


def make_delta_map(args: argparse.Namespace, graph: str, merged: pd.DataFrame) -> None:
    required = {"lat", "lng", "delta_mae_dynamic_minus_fixed"}
    if not required.issubset(merged.columns):
        return
    rows = merged.dropna(subset=["lat", "lng"]).copy()
    if rows.empty:
        return
    abs_delta = np.abs(rows["delta_mae_dynamic_minus_fixed"].to_numpy())
    limit = float(np.nanpercentile(abs_delta, 85)) if np.isfinite(abs_delta).any() else 1.0
    limit = max(limit, 1e-6)
    center_lat = float(rows["lat"].mean())
    center_lng = float(rows["lng"].mean())
    bin_order = ["strong_improve", "mild_improve", "near_tie", "mild_worse", "strong_worse"]
    bin_labels = {
        "strong_improve": "明显变好",
        "mild_improve": "轻微变好",
        "near_tie": "接近不变",
        "mild_worse": "轻微变差",
        "strong_worse": "明显变差",
    }
    points = []
    for row in rows.to_dict("records"):
        delta = float(row["delta_mae_dynamic_minus_fixed"])
        points.append(
            {
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "bin": bin_for_delta(delta, limit),
                "color": color_for_delta(delta, limit),
                "popup": (
                    f"node={int(row['node_index'])}<br>"
                    f"sensor={row.get('sensor_id', '')}<br>"
                    f"delta_mae={delta:.4f}<br>"
                    f"fixed_mae={float(row['mae_fixed']):.4f}<br>"
                    f"dynamic_mae={float(row['mae_dynamic']):.4f}<br>"
                    f"fixed_degree={float(row.get('fixed_out_degree', math.nan)):.2f}<br>"
                    f"dynamic_degree={float(row.get('dynamic_out_degree_mean', math.nan)):.2f}<br>"
                    f"fwy={row.get('Fwy', '')} dir={row.get('Direction', '')}"
                ),
            }
        )
    controls = "\n  ".join(
        [
            (
                f"<label><input type=\"checkbox\" checked "
                f"onchange=\"toggleBin('{bucket}', this.checked)\"> {bin_labels[bucket]}</label>"
            )
            for bucket in bin_order
        ]
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GWNet {graph} dynamic threshold delta map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .panel {{
      position: absolute; top: 12px; right: 12px; z-index: 999;
      background: white; padding: 10px 12px; border: 1px solid #ccc;
      font-family: sans-serif; font-size: 13px; line-height: 1.45;
    }}
    .panel label {{ display: block; }}
  </style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <b>{GRAPH_SPECS[graph]["distance_label"]}</b><br>
  delta = dynamic MAE - fixed MAE<br>
  green: dynamic better, red: dynamic worse<br>
  node radius is constant; color encodes delta.<br>
  <hr>
  {controls}
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{center_lat:.6f}, {center_lng:.6f}], 10);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);
const points = {json.dumps(points)};
const layers = {{}};
for (const bin of {json.dumps(bin_order)}) {{
  layers[bin] = L.layerGroup().addTo(map);
}}
for (const p of points) {{
  const circle = L.circleMarker([p.lat, p.lng], {{
    radius: 5,
    color: '#333333',
    weight: 0.4,
    fillColor: p.color,
    fillOpacity: 0.82
  }}).bindPopup(p.popup);
  layers[p.bin].addLayer(circle);
}}
function toggleBin(bin, enabled) {{
  if (enabled) {{
    layers[bin].addTo(map);
  }} else {{
    map.removeLayer(layers[bin]);
  }}
}}
</script>
</body>
</html>
"""
    (args.output_dir / f"map_mae_delta_gwnet_{graph}_dynamic_vs_fixed.html").write_text(html, encoding="utf-8")


def write_summary(args: argparse.Namespace, graphs: list[str], pair_tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for graph, table in pair_tables.items():
        if table is None or table.empty:
            continue
        row = {
            "graph": graph,
            "fixed_mae_mean": float(table["mae_fixed"].mean()),
            "dynamic_mae_mean": float(table["mae_dynamic"].mean()),
            "delta_mae_mean": float(table["delta_mae_dynamic_minus_fixed"].mean()),
            "improved_nodes": int((table["delta_mae_dynamic_minus_fixed"] < 0).sum()),
            "worse_nodes": int((table["delta_mae_dynamic_minus_fixed"] > 0).sum()),
            "tie_nodes": int((table["delta_mae_dynamic_minus_fixed"] == 0).sum()),
        }
        if "fixed_out_degree" in table:
            row["fixed_avg_out_degree"] = float(table["fixed_out_degree"].mean())
        if "dynamic_out_degree_mean" in table:
            row["dynamic_avg_active_out_degree"] = float(table["dynamic_out_degree_mean"].mean())
            row["mean_degree_delta"] = float(table["delta_out_degree_dynamic_minus_fixed"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "dynamic_vs_fixed_node_summary.csv", index=False)
    md_lines = ["# GWNet K64 Dynamic Threshold Analysis", ""]
    md_lines.append(f"- run_tag: `{args.run_tag}`")
    md_lines.append(f"- graphs: `{', '.join(graphs)}`")
    md_lines.append("")
    if not summary.empty:
        md_lines.append(summary.to_markdown(index=False))
        md_lines.append("")
    md_lines.append("Generated artifacts include per-node CSVs, degree/MAE/delta plots, and Leaflet delta maps.")
    (args.output_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graphs = [item.strip() for item in args.graphs.split(",") if item.strip()]
    unknown = [item for item in graphs if item not in GRAPH_SPECS]
    if unknown:
        raise ValueError(f"Unknown graph(s): {unknown}")

    summarize_graphs(args, graphs)
    manifest_rows = []
    for graph in graphs:
        for variant in ["fixed", "dynamic"]:
            metrics, dynamic = evaluate_variant(args, graph, variant)
            manifest_rows.append(
                {
                    "graph": graph,
                    "variant": variant,
                    "metrics_ready": metrics is not None,
                    "dynamic_stats_ready": dynamic is not None,
                }
            )
    with (args.output_dir / "analysis_manifest.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["graph", "variant", "metrics_ready", "dynamic_stats_ready"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    pair_tables = {}
    for graph in graphs:
        table = build_pair_tables(args, graph)
        pair_tables[graph] = table
        if table is None:
            continue
        make_plots(args, graph, table)
        make_delta_map(args, graph, table)
    write_summary(args, graphs, pair_tables)
    print(json.dumps({"output_dir": str(args.output_dir), "graphs": graphs}, indent=2))


if __name__ == "__main__":
    main()

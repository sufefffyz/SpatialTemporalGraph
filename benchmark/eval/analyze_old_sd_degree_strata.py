#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from old_sd_analysis_utils import (
    build_graph_matrices,
    compute_metrics_numpy,
    infer_num_samples,
    load_dataset_flow,
    load_sensor_ids,
    resolve_existing_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "eval" / "old_sd_degree_strata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze old-SD results by degree strata defined on the physical directed graph."
    )
    parser.add_argument("--sd-dir", type=Path, default=DEFAULT_SD_DIR, help="Old SD dataset directory.")
    parser.add_argument("--sd-phys-dir", type=Path, default=DEFAULT_SD_PHYS_DIR, help="Old SD_phys dataset directory.")
    parser.add_argument(
        "--run-dist",
        required=True,
        help="Run spec for distance-threshold result: label=/abs/path/to/checkpoint_dir",
    )
    parser.add_argument(
        "--run-phys",
        required=True,
        help="Run spec for physical-dir result: label=/abs/path/to/checkpoint_dir",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=[3, 6, 12],
        help="Horizons to report in addition to overall. Default: 3 6 12",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="Minimum node count required to keep one physical-degree group. Default: 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store outputs.",
    )
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected run spec label=/abs/path/to/checkpoint_dir, got: {spec}")
    label, raw_path = spec.split("=", 1)
    return label.strip(), resolve_existing_path(raw_path.strip(), f"checkpoint dir {label.strip()}")


def load_memmap_array(path: Path, output_len: int, num_nodes: int) -> np.memmap:
    num_samples = infer_num_samples(path, output_len, num_nodes, channels=1)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(num_samples, output_len, num_nodes, 1))


def load_run_predictions(ckpt_dir: Path, output_len: int, num_nodes: int) -> tuple[np.memmap, np.memmap]:
    pred_path = ckpt_dir / "test_results" / "predictions.npy"
    tgt_path = ckpt_dir / "test_results" / "targets.npy"
    if not pred_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_dir} missing test_results/predictions.npy or targets.npy. "
            "Please run BasicTS evaluate.py on the checkpoint first."
        )
    pred = load_memmap_array(pred_path, output_len, num_nodes)
    tgt = load_memmap_array(tgt_path, output_len, num_nodes)
    return pred, tgt


def compute_degree_tables(sd_dir: Path, sd_phys_dir: Path, sensor_ids: np.ndarray) -> pd.DataFrame:
    graph_matrices = build_graph_matrices(sd_dir, sd_phys_dir)
    dist_adj = graph_matrices["distthre"]
    phys_adj = graph_matrices["phys_dir"]

    phys_in = (phys_adj > 0).sum(axis=0).astype(int)
    phys_out = (phys_adj > 0).sum(axis=1).astype(int)
    phys_total = phys_in + phys_out

    dist_in = (dist_adj > 0).sum(axis=0).astype(int)
    dist_out = (dist_adj > 0).sum(axis=1).astype(int)
    dist_total = dist_in + dist_out

    return pd.DataFrame(
        {
            "node_index": np.arange(len(sensor_ids), dtype=int),
            "sensor_id": sensor_ids,
            "phys_in_degree": phys_in,
            "phys_out_degree": phys_out,
            "phys_total_degree": phys_total,
            "dist_in_degree": dist_in,
            "dist_out_degree": dist_out,
            "dist_total_degree": dist_total,
        }
    )


def evaluate_degree_group(
    pred: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray,
    null_val: float,
    horizons: list[int],
) -> list[dict]:
    rows = []
    group_pred = np.asarray(pred[:, :, mask, :])
    group_tgt = np.asarray(tgt[:, :, mask, :])
    overall = compute_metrics_numpy(group_pred, group_tgt, null_val)
    rows.append({"horizon": "overall", **overall})

    for horizon in horizons:
        idx = horizon - 1
        pred_h = np.asarray(group_pred[:, idx : idx + 1, :, :])
        tgt_h = np.asarray(group_tgt[:, idx : idx + 1, :, :])
        metrics_h = compute_metrics_numpy(pred_h, tgt_h, null_val)
        rows.append({"horizon": f"h{horizon}", **metrics_h})
    return rows


def main() -> None:
    args = parse_args()
    sd_dir = resolve_existing_path(args.sd_dir, "SD dataset dir")
    sd_phys_dir = resolve_existing_path(args.sd_phys_dir, "SD_phys dataset dir")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, desc = load_dataset_flow(sd_dir)
    num_nodes = int(desc["num_nodes"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    null_val = float(desc["regular_settings"].get("NULL_VAL", 0.0))
    sensor_ids = load_sensor_ids(sd_dir, num_nodes)

    dist_label, dist_ckpt = parse_run_spec(args.run_dist)
    phys_label, phys_ckpt = parse_run_spec(args.run_phys)
    pred_dist, tgt_dist = load_run_predictions(dist_ckpt, output_len, num_nodes)
    pred_phys, tgt_phys = load_run_predictions(phys_ckpt, output_len, num_nodes)

    degree_df = compute_degree_tables(sd_dir, sd_phys_dir, sensor_ids)
    degree_df.to_csv(output_dir / "node_degree_catalog.csv", index=False)

    rows = []
    for degree_value in sorted(degree_df["phys_total_degree"].unique().tolist()):
        group_df = degree_df[degree_df["phys_total_degree"] == degree_value].copy()
        if len(group_df) < args.min_nodes:
            continue
        mask = np.zeros(num_nodes, dtype=bool)
        mask[group_df["node_index"].to_numpy(dtype=int)] = True

        base_info = {
            "phys_total_degree": int(degree_value),
            "node_count": int(len(group_df)),
            "avg_dist_total_degree": float(group_df["dist_total_degree"].mean()),
            "avg_dist_in_degree": float(group_df["dist_in_degree"].mean()),
            "avg_dist_out_degree": float(group_df["dist_out_degree"].mean()),
            "avg_phys_in_degree": float(group_df["phys_in_degree"].mean()),
            "avg_phys_out_degree": float(group_df["phys_out_degree"].mean()),
        }

        for run_label, pred, tgt in [
            (dist_label, pred_dist, tgt_dist),
            (phys_label, pred_phys, tgt_phys),
        ]:
            for metric_row in evaluate_degree_group(pred, tgt, mask, null_val, args.horizons):
                rows.append(
                    {
                        "run": run_label,
                        **base_info,
                        **metric_row,
                    }
                )

    result_df = pd.DataFrame(rows).sort_values(["phys_total_degree", "horizon", "run"]).reset_index(drop=True)
    result_df.to_csv(output_dir / "degree_strata_metrics.csv", index=False)

    pivot_df = result_df.pivot_table(
        index=["phys_total_degree", "node_count", "avg_dist_total_degree", "horizon"],
        columns="run",
        values=["MAE", "RMSE", "MAPE"],
    )
    pivot_df = pivot_df.reset_index()
    pivot_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else str(col)
        for col in pivot_df.columns
    ]
    pivot_df.to_csv(output_dir / "degree_strata_metrics_pivot.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "degree_catalog_csv": str(output_dir / "node_degree_catalog.csv"),
        "metrics_csv": str(output_dir / "degree_strata_metrics.csv"),
        "pivot_csv": str(output_dir / "degree_strata_metrics_pivot.csv"),
        "rows": int(len(result_df)),
        "dist_run": dist_label,
        "phys_run": phys_label,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

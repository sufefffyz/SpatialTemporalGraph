#!/usr/bin/env python3
"""Plot same-date time-series comparisons between Xuancheng 1d and 30d datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        default="/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/datasets",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/yuzhang_fei/code/SpatialTemporalGraph/mvp_experiments/active/"
        "xuancheng_cityflow_pipeline/results/same_date_timeseries_compare",
    )
    parser.add_argument("--date", default="2023-04-03")
    parser.add_argument("--day-index", type=int, default=2, help="0-based day index in the 30d dataset.")
    parser.add_argument("--roads", default="34180205109_1,34180205108")
    return parser.parse_args()


def load_dataset(root: Path, name: str) -> tuple[np.ndarray, dict]:
    desc = json.loads((root / name / "desc.json").read_text(encoding="utf-8"))
    data = np.memmap(root / name / "data.dat", dtype="float32", mode="r", shape=tuple(desc["shape"]))
    return np.asarray(data[:, :, 0]), desc


def load_meta(root: Path, name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (root / name / "meta.csv").open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    return rows


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])


def save_stock_plot(
    out_dir: Path,
    date: str,
    time_hours: np.ndarray,
    stock_1d: np.ndarray,
    stock_30d_day: np.ndarray,
    meta: list[dict[str, str]],
    road_ids: list[str],
) -> dict[str, float | str]:
    road_to_idx = {row["road_id"]: int(row["node_index"]) for row in meta}
    colors = {"one": "#1f77b4", "thirty": "#d62728", "diff": "#2ca02c"}
    absdiff = np.abs(stock_1d - stock_30d_day)

    fig, axes = plt.subplots(2 + len(road_ids), 1, figsize=(9.5, 4.6 + 2.0 * len(road_ids)), sharex=True)
    axes[0].plot(time_hours, stock_1d.mean(axis=1), label="1d DTIGNN-style stock", color=colors["one"], linewidth=1.8)
    axes[0].plot(
        time_hours,
        stock_30d_day.mean(axis=1),
        label="30d roadagg same-date stock",
        color=colors["thirty"],
        linewidth=1.8,
    )
    axes[0].set_ylabel("network mean")
    axes[0].set_title(f"Xuancheng {date} stock time series: 1d vs 30d same-date slice")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    axes[1].plot(time_hours, absdiff.mean(axis=1), color=colors["diff"], linewidth=1.8)
    axes[1].set_ylabel("mean abs diff")

    for ax, road_id in zip(axes[2:], road_ids):
        idx = road_to_idx[road_id]
        ax.plot(time_hours, stock_1d[:, idx], label="1d DTIGNN-style", color=colors["one"], linewidth=1.6)
        ax.plot(time_hours, stock_30d_day[:, idx], label="30d roadagg same date", color=colors["thirty"], linewidth=1.6)
        row = meta[idx]
        ax.set_ylabel("road stock")
        ax.set_title(f"road {road_id} | len={float(row['length_m']):.0f}m, lanes={float(row['lane_count']):.0f}")
        ax.legend(frameon=False, ncol=2, loc="upper left")

    for ax in axes:
        style_axes(ax)
    axes[-1].set_xlabel("time of day (hour)")
    fig.tight_layout()

    png_path = out_dir / f"xuancheng_{date}_stock_1d_vs_30d_timeseries.png"
    pdf_path = out_dir / f"xuancheng_{date}_stock_1d_vs_30d_timeseries.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "stock_png": str(png_path),
        "stock_pdf": str(pdf_path),
        "stock_global_mae": float(absdiff.mean()),
        "stock_global_rmse": float(np.sqrt(np.mean((stock_1d - stock_30d_day) ** 2))),
    }


def save_flow_plot(
    out_dir: Path,
    date: str,
    time_hours: np.ndarray,
    flow_1d: np.ndarray,
    flow_30d_day: np.ndarray,
) -> dict[str, float | str]:
    colors = {"one": "#1f77b4", "thirty": "#d62728", "diff": "#2ca02c"}
    absdiff = np.abs(flow_1d - flow_30d_day)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.8), sharex=True)
    axes[0].plot(time_hours, flow_1d.mean(axis=1), label="1d DTIGNN-style flow", color=colors["one"], linewidth=1.8)
    axes[0].plot(
        time_hours,
        flow_30d_day.mean(axis=1),
        label="30d roadagg same-date flow",
        color=colors["thirty"],
        linewidth=1.8,
    )
    axes[0].set_ylabel("network mean flow")
    axes[0].set_title(f"Xuancheng {date} flow time series: 1d vs 30d same-date slice")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].plot(time_hours, absdiff.mean(axis=1), color=colors["diff"], linewidth=1.8)
    axes[1].set_ylabel("mean abs diff")

    for ax in axes:
        style_axes(ax)
    axes[-1].set_xlabel("time of day (hour)")
    fig.tight_layout()

    png_path = out_dir / f"xuancheng_{date}_flow_1d_vs_30d_timeseries.png"
    pdf_path = out_dir / f"xuancheng_{date}_flow_1d_vs_30d_timeseries.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "flow_png": str(png_path),
        "flow_pdf": str(pdf_path),
        "flow_global_mae": float(absdiff.mean()),
        "flow_global_rmse": float(np.sqrt(np.mean((flow_1d - flow_30d_day) ** 2))),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.datasets_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_1d, _ = load_dataset(root, "XCHENG_5MIN_STOCK")
    stock_30d, _ = load_dataset(root, "XCHENG30D_5MIN_STOCK")
    flow_1d, _ = load_dataset(root, "XCHENG_5MIN_FLOW")
    flow_30d, _ = load_dataset(root, "XCHENG30D_5MIN_FLOW")
    meta = load_meta(root, "XCHENG_5MIN_STOCK")

    steps_per_day = stock_1d.shape[0]
    start = args.day_index * steps_per_day
    end = start + steps_per_day
    time_hours = np.arange(steps_per_day) * 5 / 60.0
    road_ids = [road.strip() for road in args.roads.split(",") if road.strip()]

    summary: dict[str, object] = {
        "date": args.date,
        "day_index": args.day_index,
        "slice": [start, end],
        "roads": road_ids,
    }
    summary.update(save_stock_plot(out_dir, args.date, time_hours, stock_1d, stock_30d[start:end], meta, road_ids))
    summary.update(save_flow_plot(out_dir, args.date, time_hours, flow_1d, flow_30d[start:end]))

    summary_path = out_dir / f"xuancheng_{args.date}_1d_vs_30d_timeseries_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

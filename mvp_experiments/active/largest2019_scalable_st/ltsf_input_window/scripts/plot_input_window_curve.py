#!/usr/bin/env python3
"""Plot MAE/RMSE versus input window from collected LargeST-LTSF summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_metric(rows: list[dict], metric: str, out_path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric.lower(), "")
        if value in {"", "nan", "None"}:
            continue
        key = (row["dataset"], row["model"])
        grouped[key].append((int(row["input_len"]), float(value)))

    fig, axes = plt.subplots(1, max(1, len({k[0] for k in grouped})), figsize=(6 * max(1, len({k[0] for k in grouped})), 4), squeeze=False)
    datasets = sorted({k[0] for k in grouped}) or ["NA"]
    for ax, dataset in zip(axes[0], datasets):
        for (ds, model), points in sorted(grouped.items()):
            if ds != dataset:
                continue
            points = sorted(points)
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=model)
        ax.set_title(dataset)
        ax.set_xlabel("Input length")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", default="mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/outputs/figures")
    args = parser.parse_args()

    rows = read_rows(Path(args.summary))
    out_dir = Path(args.out_dir)
    plot_metric(rows, "mae", out_dir / "input_window_mae.png")
    plot_metric(rows, "rmse", out_dir / "input_window_rmse.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEY_METRICS = [
    "MAE",
    "RMSE",
    "WAPE",
    "occurrence_f1",
    "occurrence_AP_approx",
    "MAE_pos",
    "WAPE_pos",
    "high_f1",
    "high_AP_approx",
    "high_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect zero-aware MVP result JSON files into one CSV.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def rows_from_naive(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for system, metrics in data.get("baselines", {}).items():
        row = {
            "source": "naive",
            "dataset_name": data.get("dataset_name"),
            "node_filter": data.get("node_filter", "all"),
            "selected_nodes": data.get("node_filter_stats", {}).get("selected_nodes"),
            "system": system,
        }
        row.update({k: metrics.get(k) for k in KEY_METRICS})
        rows.append(row)
    return rows


def row_from_posthoc(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    row = {
        "source": "basicts",
        "dataset_name": data.get("dataset_name"),
        "node_filter": data.get("node_filter", "all"),
        "selected_nodes": data.get("node_filter_stats", {}).get("selected_nodes"),
        "system": data.get("system"),
    }
    row.update({k: metrics.get(k) for k in KEY_METRICS})
    return row


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted((args.results_root / "naive_baselines").glob("naive_metrics_*.json")):
        rows.extend(rows_from_naive(path))
    for path in sorted((args.results_root / "posthoc").glob("basicts_zero_metrics_*.json")):
        rows.append(row_from_posthoc(path))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["source", "dataset_name", "node_filter", "selected_nodes", "system", *KEY_METRICS],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

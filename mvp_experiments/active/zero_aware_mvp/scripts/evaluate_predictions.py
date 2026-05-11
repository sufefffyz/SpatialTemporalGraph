#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from zero_aware_mvp.lib.basicts_io import load_desc, test_prediction_starts, train_targets
from zero_aware_mvp.lib.metrics import MetricConfig, ZeroAwareMetricAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc zero-aware evaluation for BasicTS saved predictions.")
    parser.add_argument("--dataset-name", default="TRAFFIC_VOLUME_FULL_5MIN")
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--high-q", type=float, default=0.90)
    parser.add_argument("--min-positive-support", type=int, default=24)
    parser.add_argument("--system-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    desc = load_desc(args.dataset_name)
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    num_nodes = int(desc["num_nodes"])
    starts = test_prediction_starts(args.dataset_name)
    num_samples = int(len(starts))
    result_dir = args.ckpt_dir / "test_results"
    pred_path = result_dir / "predictions.npy"
    target_path = result_dir / "targets.npy"
    if not pred_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing BasicTS test_results under {result_dir}")

    pred = np.memmap(pred_path, dtype="float32", mode="r", shape=(num_samples, output_len, num_nodes, 1))
    target = np.memmap(target_path, dtype="float32", mode="r", shape=(num_samples, output_len, num_nodes, 1))
    train_flow = train_targets(args.dataset_name)
    acc = ZeroAwareMetricAccumulator(
        train_flow,
        config=MetricConfig(high_q=args.high_q, min_positive_support=args.min_positive_support),
    )

    for start in range(0, num_samples, args.batch_size):
        end = min(start + args.batch_size, num_samples)
        acc.update(pred[start:end], target[start:end])

    metrics = acc.summary()
    system = args.system_name or args.ckpt_dir.name
    report = {
        "dataset_name": args.dataset_name,
        "system": system,
        "ckpt_dir": str(args.ckpt_dir),
        "num_test_windows": num_samples,
        "metrics": metrics,
    }
    safe_system = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in system)
    json_path = args.output_dir / f"basicts_zero_metrics_{args.dataset_name}_{safe_system}.json"
    csv_path = args.output_dir / f"basicts_zero_metrics_{args.dataset_name}_{safe_system}.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["system", *sorted(metrics.keys())])
        writer.writeheader()
        writer.writerow({"system": system, **metrics})
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "metrics": metrics}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

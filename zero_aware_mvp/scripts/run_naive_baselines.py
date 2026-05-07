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

from zero_aware_mvp.lib.basicts_io import load_data, load_desc, load_split, load_unix_timestamps, prediction_starts
from zero_aware_mvp.lib.metrics import MetricConfig, ZeroAwareMetricAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-aware naive baselines on a BasicTS dataset.")
    parser.add_argument("--dataset-name", default="TRAFFIC_VOLUME_FULL_5MIN")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--high-q", type=float, default=0.90)
    parser.add_argument("--min-positive-support", type=int, default=24)
    return parser.parse_args()


def slot_index(unix_timestamps: np.ndarray | None, total_len: int, frequency_minutes: int) -> np.ndarray:
    if unix_timestamps is not None:
        return ((unix_timestamps % 86400) // (frequency_minutes * 60)).astype(np.int64)
    return (np.arange(total_len) % (1440 // frequency_minutes)).astype(np.int64)


def compute_slot_median(data: np.ndarray, train_idx: np.ndarray, slots: np.ndarray, num_slots: int) -> np.ndarray:
    num_nodes = data.shape[1]
    med = np.zeros((num_slots, num_nodes), dtype=np.float32)
    global_med = np.median(data[train_idx, :, 0], axis=0).astype(np.float32)
    for slot in range(num_slots):
        idx = train_idx[slots[train_idx] == slot]
        med[slot] = np.median(data[idx, :, 0], axis=0).astype(np.float32) if len(idx) else global_med
    return med


def build_prediction(
    name: str,
    data: np.ndarray,
    starts: np.ndarray,
    target_times: np.ndarray,
    output_len: int,
    node_mean: np.ndarray,
    node_median: np.ndarray,
    slot_median: np.ndarray,
    slots: np.ndarray,
    period: int,
) -> np.ndarray:
    if name == "all_zero":
        return np.zeros((len(starts), output_len, data.shape[1]), dtype=np.float32)
    if name == "previous_step":
        last = np.asarray(data[starts - 1, :, 0], dtype=np.float32)
        return np.repeat(last[:, None, :], repeats=output_len, axis=1)
    if name == "node_mean":
        return np.broadcast_to(node_mean.reshape(1, 1, -1), (len(starts), output_len, data.shape[1])).astype(np.float32)
    if name == "node_median":
        return np.broadcast_to(node_median.reshape(1, 1, -1), (len(starts), output_len, data.shape[1])).astype(np.float32)
    if name == "slot_median":
        return slot_median[slots[target_times]]
    if name == "seasonal_day_ago":
        source_times = target_times - period
        pred = np.broadcast_to(node_median.reshape(1, 1, -1), (len(starts), output_len, data.shape[1])).copy()
        valid = source_times >= 0
        if np.any(valid):
            pred[valid] = data[source_times[valid], :, 0]
        return pred.astype(np.float32)
    raise ValueError(f"Unknown baseline: {name}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    desc = load_desc(args.dataset_name)
    settings = desc["regular_settings"]
    input_len = int(settings["INPUT_LEN"])
    output_len = int(settings["OUTPUT_LEN"])
    frequency = int(desc["frequency (minutes)"])
    data = load_data(args.dataset_name)
    split = load_split(args.dataset_name)
    starts = prediction_starts(
        total_len=int(desc["num_time_steps"]),
        split_idx=split["test"],
        input_len=input_len,
        output_len=output_len,
        mode="test",
    )
    if len(starts) == 0:
        raise RuntimeError(f"No test windows for {args.dataset_name}")

    train_flow = np.asarray(data[split["train"], :, 0], dtype=np.float32).copy()
    node_mean = np.mean(train_flow, axis=0).astype(np.float32)
    node_median = np.median(train_flow, axis=0).astype(np.float32)
    timestamps = load_unix_timestamps(args.dataset_name)
    slots = slot_index(timestamps, int(desc["num_time_steps"]), frequency)
    num_slots = 1440 // frequency
    slot_median = compute_slot_median(data, split["train"], slots, num_slots)
    period = num_slots

    baseline_names = ["all_zero", "previous_step", "node_mean", "node_median", "slot_median", "seasonal_day_ago"]
    results = {}
    metric_config = MetricConfig(high_q=args.high_q, min_positive_support=args.min_positive_support)
    horizon_offsets = np.arange(output_len, dtype=np.int64)

    for baseline in baseline_names:
        acc = ZeroAwareMetricAccumulator(train_flow, config=metric_config)
        for start in range(0, len(starts), args.batch_size):
            batch_starts = starts[start : start + args.batch_size]
            target_times = batch_starts[:, None] + horizon_offsets[None, :]
            target = np.asarray(data[target_times, :, 0], dtype=np.float32)
            pred = build_prediction(
                baseline,
                data,
                batch_starts,
                target_times,
                output_len,
                node_mean,
                node_median,
                slot_median,
                slots,
                period,
            )
            acc.update(pred, target)
        results[baseline] = acc.summary()
        print(f"[zero-aware] {baseline}: MAE={results[baseline]['MAE']:.4f} "
              f"occ_f1={results[baseline]['occurrence_f1']:.4f} high_f1={results[baseline]['high_f1']:.4f}")

    report = {
        "dataset_name": args.dataset_name,
        "num_test_windows": int(len(starts)),
        "input_len": input_len,
        "output_len": output_len,
        "frequency_minutes": frequency,
        "baselines": results,
    }
    json_path = args.output_dir / f"naive_metrics_{args.dataset_name}.json"
    csv_path = args.output_dir / f"naive_metrics_{args.dataset_name}.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    metric_names = sorted(next(iter(results.values())).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["system", *metric_names])
        writer.writeheader()
        for system, metrics in results.items():
            writer.writerow({"system": system, **metrics})
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()

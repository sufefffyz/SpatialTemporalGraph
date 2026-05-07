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
    parser.add_argument(
        "--node-filter",
        choices=["all", "train_nonzero", "train_active24", "full_nonzero"],
        default="all",
        help=(
            "Evaluate on all roads, roads with at least one positive training sample, "
            "roads with at least min-positive-support positive training samples, or roads "
            "that are nonzero anywhere in the full series. full_nonzero is diagnostic only."
        ),
    )
    return parser.parse_args()


def slot_index(unix_timestamps: np.ndarray | None, total_len: int, frequency_minutes: int) -> np.ndarray:
    if unix_timestamps is not None:
        return ((unix_timestamps % 86400) // (frequency_minutes * 60)).astype(np.int64)
    return (np.arange(total_len) % (1440 // frequency_minutes)).astype(np.int64)


def compute_slot_median(train_flow: np.ndarray, train_idx: np.ndarray, slots: np.ndarray, num_slots: int) -> np.ndarray:
    num_nodes = train_flow.shape[1]
    med = np.zeros((num_slots, num_nodes), dtype=np.float32)
    global_med = np.median(train_flow, axis=0).astype(np.float32)
    train_slots = slots[train_idx]
    for slot in range(num_slots):
        idx = train_slots == slot
        med[slot] = np.median(train_flow[idx], axis=0).astype(np.float32) if np.any(idx) else global_med
    return med


def full_nonzero_mask(data: np.ndarray, eps: float, chunk_size: int = 256) -> np.ndarray:
    mask = np.zeros(data.shape[1], dtype=bool)
    for start in range(0, data.shape[0], chunk_size):
        chunk = np.asarray(data[start : start + chunk_size, :, 0], dtype=np.float32)
        mask |= np.any(chunk > eps, axis=0)
    return mask


def build_node_mask(
    node_filter: str,
    data: np.ndarray,
    train_flow: np.ndarray,
    eps: float,
    min_positive_support: int,
) -> tuple[np.ndarray, dict[str, float]]:
    support = np.sum(train_flow > eps, axis=0)
    if node_filter == "all":
        mask = np.ones(train_flow.shape[1], dtype=bool)
    elif node_filter == "train_nonzero":
        mask = support > 0
    elif node_filter == "train_active24":
        mask = support >= min_positive_support
    elif node_filter == "full_nonzero":
        mask = full_nonzero_mask(data, eps=eps)
    else:
        raise ValueError(f"Unknown node filter: {node_filter}")

    selected_support = support[mask]
    stats = {
        "total_nodes": int(train_flow.shape[1]),
        "selected_nodes": int(np.sum(mask)),
        "selected_node_ratio": float(np.mean(mask)),
        "selected_train_positive_support_min": float(np.min(selected_support)) if selected_support.size else float("nan"),
        "selected_train_positive_support_median": float(np.median(selected_support)) if selected_support.size else float("nan"),
        "selected_train_positive_support_mean": float(np.mean(selected_support)) if selected_support.size else float("nan"),
    }
    return mask, stats


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
    node_mask: np.ndarray,
) -> np.ndarray:
    num_nodes = int(np.sum(node_mask))
    if name == "all_zero":
        return np.zeros((len(starts), output_len, num_nodes), dtype=np.float32)
    if name == "previous_step":
        last = np.asarray(data[starts - 1, :, 0], dtype=np.float32)[:, node_mask]
        return np.repeat(last[:, None, :], repeats=output_len, axis=1)
    if name == "node_mean":
        return np.broadcast_to(node_mean.reshape(1, 1, -1), (len(starts), output_len, num_nodes)).astype(np.float32)
    if name == "node_median":
        return np.broadcast_to(node_median.reshape(1, 1, -1), (len(starts), output_len, num_nodes)).astype(np.float32)
    if name == "slot_median":
        return slot_median[slots[target_times]]
    if name == "seasonal_day_ago":
        source_times = target_times - period
        pred = np.broadcast_to(node_median.reshape(1, 1, -1), (len(starts), output_len, num_nodes)).copy()
        valid = source_times >= 0
        if np.any(valid):
            pred[valid] = np.asarray(data[source_times[valid], :, 0], dtype=np.float32)[:, node_mask]
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

    train_flow_full = np.asarray(data[split["train"], :, 0], dtype=np.float32).copy()
    metric_config = MetricConfig(high_q=args.high_q, min_positive_support=args.min_positive_support)
    node_mask, node_filter_stats = build_node_mask(
        args.node_filter,
        data=data,
        train_flow=train_flow_full,
        eps=metric_config.eps,
        min_positive_support=args.min_positive_support,
    )
    train_flow = train_flow_full if np.all(node_mask) else np.ascontiguousarray(train_flow_full[:, node_mask])
    if train_flow is not train_flow_full:
        del train_flow_full
    node_mean = np.mean(train_flow, axis=0).astype(np.float32)
    node_median = np.median(train_flow, axis=0).astype(np.float32)
    timestamps = load_unix_timestamps(args.dataset_name)
    slots = slot_index(timestamps, int(desc["num_time_steps"]), frequency)
    num_slots = 1440 // frequency
    slot_median = compute_slot_median(train_flow, split["train"], slots, num_slots)
    period = num_slots

    baseline_names = ["all_zero", "previous_step", "node_mean", "node_median", "slot_median", "seasonal_day_ago"]
    results = {}
    horizon_offsets = np.arange(output_len, dtype=np.int64)
    metric_template = ZeroAwareMetricAccumulator(train_flow, config=metric_config)

    for baseline in baseline_names:
        acc = metric_template.clone_empty()
        for start in range(0, len(starts), args.batch_size):
            batch_starts = starts[start : start + args.batch_size]
            target_times = batch_starts[:, None] + horizon_offsets[None, :]
            target = np.asarray(data[target_times, :, 0], dtype=np.float32)[:, :, node_mask]
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
                node_mask,
            )
            acc.update(pred, target)
        results[baseline] = acc.summary()
        print(f"[zero-aware] {args.node_filter} {baseline}: MAE={results[baseline]['MAE']:.4f} "
              f"occ_f1={results[baseline]['occurrence_f1']:.4f} high_f1={results[baseline]['high_f1']:.4f}")

    report = {
        "dataset_name": args.dataset_name,
        "node_filter": args.node_filter,
        "node_filter_stats": node_filter_stats,
        "num_test_windows": int(len(starts)),
        "input_len": input_len,
        "output_len": output_len,
        "frequency_minutes": frequency,
        "baselines": results,
    }
    suffix = "" if args.node_filter == "all" else f"__{args.node_filter}"
    json_path = args.output_dir / f"naive_metrics_{args.dataset_name}{suffix}.json"
    csv_path = args.output_dir / f"naive_metrics_{args.dataset_name}{suffix}.csv"
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

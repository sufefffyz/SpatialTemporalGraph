#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-aware diagnostics on the raw UTB volume NPZ.")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-nodes", type=int, default=0, help="Use a deterministic node sample for smoke tests.")
    parser.add_argument("--max-time-steps", type=int, default=0, help="Use a prefix for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-5)
    return parser.parse_args()


def basic_stats(values: np.ndarray, eps: float) -> dict[str, float]:
    if values.size == 0:
        return {
            "finite_ratio": float("nan"),
            "zero_rate": float("nan"),
            "mean": float("nan"),
            "variance": float("nan"),
            "variance_to_mean": float("nan"),
            "positive_rate": float("nan"),
            "positive_mean": float("nan"),
            "positive_q90": float("nan"),
            "positive_q95": float("nan"),
            "max": float("nan"),
        }
    finite = np.isfinite(values)
    finite_values = values[finite]
    positive = finite_values[finite_values > eps]
    mean = float(np.mean(finite_values)) if finite_values.size else float("nan")
    var = float(np.var(finite_values)) if finite_values.size else float("nan")
    return {
        "finite_ratio": float(np.mean(finite)),
        "zero_rate": float(np.mean(np.isclose(finite_values, 0.0, atol=eps))) if finite_values.size else float("nan"),
        "mean": mean,
        "variance": var,
        "variance_to_mean": float(var / mean) if mean > 0 else float("nan"),
        "positive_rate": float(np.mean(finite_values > eps)) if finite_values.size else float("nan"),
        "positive_mean": float(np.mean(positive)) if positive.size else float("nan"),
        "positive_q90": float(np.quantile(positive, 0.90)) if positive.size else float("nan"),
        "positive_q95": float(np.quantile(positive, 0.95)) if positive.size else float("nan"),
        "max": float(np.max(finite_values)) if finite_values.size else float("nan"),
    }


def coerce_indices(values: np.ndarray, total_len: int, unix_timestamps: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if np.issubdtype(arr.dtype, np.integer) and arr.size and int(arr.min()) >= 0 and int(arr.max()) < total_len:
        return arr.astype(np.int64)
    ts_to_idx = {int(ts): i for i, ts in enumerate(unix_timestamps.tolist())}
    return np.asarray([ts_to_idx[int(v)] for v in arr.tolist() if int(v) in ts_to_idx], dtype=np.int64)


def group_zero_rate(values: np.ndarray, groups: np.ndarray, eps: float, max_groups: int = 50) -> list[dict]:
    rows = []
    for group in np.unique(groups)[:max_groups]:
        mask = groups == group
        if not np.any(mask):
            continue
        sub = values[:, mask]
        rows.append(
            {
                "group": str(group),
                "nodes": int(np.sum(mask)),
                "zero_rate": float(np.mean(np.isclose(sub, 0.0, atol=eps))),
                "mean": float(np.mean(sub)),
            }
        )
    return rows


def aggregation_stats(values: np.ndarray, native_minutes: int, eps: float) -> dict[str, dict]:
    out = {}
    for minutes in [native_minutes, 15, 30, 60]:
        if minutes < native_minutes or minutes % native_minutes:
            continue
        factor = minutes // native_minutes
        usable = (values.shape[0] // factor) * factor
        reshaped = values[:usable].reshape(usable // factor, factor, values.shape[1])
        aggregated = np.sum(reshaped, axis=1)
        out[f"{minutes}min"] = basic_stats(aggregated, eps)
    return out


def poisson_nb_expected_zero(
    values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hours: np.ndarray,
    eps: float,
) -> dict[str, float]:
    train_idx = train_idx[(train_idx >= 0) & (train_idx < values.shape[0])]
    test_idx = test_idx[(test_idx >= 0) & (test_idx < values.shape[0])]
    train = values[train_idx]
    test = values[test_idx]
    if train.size == 0 or test.size == 0:
        return {}

    road_mean = np.mean(train, axis=0)
    road_var = np.var(train, axis=0)
    mu_by_hour = np.tile(road_mean.reshape(1, -1), (24, 1))
    for hour in range(24):
        idx = train_idx[hours[train_idx] == hour]
        if len(idx):
            mu_by_hour[hour] = np.mean(values[idx], axis=0)
    poisson_expected = []
    batch = 256
    for start in range(0, len(test_idx), batch):
        idx = test_idx[start : start + batch]
        poisson_expected.append(np.exp(-mu_by_hour[hours[idx]]))
    poisson_expected_zero = float(np.mean(np.concatenate(poisson_expected, axis=0)))

    nb_p0 = np.exp(-road_mean)
    over = road_var > road_mean + eps
    r = np.zeros_like(road_mean, dtype=np.float64)
    r[over] = (road_mean[over] ** 2) / np.maximum(road_var[over] - road_mean[over], eps)
    nb_p0[over] = (r[over] / (r[over] + road_mean[over] + eps)) ** r[over]
    nb_expected_zero = float(np.mean(nb_p0))

    return {
        "test_observed_zero_rate": float(np.mean(np.isclose(test, 0.0, atol=eps))),
        "poisson_road_hour_expected_zero_rate": poisson_expected_zero,
        "poisson_excess_zero_gap": float(np.mean(np.isclose(test, 0.0, atol=eps)) - poisson_expected_zero),
        "nb_road_expected_zero_rate": nb_expected_zero,
        "nb_excess_zero_gap": float(np.mean(np.isclose(test, 0.0, atol=eps)) - nb_expected_zero),
    }


def transition_stats(values: np.ndarray, eps: float) -> dict[str, float]:
    z0 = np.isclose(values[:-1], 0.0, atol=eps)
    z1 = np.isclose(values[1:], 0.0, atol=eps)
    total = float(z0.size)
    return {
        "p_0_to_0": float(np.sum(z0 & z1) / np.sum(z0)) if np.sum(z0) else float("nan"),
        "p_0_to_pos": float(np.sum(z0 & ~z1) / np.sum(z0)) if np.sum(z0) else float("nan"),
        "p_pos_to_0": float(np.sum(~z0 & z1) / np.sum(~z0)) if np.sum(~z0) else float("nan"),
        "p_pos_to_pos": float(np.sum(~z0 & ~z1) / np.sum(~z0)) if np.sum(~z0) else float("nan"),
        "transition_observations": total,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input_npz, allow_pickle=True) as data:
        targets = np.asarray(data["targets"], dtype=np.float32)
        unix_timestamps = np.asarray(data["unix_timestamps"], dtype=np.int64)
        spatial_features = np.asarray(data["spatial_node_features"], dtype=np.float32)[0]
        spatial_names = [str(x) for x in data["spatial_node_feature_names"].tolist()]
        train_idx = coerce_indices(data["train_timestamps"], targets.shape[0], unix_timestamps)
        val_idx = coerce_indices(data["val_timestamps"], targets.shape[0], unix_timestamps)
        test_idx = coerce_indices(data["test_timestamps"], targets.shape[0], unix_timestamps)

    if args.max_time_steps > 0:
        targets = targets[: args.max_time_steps]
        unix_timestamps = unix_timestamps[: args.max_time_steps]
    if args.sample_nodes > 0 and args.sample_nodes < targets.shape[1]:
        rng = np.random.default_rng(args.seed)
        node_idx = np.sort(rng.choice(targets.shape[1], size=args.sample_nodes, replace=False))
        targets = targets[:, node_idx]
        spatial_features = spatial_features[node_idx]

    diffs = np.diff(unix_timestamps)
    native_minutes = int(round(float(np.median(diffs)) / 60.0)) if diffs.size else 5
    hours = ((unix_timestamps % 86400) // 3600).astype(np.int64)
    report = {
        "input_npz": str(args.input_npz),
        "shape": list(targets.shape),
        "native_minutes": native_minutes,
        "sample_nodes": int(args.sample_nodes),
        "max_time_steps": int(args.max_time_steps),
        "overall": basic_stats(targets, args.eps),
        "splits": {
            "train": basic_stats(targets[train_idx[train_idx < targets.shape[0]]], args.eps),
            "valid": basic_stats(targets[val_idx[val_idx < targets.shape[0]]], args.eps),
            "test": basic_stats(targets[test_idx[test_idx < targets.shape[0]]], args.eps),
        },
        "by_hour": [],
        "by_road_feature": {},
        "aggregation": aggregation_stats(targets, native_minutes, args.eps),
        "excess_zero": poisson_nb_expected_zero(targets, train_idx, test_idx, hours, args.eps),
        "transitions": transition_stats(targets, args.eps),
    }
    for hour in range(24):
        mask = hours == hour
        if np.any(mask):
            report["by_hour"].append({"hour_utc": hour, **basic_stats(targets[mask], args.eps)})

    for feature_name in ["category", "speed_limit", "region_id", "edge_type"]:
        if feature_name in spatial_names:
            idx = spatial_names.index(feature_name)
            report["by_road_feature"][feature_name] = group_zero_rate(
                targets, spatial_features[:, idx], args.eps
            )

    suffix = "sample" if args.sample_nodes or args.max_time_steps else "full"
    out_path = args.output_dir / f"zero_diagnostics_{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "overall": report["overall"], "excess_zero": report["excess_zero"]}, indent=2))


if __name__ == "__main__":
    main()

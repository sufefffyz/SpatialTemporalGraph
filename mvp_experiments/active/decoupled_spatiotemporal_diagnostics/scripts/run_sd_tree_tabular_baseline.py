#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


REPO_ROOT = Path(__file__).resolve().parents[4]
BASICTS_ROOT = REPO_ROOT / "BasicTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight tabular GBDT baselines on SD.")
    parser.add_argument("--dataset-dir", type=Path, default=BASICTS_ROOT / "datasets" / "SD")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adj-path", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=["lagtime", "tabst"], choices=["lagtime", "tabst"])
    parser.add_argument("--backend", choices=["sklearn-hist", "lightgbm"], default="sklearn-hist")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--train-rows-per-horizon", type=int, default=200_000)
    parser.add_argument("--feature-chunk-rows", type=int, default=250_000)
    parser.add_argument("--predict-chunk-rows", type=int, default=350_000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.03)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=-1)
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def weekday_codes(data: np.ndarray, indices: np.ndarray, slots_per_day: int) -> np.ndarray:
    if data.shape[-1] >= 3:
        codes = np.rint(np.asarray(data[indices, 0, 2]) * 7).astype(np.int16)
        return np.clip(codes, 0, 6)
    return ((indices // slots_per_day) % 7).astype(np.int16)


def build_seasonal_reference(
    data: np.ndarray,
    reference_end: int,
    slots_per_day: int,
    target_channel: int,
) -> dict[str, np.ndarray]:
    num_nodes = data.shape[1]
    refs = {
        "q10": np.full((slots_per_day, 7, num_nodes), np.nan, dtype=np.float32),
        "median": np.full((slots_per_day, 7, num_nodes), np.nan, dtype=np.float32),
        "q90": np.full((slots_per_day, 7, num_nodes), np.nan, dtype=np.float32),
    }
    indices = np.arange(reference_end)
    dows = weekday_codes(data, indices, slots_per_day)
    for slot in range(slots_per_day):
        slot_mask = (indices % slots_per_day) == slot
        for dow in range(7):
            selected = indices[slot_mask & (dows == dow)]
            if selected.size == 0:
                continue
            values = np.asarray(data[selected, :, target_channel], dtype=np.float32)
            values = np.where(np.isfinite(values) & (values != 0), values, np.nan)
            with np.errstate(invalid="ignore"):
                refs["q10"][slot, dow] = np.nanquantile(values, 0.10, axis=0)
                refs["median"][slot, dow] = np.nanmedian(values, axis=0)
                refs["q90"][slot, dow] = np.nanquantile(values, 0.90, axis=0)
    return refs


def load_adjacency(path: Path | None, num_nodes: int) -> list[np.ndarray]:
    if path is None or not path.exists():
        return [np.empty(0, dtype=np.int64) for _ in range(num_nodes)]
    with path.open("rb") as fp:
        obj = pickle.load(fp)
    if isinstance(obj, dict):
        for key in ("adj_mx", "adj", "matrix"):
            if key in obj:
                adj = np.asarray(obj[key], dtype=np.float32)
                break
        else:
            adj = np.asarray(obj, dtype=np.float32)
    elif isinstance(obj, (tuple, list)) and len(obj) >= 3:
        adj = np.asarray(obj[-1], dtype=np.float32)
    else:
        adj = np.asarray(obj, dtype=np.float32)
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Adjacency shape {adj.shape} does not match num_nodes={num_nodes}")
    base = (adj > 0) | (adj.T > 0)
    np.fill_diagonal(base, False)
    return [np.flatnonzero(base[node]).astype(np.int64) for node in range(num_nodes)]


def periodic_features(indices: np.ndarray, slots_per_day: int, dows: np.ndarray) -> np.ndarray:
    slots = indices % slots_per_day
    angle = 2.0 * np.pi * slots.astype(np.float32) / float(slots_per_day)
    dow_angle = 2.0 * np.pi * dows.astype(np.float32) / 7.0
    return np.column_stack(
        [
            np.sin(angle),
            np.cos(angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
            slots.astype(np.float32) / float(slots_per_day - 1),
            dows.astype(np.float32) / 6.0,
        ]
    ).astype(np.float32)


def neighbor_stats(
    data: np.ndarray,
    starts: np.ndarray,
    nodes: np.ndarray,
    input_len: int,
    neighbors_by_node: list[np.ndarray],
    seasonal_refs: dict[str, np.ndarray],
    target_indices: np.ndarray,
    target_dows: np.ndarray,
    slots_per_day: int,
    target_channel: int,
) -> np.ndarray:
    def safe_nanmean(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        counts = finite.sum(axis=1)
        sums = np.where(finite, values, 0.0).sum(axis=1)
        return np.divide(sums, counts, out=np.zeros(values.shape[0], dtype=np.float32), where=counts > 0)

    def safe_nanmax(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        max_values = np.where(finite, values, -np.inf).max(axis=1)
        return np.where(finite.any(axis=1), max_values, 0.0).astype(np.float32)

    out = np.zeros((starts.shape[0], 8), dtype=np.float32)
    out[:, -1] = 0.0
    for node in np.unique(nodes):
        pos = np.flatnonzero(nodes == node)
        neigh = neighbors_by_node[int(node)]
        if neigh.size == 0:
            continue
        first_idx = starts[pos]
        last_idx = starts[pos] + input_len - 1
        last_vals = np.asarray(data[last_idx[:, None], neigh[None, :], target_channel], dtype=np.float32)
        first_vals = np.asarray(data[first_idx[:, None], neigh[None, :], target_channel], dtype=np.float32)
        trends = last_vals - first_vals
        target_slots = (target_indices[pos] % slots_per_day).astype(np.int16)
        med = seasonal_refs["median"][target_slots, target_dows[pos], :][:, neigh]
        out[pos, 0] = safe_nanmean(last_vals)
        out[pos, 1] = safe_nanmax(last_vals)
        out[pos, 2] = safe_nanmean(trends)
        out[pos, 3] = safe_nanmax(trends)
        out[pos, 4] = safe_nanmean(med)
        out[pos, 5] = safe_nanmax(med)
        out[pos, 6] = float(neigh.size)
        out[pos, 7] = 1.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(
    data: np.ndarray,
    starts: np.ndarray,
    nodes: np.ndarray,
    horizon: int,
    input_len: int,
    slots_per_day: int,
    num_nodes: int,
    target_channel: int,
    variant: str,
    seasonal_refs: dict[str, np.ndarray],
    neighbors_by_node: list[np.ndarray],
) -> np.ndarray:
    starts = starts.astype(np.int64, copy=False)
    nodes = nodes.astype(np.int64, copy=False)
    h_idx = horizon - 1
    history_idx = starts[:, None] + np.arange(input_len, dtype=np.int64)[None, :]
    history = np.asarray(data[history_idx, nodes[:, None], target_channel], dtype=np.float32)
    first = history[:, 0]
    last = history[:, -1]
    mean = history.mean(axis=1)
    std = history.std(axis=1)
    min_v = history.min(axis=1)
    max_v = history.max(axis=1)
    zero_count = (history == 0).sum(axis=1).astype(np.float32)
    zero_share = zero_count / float(input_len)
    current_idx = starts + input_len - 1
    target_idx = starts + input_len + h_idx
    current_dow = weekday_codes(data, current_idx, slots_per_day)
    target_dow = weekday_codes(data, target_idx, slots_per_day)
    base = [
        history,
        np.column_stack(
            [
                first,
                last,
                mean,
                std,
                min_v,
                max_v,
                last - first,
                last - mean,
                zero_count,
                zero_share,
                nodes.astype(np.float32) / float(max(1, num_nodes - 1)),
                np.full(starts.shape[0], horizon / 12.0, dtype=np.float32),
            ]
        ).astype(np.float32),
        periodic_features(current_idx, slots_per_day, current_dow),
        periodic_features(target_idx, slots_per_day, target_dow),
    ]
    if variant == "tabst":
        target_slots = (target_idx % slots_per_day).astype(np.int16)
        current_slots = (current_idx % slots_per_day).astype(np.int16)
        fut_q10 = seasonal_refs["q10"][target_slots, target_dow, nodes]
        fut_med = seasonal_refs["median"][target_slots, target_dow, nodes]
        fut_q90 = seasonal_refs["q90"][target_slots, target_dow, nodes]
        cur_med = seasonal_refs["median"][current_slots, current_dow, nodes]
        seasonal = np.column_stack(
            [
                fut_q10,
                fut_med,
                fut_q90,
                cur_med,
                fut_med - last,
                fut_med - cur_med,
                (fut_q90 - fut_q10),
            ]
        ).astype(np.float32)
        neigh = neighbor_stats(
            data,
            starts,
            nodes,
            input_len,
            neighbors_by_node,
            seasonal_refs,
            target_idx,
            target_dow,
            slots_per_day,
            target_channel,
        )
        base.extend([seasonal, neigh])
    return np.nan_to_num(np.concatenate(base, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sample_training_rows(
    rng: np.random.Generator,
    data: np.ndarray,
    train_start_count: int,
    num_nodes: int,
    input_len: int,
    horizon: int,
    target_channel: int,
    target_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_start_count * num_nodes
    starts_all = []
    nodes_all = []
    y_all = []
    need = target_rows
    while need > 0:
        draw = min(total, max(int(need * 1.6), 20_000))
        flat = rng.choice(total, size=draw, replace=False if draw < total else True)
        starts = (flat // num_nodes).astype(np.int64)
        nodes = (flat % num_nodes).astype(np.int64)
        target_idx = starts + input_len + (horizon - 1)
        y = np.asarray(data[target_idx, nodes, target_channel], dtype=np.float32)
        valid = np.isfinite(y) & (y != 0)
        if valid.any():
            starts_all.append(starts[valid])
            nodes_all.append(nodes[valid])
            y_all.append(y[valid])
            need = target_rows - sum(part.shape[0] for part in y_all)
        if draw >= total and need > 0:
            break
    starts = np.concatenate(starts_all)[:target_rows]
    nodes = np.concatenate(nodes_all)[:target_rows]
    y = np.concatenate(y_all)[:target_rows]
    return starts, nodes, y


def make_model(args: argparse.Namespace, seed: int):
    if args.backend == "sklearn-hist":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=0,
        )
    if args.backend == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install LightGBM first, for example: pip install lightgbm==4.6.0") from exc
        return lgb.LGBMRegressor(
            boosting_type="gbdt",
            objective="regression",
            metric="l2",
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=args.max_leaf_nodes,
            reg_lambda=args.l2_regularization,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            min_child_samples=args.min_child_samples,
            n_jobs=args.num_threads,
            random_state=seed,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def anomaly_masks_for_horizon(
    data: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    input_len: int,
    slots_per_day: int,
    seasonal_refs_eval: dict[str, np.ndarray],
    target_channel: int,
) -> dict[str, np.ndarray]:
    h_idx = horizon - 1
    valid = np.isfinite(y) & (y != 0)
    target_idx = starts + input_len + h_idx
    slots = (target_idx % slots_per_day).astype(np.int16)
    dows = weekday_codes(data, target_idx, slots_per_day)
    q10 = seasonal_refs_eval["q10"][slots, dows, :]
    median = seasonal_refs_eval["median"][slots, dows, :]
    contextual_low_q10 = valid & np.isfinite(q10) & np.isfinite(median) & (median >= 100.0) & (y <= q10)
    contextual_low_extreme = contextual_low_q10 & (y <= 0.5 * median)
    prev_max = np.full_like(y, np.nan, dtype=np.float32)
    for lag in range(1, 7):
        prev = np.asarray(data[target_idx - lag, :, target_channel], dtype=np.float32)
        prev_max = np.fmax(prev_max, prev)
    recent_drop = valid & np.isfinite(prev_max) & (prev_max >= 100.0) & (y <= 0.5 * prev_max) & ((prev_max - y) >= 100.0)
    input_zero_count = np.zeros_like(y, dtype=np.int16)
    sample_zero_share = np.zeros(starts.shape[0], dtype=np.float32)
    for row, start in enumerate(starts.tolist()):
        hist = np.asarray(data[start : start + input_len, :, target_channel], dtype=np.float32)
        zeros = hist == 0
        input_zero_count[row] = zeros.sum(axis=0)
        sample_zero_share[row] = float(zeros.mean())
    input_zero_target_half = valid & (input_zero_count >= 6)
    input_zero_sample_high = valid & (sample_zero_share.reshape(-1, 1) >= 0.05)
    input_quality_union = input_zero_target_half | input_zero_sample_high
    strict_union = contextual_low_extreme | recent_drop
    broad_union = contextual_low_q10 | recent_drop
    return {
        "all": valid,
        "strict_union": strict_union,
        "input_quality_union": input_quality_union,
        "strict_plus_input_union": strict_union | input_quality_union,
        "broad_union": broad_union,
    }


def periodic_neighbor_cue_mask(
    data: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    input_len: int,
    slots_per_day: int,
    seasonal_refs_eval: dict[str, np.ndarray],
    neighbors_by_node: list[np.ndarray],
    target_channel: int,
) -> np.ndarray:
    valid = np.isfinite(y) & (y != 0)
    target_idx = starts + input_len + (horizon - 1)
    slots = (target_idx % slots_per_day).astype(np.int16)
    dows = weekday_codes(data, target_idx, slots_per_day)
    med = seasonal_refs_eval["median"][slots, dows, :]
    last = np.asarray(data[starts + input_len - 1, :, target_channel], dtype=np.float32)
    seasonal_tol = np.maximum(50.0, 0.15 * np.abs(med))
    periodic = valid & np.isfinite(med) & (med >= 300.0) & ((med - last) >= 200.0) & (np.abs(y - med) <= seasonal_tol)
    neighbor_hint = np.zeros_like(valid, dtype=bool)
    first_idx = starts
    last_idx = starts + input_len - 1
    for node, neigh in enumerate(neighbors_by_node):
        if neigh.size == 0:
            continue
        first_vals = np.asarray(data[first_idx[:, None], neigh[None, :], target_channel], dtype=np.float32)
        last_vals = np.asarray(data[last_idx[:, None], neigh[None, :], target_channel], dtype=np.float32)
        hint = ((last_vals - first_vals) >= 100.0) & (last_vals >= 100.0)
        neighbor_hint[:, node] = hint.any(axis=1)
    return periodic & neighbor_hint


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    desc = load_json(args.dataset_dir / "desc.json")
    regular = desc["regular_settings"]
    data_shape = tuple(int(x) for x in desc["shape"])
    data = np.memmap(args.dataset_dir / "data.dat", dtype=args.dtype, mode="r", shape=data_shape)
    num_steps, num_nodes, _ = data_shape
    input_len = int(regular["INPUT_LEN"])
    output_len = int(regular["OUTPUT_LEN"])
    frequency = int(desc.get("frequency (minutes)", 1))
    slots_per_day = max(1, int(round(24 * 60 / frequency)))
    train_ratio, val_ratio, _ = [float(v) for v in regular["TRAIN_VAL_TEST_RATIO"]]
    train_end = int(round(num_steps * train_ratio))
    test_start = int(round(num_steps * (train_ratio + val_ratio)))
    test_samples = num_steps - test_start - input_len - output_len + 1
    train_start_count = train_end - input_len - output_len + 1
    horizons = sorted({h for h in args.horizons if 1 <= h <= output_len})
    adj_path = args.adj_path or (args.dataset_dir / "adj_mx.pkl")
    neighbors_by_node = load_adjacency(adj_path, num_nodes)
    print(
        json.dumps(
            {
                "dataset": desc["name"],
                "train_start_count": train_start_count,
                "test_samples": test_samples,
                "num_nodes": num_nodes,
                "backend": args.backend,
                "variants": args.variants,
                "horizons": horizons,
            },
            indent=2,
        ),
        flush=True,
    )
    seasonal_refs_model = build_seasonal_reference(data, train_end, slots_per_day, args.target_channel)
    seasonal_refs_eval = build_seasonal_reference(data, test_start, slots_per_day, args.target_channel)
    rng = np.random.default_rng(args.seed)
    test_starts = test_start + np.arange(test_samples, dtype=np.int64)
    flat_total = test_samples * num_nodes

    average_rows: list[dict] = []
    horizon_rows: list[dict] = []
    subset_rows: list[dict] = []

    for variant in args.variants:
        variant_dir = output_dir / variant / "test_results"
        variant_dir.mkdir(parents=True, exist_ok=True)
        pred_mem = np.memmap(
            variant_dir / "predictions.npy",
            dtype=np.float32,
            mode="w+",
            shape=(test_samples, output_len, num_nodes),
        )
        target_mem = np.memmap(
            variant_dir / "targets.npy",
            dtype=np.float32,
            mode="w+",
            shape=(test_samples, output_len, num_nodes),
        )
        all_abs = 0.0
        all_sq = 0.0
        all_count = 0
        started = time.time()
        for horizon in horizons:
            h_started = time.time()
            train_starts, train_nodes, train_y = sample_training_rows(
                rng,
                data,
                train_start_count,
                num_nodes,
                input_len,
                horizon,
                args.target_channel,
                args.train_rows_per_horizon,
            )
            train_x = build_features(
                data,
                train_starts,
                train_nodes,
                horizon,
                input_len,
                slots_per_day,
                num_nodes,
                args.target_channel,
                variant,
                seasonal_refs_model,
                neighbors_by_node,
            )
            model = make_model(args, args.seed + 17 * horizon + (0 if variant == "lagtime" else 1000))
            model.fit(train_x, train_y)
            del train_x

            pred_h = np.empty((test_samples, num_nodes), dtype=np.float32)
            target_idx = test_starts + input_len + (horizon - 1)
            target_h = np.asarray(data[target_idx[:, None], np.arange(num_nodes)[None, :], args.target_channel], dtype=np.float32)
            target_mem[:, horizon - 1, :] = target_h
            for start_flat in range(0, flat_total, args.predict_chunk_rows):
                end_flat = min(flat_total, start_flat + args.predict_chunk_rows)
                flat = np.arange(start_flat, end_flat, dtype=np.int64)
                sample_idx = flat // num_nodes
                nodes = flat % num_nodes
                starts = test_starts[sample_idx]
                x = build_features(
                    data,
                    starts,
                    nodes,
                    horizon,
                    input_len,
                    slots_per_day,
                    num_nodes,
                    args.target_channel,
                    variant,
                    seasonal_refs_model,
                    neighbors_by_node,
                )
                pred = model.predict(x).astype(np.float32)
                pred_h.reshape(-1)[start_flat:end_flat] = pred
            pred_mem[:, horizon - 1, :] = pred_h

            valid = np.isfinite(target_h) & (target_h != 0)
            err = pred_h - target_h
            abs_sum = float(np.abs(err[valid]).sum(dtype=np.float64))
            sq_sum = float((err[valid] ** 2).sum(dtype=np.float64))
            count = int(valid.sum())
            all_abs += abs_sum
            all_sq += sq_sum
            all_count += count
            horizon_rows.append(
                {
                    "variant": variant,
                    "horizon": horizon,
                    "mae": abs_sum / count,
                    "rmse": float(np.sqrt(sq_sum / count)),
                    "valid_points": count,
                    "fit_predict_seconds": time.time() - h_started,
                    "train_rows": int(train_y.shape[0]),
                    "model_iterations": int(getattr(model, "n_iter_", args.max_iter)),
                }
            )
            masks = anomaly_masks_for_horizon(
                data,
                target_h,
                test_starts,
                horizon,
                input_len,
                slots_per_day,
                seasonal_refs_eval,
                args.target_channel,
            )
            masks["periodic_neighbor_cue"] = periodic_neighbor_cue_mask(
                data,
                target_h,
                test_starts,
                horizon,
                input_len,
                slots_per_day,
                seasonal_refs_eval,
                neighbors_by_node,
                args.target_channel,
            )
            for subset, mask in masks.items():
                subset_count = int(mask.sum())
                if subset_count == 0:
                    continue
                subset_err = err[mask]
                subset_rows.append(
                    {
                        "variant": variant,
                        "horizon": horizon,
                        "subset": subset,
                        "mae": float(np.abs(subset_err).mean()),
                        "rmse": float(np.sqrt(np.mean(subset_err ** 2))),
                        "points": subset_count,
                        "point_share": subset_count / count,
                    }
                )
            print(
                f"{variant} H{horizon}: MAE={abs_sum / count:.3f}, RMSE={np.sqrt(sq_sum / count):.3f}, "
                f"iters={getattr(model, 'n_iter_', args.max_iter)}, sec={time.time() - h_started:.1f}",
                flush=True,
            )
        pred_mem.flush()
        target_mem.flush()
        average_rows.append(
            {
                "variant": variant,
                "mae": all_abs / all_count,
                "rmse": float(np.sqrt(all_sq / all_count)),
                "valid_points": all_count,
                "seconds": time.time() - started,
                "result_dir": str(variant_dir),
            }
        )
    write_csv(output_dir / "average_metrics.csv", average_rows)
    write_csv(output_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(output_dir / "subset_metrics.csv", subset_rows)
    manifest = {
        "dataset": desc["name"],
        "frequency_minutes": frequency,
        "input_len": input_len,
        "output_len": output_len,
        "train_end": train_end,
        "test_start": test_start,
        "test_samples": test_samples,
        "num_nodes": num_nodes,
        "model": args.backend,
        "params": {
            "train_rows_per_horizon": args.train_rows_per_horizon,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "min_child_samples": args.min_child_samples,
            "num_threads": args.num_threads,
            "seed": args.seed,
        },
        "variants": args.variants,
        "outputs": ["average_metrics.csv", "horizon_metrics.csv", "subset_metrics.csv"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"average": average_rows, "output_dir": str(output_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

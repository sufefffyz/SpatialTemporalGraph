#!/usr/bin/env python3
"""Prepare Xuancheng into a DTIGNN-style road-state forecasting dataset.

This converter consumes NPZ files produced by ``run_cityflow_dtignn_generation.py``.
Unlike the earlier movement-count baseline, the default target here is
``active_volume_lsr``: mean active vehicles on each road, split by the next
left/straight/right movement intent.  This is the closest Xuancheng analogue to
DTIGNN's road-segment traffic-volume state.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_FEATURE_KEY = "active_volume_lsr"
TURN_NAMES = ("turn_left", "go_straight", "turn_right")


@dataclass(frozen=True)
class LoadedDay:
    path: Path
    date: str
    feature_lsr: np.ndarray
    bucket_start_s: np.ndarray
    bucket_end_s: np.ndarray
    phase_id_end: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        action="append",
        default=[],
        help="Input NPZ file. May be provided multiple times.",
    )
    parser.add_argument("--input-dir", default=None, help="Directory to scan for input NPZ files.")
    parser.add_argument(
        "--pattern",
        default="xuancheng_*_dtignn_turn_*s_start*_dur*.npz",
        help="Glob pattern used with --input-dir.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for prepared dataset files.")
    parser.add_argument("--dataset-name", default="xuancheng_dtignn_active_lsr_10s")
    parser.add_argument(
        "--feature-key",
        default=DEFAULT_FEATURE_KEY,
        help="Road-state key to use as traffic-volume channels. Default: active_volume_lsr.",
    )
    parser.add_argument("--input-window", type=int, default=30, help="Historical steps per sample.")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in steps.")
    parser.add_argument("--split-ratios", default="0.6,0.2,0.2")
    parser.add_argument(
        "--split-mode",
        choices=("chronological", "official_shuffle"),
        default="chronological",
        help=(
            "chronological is forecasting-safe. official_shuffle mimics the public DTIGNN "
            "prepareData.py sample shuffle before 6:2:2 split."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument(
        "--phase-start-id",
        type=int,
        default=1,
        help="First active CityFlow phase id to encode. Default 1 drops all-red phase 0.",
    )
    parser.add_argument(
        "--phase-count",
        type=int,
        default=None,
        help="Number of phase one-hot channels. Default inferred as max phase id - phase-start-id + 1.",
    )
    parser.add_argument(
        "--missing-ratio",
        default="dense",
        help="dense or a ratio stored in observed_road_masks, e.g. 0.3.",
    )
    parser.add_argument(
        "--apply-official-mask-op",
        action="store_true",
        help="For sparse masks, impute hidden roads from observed in-neighbors as DTIGNN prepareData.py does.",
    )
    parser.add_argument("--save-npz-split", action="store_true", help="Also save data_split.npz.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests after sample construction.",
    )
    return parser.parse_args()


def parse_split_ratios(text: str) -> tuple[float, float, float]:
    parts = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError(f"expected 3 split ratios, got {parts}")
    if not np.isclose(sum(parts), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {sum(parts)}")
    return parts  # type: ignore[return-value]


def scalar_string(archive: np.lib.npyio.NpzFile, key: str, fallback: str) -> str:
    if key not in archive:
        return fallback
    value = archive[key]
    if value.shape == ():
        return str(value.tolist())
    return str(value)


def collect_input_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p).expanduser().resolve() for p in args.input_npz]
    if args.input_dir:
        paths.extend(sorted(Path(args.input_dir).expanduser().resolve().glob(args.pattern)))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise SystemExit("no input NPZ files found; provide --input-npz or --input-dir")
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise SystemExit(f"input NPZ files do not exist: {missing}")
    return unique


def choose_road_mask(archive: np.lib.npyio.NpzFile, label: str, n_roads: int) -> tuple[np.ndarray, str]:
    if label == "dense":
        return np.ones(n_roads, dtype=bool), "dense"
    ratio = float(label)
    if "missing_ratios" not in archive or "observed_road_masks" not in archive:
        raise ValueError("input NPZ has no observed_road_masks; use --missing-ratio dense")
    stored = np.asarray(archive["missing_ratios"], dtype=np.float64)
    matches = np.where(np.isclose(stored, ratio, rtol=1e-6, atol=1e-6))[0]
    if len(matches) == 0:
        raise ValueError(f"missing ratio {ratio} not in stored ratios {stored.tolist()}")
    mask = np.asarray(archive["observed_road_masks"][int(matches[0])], dtype=bool)
    if mask.shape != (n_roads,):
        raise ValueError(f"observed road mask shape {mask.shape} does not match {(n_roads,)}")
    return mask, f"{ratio:.6g}"


def infer_phase_count(
    archives: Iterable[np.lib.npyio.NpzFile],
    phase_start_id: int,
    requested_count: int | None,
) -> int:
    if requested_count is not None:
        if requested_count <= 0:
            raise ValueError("--phase-count must be positive")
        return requested_count
    max_phase = phase_start_id
    for archive in archives:
        if "phase_id_end" in archive:
            max_phase = max(max_phase, int(np.max(archive["phase_id_end"])))
        if "phase_edge_phase_id" in archive and len(archive["phase_edge_phase_id"]):
            max_phase = max(max_phase, int(np.max(archive["phase_edge_phase_id"])))
    return max(1, max_phase - phase_start_id + 1)


def build_road_signal_index(first: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    n_roads = len(first["road_ids"])
    road_signal_idx = np.full(n_roads, -1, dtype=np.int16)
    conflicts: list[tuple[int, int, int]] = []
    starts = np.asarray(first["edge_start_road_idx"], dtype=np.int64)
    signals = np.asarray(first["edge_signal_idx"], dtype=np.int64)
    for road_idx, signal_idx in zip(starts, signals):
        if signal_idx < 0:
            continue
        existing = int(road_signal_idx[road_idx])
        if existing >= 0 and existing != int(signal_idx):
            conflicts.append((int(road_idx), existing, int(signal_idx)))
            continue
        road_signal_idx[road_idx] = int(signal_idx)
    return road_signal_idx, conflicts


def encode_road_phase_onehot(
    phase_id_end: np.ndarray,
    road_signal_idx: np.ndarray,
    phase_start_id: int,
    phase_count: int,
) -> tuple[np.ndarray, int]:
    n_steps = phase_id_end.shape[0]
    n_roads = road_signal_idx.shape[0]
    onehot = np.zeros((n_steps, n_roads, phase_count), dtype=np.float32)
    encoded = 0
    signal_roads = np.where(road_signal_idx >= 0)[0]
    for road_idx in signal_roads:
        signal_idx = int(road_signal_idx[road_idx])
        ids = phase_id_end[:, signal_idx].astype(np.int64) - phase_start_id
        valid = (ids >= 0) & (ids < phase_count)
        rows = np.where(valid)[0]
        onehot[rows, road_idx, ids[valid]] = 1.0
        encoded += int(valid.sum())
    return onehot, encoded


def load_days(
    paths: list[Path],
    feature_key: str,
    road_ids_ref: np.ndarray | None,
    signal_ids_ref: np.ndarray | None,
) -> tuple[list[LoadedDay], list[np.lib.npyio.NpzFile]]:
    days: list[LoadedDay] = []
    archives: list[np.lib.npyio.NpzFile] = []
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        archives.append(archive)
        if feature_key not in archive:
            raise ValueError(f"{path} has no feature key {feature_key!r}")
        road_ids = np.asarray(archive["road_ids"]).astype(str)
        signal_ids = np.asarray(archive["signal_ids"]).astype(str)
        if road_ids_ref is not None and not np.array_equal(road_ids, road_ids_ref):
            raise ValueError(f"road_ids mismatch in {path}")
        if signal_ids_ref is not None and not np.array_equal(signal_ids, signal_ids_ref):
            raise ValueError(f"signal_ids mismatch in {path}")
        feature = np.asarray(archive[feature_key], dtype=np.float32)
        if feature.ndim != 3 or feature.shape[2] != 3:
            raise ValueError(f"{feature_key} in {path} must have shape (T,N,3), got {feature.shape}")
        days.append(
            LoadedDay(
                path=path,
                date=scalar_string(archive, "date", path.stem),
                feature_lsr=feature,
                bucket_start_s=np.asarray(archive["bucket_start_s"], dtype=np.float32),
                bucket_end_s=np.asarray(archive["bucket_end_s"], dtype=np.float32),
                phase_id_end=np.asarray(archive["phase_id_end"], dtype=np.int16),
            )
        )
    return days, archives


def make_adjacency(first: np.lib.npyio.NpzFile) -> np.ndarray:
    n_roads = len(first["road_ids"])
    adj = np.zeros((n_roads, n_roads), dtype=np.float32)
    starts = np.asarray(first["edge_start_road_idx"], dtype=np.int64)
    ends = np.asarray(first["edge_end_road_idx"], dtype=np.int64)
    valid = (starts >= 0) & (starts < n_roads) & (ends >= 0) & (ends < n_roads)
    adj[starts[valid], ends[valid]] = 1.0
    return adj


def make_samples(
    sequences: list[np.ndarray],
    input_window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stamps: list[tuple[int, int]] = []
    for seq_idx, seq in enumerate(sequences):
        n_steps = seq.shape[0]
        for target_idx in range(input_window + horizon - 1, n_steps):
            input_start = target_idx - horizon - input_window + 1
            input_end = input_start + input_window
            target_start = target_idx - horizon + 1
            target_end = target_start + horizon
            if input_start < 0 or target_end > n_steps:
                continue
            xs.append(seq[input_start:input_end].transpose(1, 2, 0)[None, ...])
            ys.append(seq[target_start:target_end].transpose(1, 2, 0)[None, ...])
            stamps.append((seq_idx, target_start))
    if not xs:
        raise ValueError("no samples constructed; input-window/horizon may be too large")
    x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
    timestamp = np.asarray(stamps, dtype=np.int64)
    return x, y, timestamp


def split_indices(num_samples: int, ratios: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_count = int(num_samples * ratios[0])
    val_count = int(num_samples * ratios[1])
    test_count = num_samples - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(f"invalid split counts {(train_count, val_count, test_count)} for {num_samples} samples")
    idx = np.arange(num_samples)
    return idx[:train_count], idx[train_count : train_count + val_count], idx[train_count + val_count :]


def normalize_inputs(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1, 3), keepdims=True)
    std = train_x.std(axis=(0, 1, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def norm(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - mean) / std).astype(np.float32)

    return {"_mean": mean.astype(np.float32), "_std": std.astype(np.float32)}, norm(train_x), norm(val_x), norm(test_x)


def apply_mask_op(data: np.ndarray, road_update: np.ndarray, adj: np.ndarray, rng: random.Random) -> np.ndarray:
    out = data.copy()
    observed = np.where(road_update == 1)[0]
    if len(observed) == 0:
        raise ValueError("mask operation needs at least one observed road")
    for road_idx, value in enumerate(road_update):
        if value == 1:
            continue
        neighbors = [idx for idx, edge in enumerate(adj[:, road_idx]) if edge == 1.0 and road_update[idx] == 1]
        if neighbors:
            out[:, road_idx, :3] = out[:, neighbors, :3].mean(axis=1)
        else:
            out[:, road_idx, :3] = out[:, rng.choice(observed), :3]
    return out


def zero_rate(values: np.ndarray, eps: float = 1e-9) -> float:
    return float((np.abs(values) <= eps).mean())


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    split_ratios = parse_split_ratios(args.split_ratios)
    paths = collect_input_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    first_archive = np.load(paths[0], allow_pickle=True)
    road_ids = np.asarray(first_archive["road_ids"]).astype(str)
    signal_ids = np.asarray(first_archive["signal_ids"]).astype(str)
    days, archives = load_days(paths, args.feature_key, road_ids, signal_ids)
    phase_count = infer_phase_count(archives, args.phase_start_id, args.phase_count)
    road_signal_idx, road_signal_conflicts = build_road_signal_index(first_archive)
    adj_road = make_adjacency(first_archive)
    observed_mask, missing_label = choose_road_mask(first_archive, args.missing_ratio, len(road_ids))
    road_update = np.where(observed_mask, 1, 2).astype(np.int32)
    mask_or = {int(idx): str(road_ids[idx]) for idx in np.where(~observed_mask)[0]}

    road_feature_sequences: list[np.ndarray] = []
    phase_encoded_cells = 0
    for day in days:
        phase_onehot, encoded = encode_road_phase_onehot(
            day.phase_id_end,
            road_signal_idx,
            args.phase_start_id,
            phase_count,
        )
        phase_encoded_cells += encoded
        road_feature_sequences.append(np.concatenate([day.feature_lsr, phase_onehot], axis=2).astype(np.float32))

    samples_x, samples_y, timestamp = make_samples(road_feature_sequences, args.input_window, args.horizon)
    if args.split_mode == "official_shuffle":
        rng = np.random.default_rng(args.shuffle_seed)
        order = np.arange(samples_x.shape[0])
        rng.shuffle(order)
        samples_x = samples_x[order]
        samples_y = samples_y[order]
        timestamp = timestamp[order]
    if args.max_samples is not None and args.max_samples > 0:
        samples_x = samples_x[: args.max_samples]
        samples_y = samples_y[: args.max_samples]
        timestamp = timestamp[: args.max_samples]

    train_idx, val_idx, test_idx = split_indices(samples_x.shape[0], split_ratios)
    train_x_raw, val_x_raw, test_x_raw = samples_x[train_idx], samples_x[val_idx], samples_x[test_idx]
    train_target, val_target, test_target = samples_y[train_idx], samples_y[val_idx], samples_y[test_idx]
    rng = random.Random(args.shuffle_seed)
    if args.apply_official_mask_op:
        train_x_raw = apply_mask_op(train_x_raw, road_update, adj_road, rng)
        train_target = apply_mask_op(train_target, road_update, adj_road, rng)
        val_x_raw = apply_mask_op(val_x_raw, road_update, adj_road, rng)
        test_x_raw = apply_mask_op(test_x_raw, road_update, adj_road, rng)
    stats, train_x, val_x, test_x = normalize_inputs(train_x_raw, val_x_raw, test_x_raw)

    state_payload = {
        "road_feature": road_feature_sequences,
        "adj_road": adj_road,
        "road_update": road_update,
        "mask_or": mask_or,
        "road_ids": road_ids,
        "turn_names": np.asarray(TURN_NAMES),
        "signal_ids": signal_ids,
        "road_signal_idx": road_signal_idx,
        "phase_start_id": args.phase_start_id,
        "phase_count": phase_count,
        "phase_feature_names": np.asarray([f"phase_{args.phase_start_id + i}" for i in range(phase_count)]),
        "feature_names": np.asarray([*TURN_NAMES, *[f"phase_{args.phase_start_id + i}" for i in range(phase_count)]]),
        "source_npz": [str(path) for path in paths],
        "target_semantics": args.feature_key,
    }
    data_split_payload = {
        "train_x": train_x,
        "train_target": train_target,
        "train_timestamp": timestamp[train_idx],
        "val_x": val_x,
        "val_target": val_target,
        "val_timestamp": timestamp[val_idx],
        "test_x": test_x,
        "test_target": test_target,
        "test_timestamp": timestamp[test_idx],
        "mean": stats["_mean"],
        "std": stats["_std"],
        "node_update": road_update,
        "mask_or": mask_or,
        "adj_road": adj_road,
    }
    relation_payload = {
        "road_dict_id2road": {int(i): str(road_id) for i, road_id in enumerate(road_ids)},
        "road_dict_road2id": {str(road_id): int(i) for i, road_id in enumerate(road_ids)},
        "signal_ids": signal_ids,
        "road_signal_idx": road_signal_idx,
        "edge_start_road_idx": np.asarray(first_archive["edge_start_road_idx"], dtype=np.int32),
        "edge_end_road_idx": np.asarray(first_archive["edge_end_road_idx"], dtype=np.int32),
        "edge_turn_type_idx": np.asarray(first_archive["edge_turn_type_idx"], dtype=np.int8),
        "edge_signal_idx": np.asarray(first_archive["edge_signal_idx"], dtype=np.int16),
        "edge_always_active": np.asarray(first_archive["edge_always_active"], dtype=np.uint8),
        "phase_edge_signal_idx": np.asarray(first_archive["phase_edge_signal_idx"], dtype=np.int16),
        "phase_edge_phase_id": np.asarray(first_archive["phase_edge_phase_id"], dtype=np.int16),
        "phase_edge_idx": np.asarray(first_archive["phase_edge_idx"], dtype=np.int32),
        "adj_road": adj_road,
    }

    state_pkl = output_dir / "road_state.pkl"
    split_pkl = output_dir / "data_split_30to1.pkl"
    relation_pkl = output_dir / "roadnet_relation_xuancheng.pkl"
    with state_pkl.open("wb") as f:
        pickle.dump(state_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with split_pkl.open("wb") as f:
        pickle.dump(data_split_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with relation_pkl.open("wb") as f:
        pickle.dump(relation_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.save_npz_split:
        np.savez_compressed(
            output_dir / "data_split_30to1.npz",
            train_x=train_x,
            train_target=train_target,
            train_timestamp=timestamp[train_idx],
            val_x=val_x,
            val_target=val_target,
            val_timestamp=timestamp[val_idx],
            test_x=test_x,
            test_target=test_target,
            test_timestamp=timestamp[test_idx],
            mean=stats["_mean"],
            std=stats["_std"],
            node_update=road_update,
            adj_road=adj_road,
            road_ids=road_ids,
            signal_ids=signal_ids,
            feature_names=state_payload["feature_names"],
        )

    traffic_all = np.concatenate([seq[:, :, :3].reshape(-1, 3) for seq in road_feature_sequences], axis=0)
    summary = {
        "dataset_name": args.dataset_name,
        "output_dir": str(output_dir),
        "input_npz": [str(path) for path in paths],
        "dates": [day.date for day in days],
        "feature_key": args.feature_key,
        "target_interpretation": (
            "mean active vehicles currently on each road, grouped by next left/straight/right intent"
            if args.feature_key == "active_volume_lsr"
            else "custom feature key; inspect source semantics"
        ),
        "num_days": len(days),
        "total_steps": int(sum(seq.shape[0] for seq in road_feature_sequences)),
        "num_roads": int(len(road_ids)),
        "num_signals": int(len(signal_ids)),
        "phase_start_id": int(args.phase_start_id),
        "phase_count": int(phase_count),
        "feature_dim": int(road_feature_sequences[0].shape[2]),
        "input_window": int(args.input_window),
        "horizon": int(args.horizon),
        "split_mode": args.split_mode,
        "split_ratios": list(split_ratios),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "shapes": {
            "road_feature_per_day": [list(seq.shape) for seq in road_feature_sequences],
            "train_x": list(train_x.shape),
            "train_target": list(train_target.shape),
            "val_x": list(val_x.shape),
            "val_target": list(val_target.shape),
            "test_x": list(test_x.shape),
            "test_target": list(test_target.shape),
        },
        "missing_ratio": missing_label,
        "observed_roads": int(observed_mask.sum()),
        "masked_roads": int((~observed_mask).sum()),
        "apply_official_mask_op": bool(args.apply_official_mask_op),
        "adj_edges": int(adj_road.sum()),
        "always_active_edges": int(np.asarray(first_archive["edge_always_active"], dtype=np.uint8).sum()),
        "controlled_roads": int((road_signal_idx >= 0).sum()),
        "phase_encoded_cells": int(phase_encoded_cells),
        "traffic_zero_rate_all_lsr": zero_rate(traffic_all),
        "traffic_zero_rate_road_sum": zero_rate(
            np.concatenate([seq[:, :, :3].sum(axis=2).reshape(-1) for seq in road_feature_sequences], axis=0)
        ),
        "traffic_mean": float(traffic_all.mean()),
        "traffic_positive_mean": float(traffic_all[traffic_all > 0].mean()) if np.any(traffic_all > 0) else 0.0,
        "road_signal_conflict_count": len(road_signal_conflicts),
        "road_signal_conflict_examples": road_signal_conflicts[:10],
        "files": {
            "road_state_pkl": str(state_pkl),
            "data_split_pkl": str(split_pkl),
            "roadnet_relation_pkl": str(relation_pkl),
            "summary_json": str(output_dir / "summary.json"),
        },
        "deviation_notes": [
            "DTIGNN official datasets store grid-like raw intersection states and then build road_feature; here road_feature is built from Xuancheng CityFlow NPZ active road states.",
            "The first 3 channels follow DTIGNN semantics more closely than movement_volume_lsr: they are current road vehicles by next-turn intent, not crossing counts during a bucket.",
            "Xuancheng phase features use CityFlow phase-id one-hot. They are not the original DTIGNN N/E/S/W phase-direction labels.",
            "Non-signal intersections are represented through always-active edges and all-zero road phase features.",
            "The default split is chronological for forecasting safety; use --split-mode official_shuffle to mimic the public prepareData.py shuffle.",
        ],
    }
    write_json(output_dir / "summary.json", summary)

    print(f"[done] wrote {state_pkl}")
    print(f"[done] wrote {split_pkl}")
    print(f"[done] wrote {relation_pkl}")
    print(f"[done] wrote {output_dir / 'summary.json'}")
    print(
        "[summary] "
        f"steps={summary['total_steps']} roads={summary['num_roads']} "
        f"feature_dim={summary['feature_dim']} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"zero_lsr={summary['traffic_zero_rate_all_lsr']:.6f} zero_roadsum={summary['traffic_zero_rate_road_sum']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

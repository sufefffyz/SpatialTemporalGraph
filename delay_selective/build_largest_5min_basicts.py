#!/usr/bin/env python3
"""Build BasicTS-style 5-minute LargeST subsets from CA raw data.

This script codifies the official LargeST notebook logic:
- GBA = CA District 4
- GLA = CA Districts 7, 8, and 12
- SD = CA District 11

Unlike the official training script, this exporter keeps the raw 5-minute
resolution and writes BasicTS-compatible data.dat/desc.json bundles for delay
auditing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_DISTRICTS = {
    "GBA": [4],
    "GLA": [7, 8, 12],
    "SD": [11],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SD/GLA/GBA 5min BasicTS datasets from LargeST CA raw files.")
    parser.add_argument("--largest-root", default="/data/yuzhang_fei/LargeST")
    parser.add_argument("--output-root", default="BasicTS/datasets")
    parser.add_argument("--graph-root", default="graphs")
    parser.add_argument("--datasets", nargs="+", default=["GBA", "GLA"])
    parser.add_argument("--year", default="2019")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--fillna-value", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-subset-h5", action="store_true", help="Also write LargeST/data/<name>/<name>_his_<year>.h5")
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dump_memmap(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp


def build_split_indices(total_len: int, input_len: int, output_len: int) -> dict[str, np.ndarray]:
    min_t = input_len - 1
    max_t = total_len - output_len
    idx = np.arange(min_t, max_t, dtype=np.int64)
    num_train = round(len(idx) * 0.6)
    num_val = round(len(idx) * 0.2)
    return {
        "train_idx": idx[:num_train],
        "val_idx": idx[num_train : num_train + num_val],
        "test_idx": idx[num_train + num_val :],
    }


def build_features(flow_df: pd.DataFrame) -> np.ndarray:
    flow = np.expand_dims(flow_df.to_numpy(dtype=np.float32), axis=-1)
    num_nodes = flow.shape[1]

    time_of_day = (
        (flow_df.index.values - flow_df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    ).astype(np.float32)
    time_of_day = np.tile(time_of_day[:, None, None], (1, num_nodes, 1))

    day_of_week = (flow_df.index.dayofweek.to_numpy(dtype=np.float32) / 7.0).astype(np.float32)
    day_of_week = np.tile(day_of_week[:, None, None], (1, num_nodes, 1))

    return np.concatenate([flow, time_of_day, day_of_week], axis=-1).astype(np.float32)


def load_subset_flow(h5_path: Path, sensor_id_strings: list[str]) -> pd.DataFrame:
    try:
        return pd.read_hdf(h5_path, key="t", columns=sensor_id_strings)
    except (TypeError, ValueError):
        full = pd.read_hdf(h5_path, key="t")
        return full[sensor_id_strings]


def export_one_dataset(
    name: str,
    districts: list[int],
    ca_meta: pd.DataFrame,
    ca_adj: np.ndarray,
    ca_his_path: Path,
    output_root: Path,
    graph_root: Path,
    year: str,
    input_len: int,
    output_len: int,
    fillna_value: float,
    overwrite: bool,
    export_subset_h5: bool,
) -> dict[str, Any]:
    output_dir = output_root / f"{name}_5min_full"
    graph_dir = graph_root / name
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to rebuild it.")

    meta = ca_meta[ca_meta["District"].isin(districts)].reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"No CA sensors found for {name} districts={districts}")

    sensor_ids = pd.to_numeric(meta["ID"], errors="raise").astype(np.int64).to_numpy()
    sensor_id_strings = [str(int(sensor_id)) for sensor_id in sensor_ids]
    id2 = pd.to_numeric(meta["ID2"], errors="raise").astype(np.int64).to_numpy()

    adj = ca_adj[np.ix_(id2, id2)].astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    flow_df = load_subset_flow(ca_his_path, sensor_id_strings)
    missing_before = int(flow_df.isna().sum().sum())
    flow_df = flow_df.fillna(fillna_value).round(0)
    missing_after = int(flow_df.isna().sum().sum())
    data = build_features(flow_df)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    dump_memmap(output_dir / "data.dat", data)
    np.save(output_dir / "shape.npy", np.asarray(data.shape, dtype=np.int64))
    np.save(output_dir / "sensor_ids.npy", sensor_ids)
    (output_dir / "sensor_ids.txt").write_text("\n".join(sensor_id_strings) + "\n")
    meta.to_csv(output_dir / "meta.csv", index=False)

    with (output_dir / "adj_mx_largeST_original.pkl").open("wb") as f:
        pickle.dump(adj, f, protocol=4)
    with (output_dir / "adj_mx.pkl").open("wb") as f:
        pickle.dump(adj, f, protocol=4)

    split = build_split_indices(data.shape[0], input_len, output_len)
    np.savez(output_dir / "split_indices_full.npz", **split)

    desc = {
        "name": f"{name}_5min_full",
        "domain": "traffic flow",
        "shape": list(data.shape),
        "num_time_steps": int(data.shape[0]),
        "num_nodes": int(data.shape[1]),
        "num_features": int(data.shape[2]),
        "feature_description": ["flow", "time of day", "day of week"],
        "has_graph": True,
        "frequency (minutes)": 5,
        "regular_settings": {
            "INPUT_LEN": input_len,
            "OUTPUT_LEN": output_len,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0,
        },
    }
    write_json(output_dir / "desc.json", desc)

    with (graph_dir / "adj_mx_largeST_original.pkl").open("wb") as f:
        pickle.dump(adj, f, protocol=4)
    np.save(graph_dir / "sensor_ids.npy", sensor_ids)
    (graph_dir / "sensor_ids.txt").write_text("\n".join(sensor_id_strings) + "\n")
    meta.to_csv(graph_dir / "meta.csv", index=False)

    subset_h5_path = None
    if export_subset_h5:
        subset_dir = Path("LargeST") / "data" / name.lower()
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_h5_path = subset_dir / f"{name.lower()}_his_{year}.h5"
        flow_df.to_hdf(subset_h5_path, key="t", mode="w")
        meta.to_csv(subset_dir / f"{name.lower()}_meta.csv", index=False)
        np.save(subset_dir / f"{name.lower()}_rn_adj.npy", adj)

    summary = {
        "dataset": name,
        "districts": districts,
        "ca_his_path": str(ca_his_path),
        "output_dir": str(output_dir),
        "graph_dir": str(graph_dir),
        "subset_h5_path": str(subset_h5_path) if subset_h5_path is not None else None,
        "shape": list(data.shape),
        "num_graph_edges": int((adj > 0).sum()),
        "missing_value_summary": {
            "fillna_value": fillna_value,
            "nan_count_before": missing_before,
            "nan_count_after": missing_after,
            "filled_ratio": float(missing_before / (flow_df.shape[0] * flow_df.shape[1])),
        },
        "split_sizes": {key: int(value.shape[0]) for key, value in split.items()},
    }
    write_json(output_dir / "build_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    largest_root = Path(args.largest_root).expanduser().resolve()
    output_root = Path(args.output_root)
    graph_root = Path(args.graph_root)
    ca_meta_path = largest_root / "ca_meta.csv"
    ca_adj_path = largest_root / "ca_rn_adj.npy"
    ca_his_path = largest_root / f"ca_his_raw_{args.year}.h5"

    if not ca_meta_path.exists():
        raise FileNotFoundError(ca_meta_path)
    if not ca_adj_path.exists():
        raise FileNotFoundError(ca_adj_path)
    if not ca_his_path.exists():
        raise FileNotFoundError(ca_his_path)

    ca_meta = pd.read_csv(ca_meta_path)
    ca_adj = np.load(ca_adj_path).astype(np.float32)

    summaries = []
    for raw_name in args.datasets:
        name = raw_name.upper()
        if name not in DATASET_DISTRICTS:
            raise ValueError(f"Unknown dataset {raw_name!r}; expected one of {sorted(DATASET_DISTRICTS)}")
        print(f"[build-largest-5min] {name}", flush=True)
        summaries.append(
            export_one_dataset(
                name=name,
                districts=DATASET_DISTRICTS[name],
                ca_meta=ca_meta,
                ca_adj=ca_adj,
                ca_his_path=ca_his_path,
                output_root=output_root,
                graph_root=graph_root,
                year=args.year,
                input_len=args.input_len,
                output_len=args.output_len,
                fillna_value=args.fillna_value,
                overwrite=args.overwrite,
                export_subset_h5=args.export_subset_h5,
            )
        )

    write_json(Path(args.output_root) / "largest_5min_build_summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

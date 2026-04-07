#!/usr/bin/env python3
"""
Generate BasicTS-ready MVP datasets from PeMS 2025 raw files and graph-construction outputs.

Outputs per district:
- datasets/PEMSD{district}_{year}_full_phys/
- datasets/PEMSD{district}_{year}_full_knn/
or
- datasets/PEMSD{district}_{year}_last{N}_phys/
- datasets/PEMSD{district}_{year}_last{N}_knn/

Each dataset contains:
- data.dat
- desc.json
- adj_mx.pkl
- sensor_ids.npy / sensor_ids.txt
- sensor_catalog.csv
- directed_edges.csv (physical graph only, if available)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_PREP_ROOT = SCRIPT_ROOT.parent
SCRIPTS_ROOT = DATA_PREP_ROOT.parent
BASICTS_ROOT = SCRIPTS_ROOT.parent
GRAPH_OUTPUT_ROOT = BASICTS_ROOT / "PEMS" / "graph_construction" / "outputs"
DEFAULT_DATA_ROOT = Path("/data/yuzhang_fei/PEMS")
DEFAULT_OUTPUT_ROOT = BASICTS_ROOT / "datasets"


def log(message: str) -> None:
    print(message, flush=True)


def resolve_existing_path(path_str: str, desc: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


def normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "id":
            col_map[col] = "ID"
        elif cl == "fwy":
            col_map[col] = "Fwy"
        elif cl in ("dir", "direction"):
            col_map[col] = "Dir"
        elif cl == "district":
            col_map[col] = "District"
        elif cl == "type":
            col_map[col] = "Type"
        elif cl in ("latitude", "lat"):
            col_map[col] = "Latitude"
        elif cl in ("longitude", "lng", "lon"):
            col_map[col] = "Longitude"
        elif cl == "name":
            col_map[col] = "Name"
        elif cl == "abs_pm":
            col_map[col] = "Abs_PM"
        elif cl == "state_pm":
            col_map[col] = "State_PM"
        elif cl == "length":
            col_map[col] = "Length"
        elif cl == "lanes":
            col_map[col] = "Lanes"
    return df.rename(columns=col_map)


def list_district_data_dirs(district: int, data_root: Path, subdir_name: str) -> list[Path]:
    prefix = f"pemsd{district}"
    matched_dirs = []
    for entry in sorted(data_root.expanduser().resolve().iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.lower().startswith(prefix):
            continue
        candidate = entry / subdir_name
        if candidate.exists():
            matched_dirs.append(candidate)
    return matched_dirs


def resolve_station_5min_files(district: int, data_root: Path, year: int) -> list[Path]:
    resolved_files: list[Path] = []
    seen_names = set()
    txt_pattern = f"d{district:02d}_text_station_5min_{year}_*.txt"
    gz_pattern = f"{txt_pattern}.gz"

    for traffic_dir in list_district_data_dirs(district, data_root, "station_5min"):
        for pattern in (txt_pattern, gz_pattern):
            for file_path in sorted(traffic_dir.glob(pattern)):
                if file_path.name not in seen_names:
                    resolved_files.append(file_path)
                    seen_names.add(file_path.name)

    if not resolved_files:
        raise FileNotFoundError(
            f"未找到 District {district} 的 {year} station_5min 文件。"
        )
    return sorted(resolved_files)


def extract_date_from_file(file_path: Path) -> date:
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})\.txt(?:\.gz)?$", file_path.name)
    if not match:
        raise ValueError(f"无法从文件名解析日期: {file_path.name}")
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@dataclass
class DistrictAssets:
    district: int
    label: str
    sensor_ids: np.ndarray
    physical_adj: np.ndarray
    sensor_catalog: pd.DataFrame
    directed_edges_path: Path | None


def load_graph_assets(district: int, year: int, graph_output_root: Path) -> DistrictAssets:
    label = f"PEMSD{district}"
    district_root = graph_output_root / label
    basicts_dir = district_root / "exports" / "basicts"
    analysis_dir = district_root / "exports" / "analysis"
    prepared_dir = district_root / "_prepared_metadata"

    sensor_ids_path = basicts_dir / "sensor_ids.npy"
    adj_path = basicts_dir / "adj_mx.pkl"
    stable_meta_path = prepared_dir / f"d{district:02d}_stable_meta_{year}.txt"
    stable_summary_path = prepared_dir / f"d{district:02d}_stable_meta_{year}_summary.csv"

    sensor_ids = np.load(sensor_ids_path, allow_pickle=True).astype(np.int64)
    physical_adj = np.asarray(load_pickle(adj_path), dtype=np.float32)

    stable_df = pd.read_csv(stable_meta_path, sep="\t", encoding="latin-1")
    stable_df = normalize_metadata_columns(stable_df)
    stable_df["ID"] = pd.to_numeric(stable_df["ID"], errors="coerce")
    stable_df = stable_df.dropna(subset=["ID"]).copy()
    stable_df["ID"] = stable_df["ID"].astype(int)

    if stable_summary_path.exists():
        summary_df = pd.read_csv(stable_summary_path)
        summary_df["ID"] = pd.to_numeric(summary_df["ID"], errors="coerce")
        summary_df = summary_df.dropna(subset=["ID"]).copy()
        summary_df["ID"] = summary_df["ID"].astype(int)
        summary_cols = [
            col
            for col in (
                "ID",
                "presence_count",
                "presence_ratio",
                "total_days",
                "first_seen_date",
                "last_seen_date",
            )
            if col in summary_df.columns
        ]
        stable_df = stable_df.merge(summary_df[summary_cols], on="ID", how="left")

    sensor_catalog = pd.DataFrame({"ID": sensor_ids.tolist()}).merge(stable_df, on="ID", how="left")

    sensor_index_path = analysis_dir / "sensor_index.csv"
    if sensor_index_path.exists():
        sensor_index_df = pd.read_csv(sensor_index_path)
        if "sensor_id" in sensor_index_df.columns:
            sensor_index_df["sensor_id"] = pd.to_numeric(sensor_index_df["sensor_id"], errors="coerce")
            sensor_index_df = sensor_index_df.dropna(subset=["sensor_id"]).copy()
            sensor_index_df["sensor_id"] = sensor_index_df["sensor_id"].astype(int)
            keep_cols = [
                col
                for col in sensor_index_df.columns
                if col not in ("Type", "Fwy", "Dir", "Lanes", "Latitude", "Longitude")
            ]
            sensor_catalog = sensor_catalog.merge(
                sensor_index_df[keep_cols].rename(columns={"sensor_id": "ID"}),
                on="ID",
                how="left",
            )

    for col in ("Latitude", "Longitude"):
        if col not in sensor_catalog.columns:
            raise ValueError(f"{label} 缺少 {col}，无法构建 kNN 图")

    directed_edges_path = analysis_dir / "directed_edges.csv"
    return DistrictAssets(
        district=district,
        label=label,
        sensor_ids=sensor_ids,
        physical_adj=physical_adj,
        sensor_catalog=sensor_catalog,
        directed_edges_path=directed_edges_path if directed_edges_path.exists() else None,
    )


def build_knn_adj(sensor_catalog: pd.DataFrame, k: int) -> np.ndarray:
    coords = sensor_catalog[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce")
    if coords.isna().any().any():
        missing = sensor_catalog.loc[coords.isna().any(axis=1), "ID"].tolist()[:10]
        raise ValueError(f"kNN 图构建失败：缺少坐标的传感器示例 {missing}")

    lat = np.radians(coords["Latitude"].to_numpy(dtype=float))
    lon = np.radians(coords["Longitude"].to_numpy(dtype=float))
    lat1 = lat[:, None]
    lat2 = lat[None, :]
    lon1 = lon[:, None]
    lon2 = lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_km = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 1e-12)))
    np.fill_diagonal(dist_km, np.inf)

    finite_dist = dist_km[np.isfinite(dist_km)]
    sigma = float(np.nanstd(finite_dist))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanmean(finite_dist))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    n = dist_km.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        knn_idx = np.argsort(dist_km[i])[:k]
        weights = np.exp(-(dist_km[i, knn_idx] ** 2) / (sigma**2 + 1e-12)).astype(np.float32)
        adj[i, knn_idx] = weights

    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def build_day_flow_matrix(file_path: Path, sensor_to_idx: dict[int, int], num_nodes: int) -> np.ndarray:
    day_matrix = np.full((288, num_nodes), np.nan, dtype=np.float32)
    chunks = []
    for chunk in pd.read_csv(
        file_path,
        header=None,
        usecols=[0, 1, 9],
        names=["Timestamp", "Station", "Total_Flow"],
        compression="infer",
        chunksize=200_000,
        low_memory=False,
    ):
        chunk["Station"] = pd.to_numeric(chunk["Station"], errors="coerce")
        chunk["Total_Flow"] = pd.to_numeric(chunk["Total_Flow"], errors="coerce")
        chunk = chunk.dropna(subset=["Station", "Total_Flow"]).copy()
        if chunk.empty:
            continue

        chunk["Station"] = chunk["Station"].astype(int)
        chunk = chunk[chunk["Station"].isin(sensor_to_idx)].copy()
        if chunk.empty:
            continue

        timestamps = pd.to_datetime(
            chunk["Timestamp"],
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce",
        )
        chunk = chunk.loc[timestamps.notna()].copy()
        if chunk.empty:
            continue

        valid_timestamps = timestamps.loc[timestamps.notna()]
        chunk["slot"] = valid_timestamps.dt.hour * 12 + valid_timestamps.dt.minute // 5
        chunk["node_idx"] = chunk["Station"].map(sensor_to_idx)
        chunk = chunk.groupby(["slot", "node_idx"], as_index=False)["Total_Flow"].mean()
        chunks.append(chunk)

    if not chunks:
        return day_matrix

    day_df = pd.concat(chunks, ignore_index=True)
    slots = day_df["slot"].to_numpy(dtype=int)
    nodes = day_df["node_idx"].to_numpy(dtype=int)
    values = day_df["Total_Flow"].to_numpy(dtype=np.float32)
    day_matrix[slots, nodes] = values
    return day_matrix


def build_temporal_features(num_days: int, weekdays: list[int], num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    time_of_day = np.tile(np.arange(288, dtype=np.float32) / 288.0, num_days)
    time_of_day = np.repeat(time_of_day[:, None], num_nodes, axis=1)

    day_of_week = np.concatenate(
        [np.full(288, weekday / 7.0, dtype=np.float32) for weekday in weekdays],
        axis=0,
    )
    day_of_week = np.repeat(day_of_week[:, None], num_nodes, axis=1)
    return time_of_day, day_of_week


def safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel_src = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel_src)


def select_daily_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    last_days: int | None,
) -> tuple[list[Path], dict[str, int], list[float]]:
    total_days = len(files)
    test_days = int(total_days * test_ratio)
    val_days = int(total_days * val_ratio)
    train_days_full = total_days - val_days - test_days
    if train_days_full <= 0:
        raise ValueError("训练天数 <= 0，请调整 train/val/test ratio")

    if last_days is None:
        selected_files = files
        train_days = train_days_full
    else:
        if last_days > train_days_full:
            raise ValueError(
                f"--last-days={last_days} 超过可用训练天数 {train_days_full}"
            )
        selected_files = files[train_days_full - last_days :]
        train_days = last_days

    selected_total_days = len(selected_files)
    split_counts = {
        "selected_total_days": selected_total_days,
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "full_total_days": total_days,
        "full_train_days": train_days_full,
    }
    split_ratios = [
        train_days / selected_total_days,
        val_days / selected_total_days,
        test_days / selected_total_days,
    ]
    return selected_files, split_counts, split_ratios


def build_flow_dataset(
    district: int,
    year: int,
    data_root: Path,
    sensor_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    last_days: int | None,
) -> tuple[np.ndarray, list[date], float, dict[str, int], list[float]]:
    files = resolve_station_5min_files(district, data_root, year)
    files, split_counts, split_ratios = select_daily_files(
        files=files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        last_days=last_days,
    )

    sensor_to_idx = {int(sensor_id): idx for idx, sensor_id in enumerate(sensor_ids.tolist())}
    day_matrices = []
    used_dates = []

    for idx, file_path in enumerate(files, start=1):
        used_dates.append(extract_date_from_file(file_path))
        day_matrices.append(build_day_flow_matrix(file_path, sensor_to_idx, len(sensor_ids)))
        if idx == len(files) or idx % 30 == 0:
            log(f"  D{district}: 已解析 {idx}/{len(files)} 天 5min 流量")

    flow = np.stack(day_matrices, axis=0).reshape(len(files) * 288, len(sensor_ids))
    missing_ratio = float(np.isnan(flow).mean())
    valid_counts = np.sum(~np.isnan(flow), axis=0)
    sensor_sum = np.nansum(flow, axis=0)
    sensor_mean = np.divide(
        sensor_sum,
        valid_counts,
        out=np.full(sensor_sum.shape, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    )
    global_mean = float(np.nanmean(flow))
    if not np.isfinite(global_mean):
        global_mean = 0.0
    sensor_mean = np.where(np.isfinite(sensor_mean), sensor_mean, global_mean)
    nan_mask = np.isnan(flow)
    if nan_mask.any():
        flow[nan_mask] = sensor_mean[np.where(nan_mask)[1]]

    weekdays = [day.weekday() for day in used_dates]
    tod, dow = build_temporal_features(len(used_dates), weekdays, len(sensor_ids))
    data = np.stack(
        [
            flow.astype(np.float32),
            tod.astype(np.float32),
            dow.astype(np.float32),
        ],
        axis=-1,
    )
    return data, used_dates, missing_ratio, split_counts, split_ratios


def dump_memmap(path: Path, array: np.ndarray) -> None:
    fp = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp


def save_dataset(
    dataset_dir: Path,
    dataset_name: str,
    data: np.ndarray,
    adj: np.ndarray,
    sensor_ids: np.ndarray,
    sensor_catalog: pd.DataFrame,
    used_dates: list[date],
    missing_ratio: float,
    split_counts: dict[str, int],
    split_ratios: list[float],
    input_len: int,
    output_len: int,
    graph_type: str,
    directed_edges_path: Path | None,
    shared_source_dir: Path | None = None,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if shared_source_dir is None:
        dump_memmap(dataset_dir / "data.dat", data)
    else:
        safe_link(shared_source_dir / "data.dat", dataset_dir / "data.dat")

    desc = {
        "name": dataset_name,
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
            "TRAIN_VAL_TEST_RATIO": split_ratios,
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE", "WAPE"],
            "NULL_VAL": 0.0,
        },
        "mvp_metadata": {
            "graph_type": graph_type,
            "num_days": len(used_dates),
            "start_date": used_dates[0].isoformat(),
            "end_date": used_dates[-1].isoformat(),
            "raw_missing_ratio_before_imputation": missing_ratio,
            "split_counts": split_counts,
        },
    }
    (dataset_dir / "desc.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")

    with open(dataset_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj.astype(np.float32), f, protocol=4)

    if shared_source_dir is None:
        np.save(dataset_dir / "sensor_ids.npy", sensor_ids.astype(np.int64))
        aligned_catalog = pd.DataFrame({"ID": sensor_ids.tolist()}).merge(sensor_catalog, on="ID", how="left")
        aligned_catalog.to_csv(dataset_dir / "sensor_catalog.csv", index=False)
        (dataset_dir / "sensor_ids.txt").write_text(
            "\n".join(str(int(sensor_id)) for sensor_id in sensor_ids.tolist()) + "\n",
            encoding="utf-8",
        )
    else:
        safe_link(shared_source_dir / "sensor_ids.npy", dataset_dir / "sensor_ids.npy")
        safe_link(shared_source_dir / "sensor_catalog.csv", dataset_dir / "sensor_catalog.csv")
        safe_link(shared_source_dir / "sensor_ids.txt", dataset_dir / "sensor_ids.txt")

    if directed_edges_path is not None and directed_edges_path.exists():
        directed_edges_df = pd.read_csv(directed_edges_path)
        directed_edges_df.to_csv(dataset_dir / "directed_edges.csv", index=False)


def format_dataset_base_name(district: int, year: int, num_days: int | None) -> str:
    if num_days is None:
        return f"PEMSD{district}_{year}_full"
    return f"PEMSD{district}_{year}_last{num_days}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BasicTS-ready PeMS MVP datasets.")
    parser.add_argument("--districts", type=int, nargs="+", required=True, help="District IDs, e.g. 3 4")
    parser.add_argument("--year", type=int, default=2025, help="Target year, default 2025")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="PeMS raw data root")
    parser.add_argument(
        "--graph-output-root",
        type=Path,
        default=GRAPH_OUTPUT_ROOT,
        help="Root directory of graph_construction outputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="BasicTS datasets root",
    )
    parser.add_argument("--full", action="store_true", help="Use all daily files in the target year")
    parser.add_argument(
        "--last-days",
        type=int,
        default=60,
        help="Training-window days before fixed val/test splits. Default: 60",
    )
    parser.add_argument("--k", type=int, default=5, help="k for Euclidean kNN graph. Default: 5")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Full-year train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Full-year validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Full-year test split ratio")
    parser.add_argument("--input-len", type=int, default=12, help="Input length, default 12")
    parser.add_argument("--output-len", type=int, default=12, help="Output length, default 12")
    args = parser.parse_args()

    args.data_root = resolve_existing_path(str(args.data_root), "data root")
    args.graph_output_root = resolve_existing_path(str(args.graph_output_root), "graph output root")
    args.output_root = args.output_root.expanduser().resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test ratio 之和必须为 1.0")
    if args.full:
        args.last_days = None
    elif args.last_days is not None and args.last_days <= 0:
        raise ValueError("--last-days 必须为正整数")
    return args


def main() -> int:
    args = parse_args()
    generated_datasets = []

    for district in args.districts:
        log("")
        log("=" * 80)
        log(f"生成 PEMSD{district} BasicTS 数据集")
        log("=" * 80)

        assets = load_graph_assets(district, args.year, args.graph_output_root)
        knn_adj = build_knn_adj(assets.sensor_catalog, args.k)
        data, used_dates, missing_ratio, split_counts, split_ratios = build_flow_dataset(
            district=district,
            year=args.year,
            data_root=args.data_root,
            sensor_ids=assets.sensor_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            last_days=args.last_days,
        )
        dataset_base_name = format_dataset_base_name(
            district,
            args.year,
            None if args.full else len(used_dates),
        )

        shared_source_dir = None
        for graph_type, adj in (("phys", assets.physical_adj), ("knn", knn_adj)):
            dataset_name = f"{dataset_base_name}_{graph_type}"
            dataset_dir = args.output_root / dataset_name
            save_dataset(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                data=data,
                adj=adj,
                sensor_ids=assets.sensor_ids,
                sensor_catalog=assets.sensor_catalog,
                used_dates=used_dates,
                missing_ratio=missing_ratio,
                split_counts=split_counts,
                split_ratios=split_ratios,
                input_len=args.input_len,
                output_len=args.output_len,
                graph_type=graph_type,
                directed_edges_path=assets.directed_edges_path if graph_type == "phys" else None,
                shared_source_dir=shared_source_dir,
            )
            if graph_type == "phys":
                shared_source_dir = dataset_dir
            generated_datasets.append(dataset_name)
            log(f"{dataset_name}: 已写入 {dataset_dir}")

    log("")
    log("已生成数据集:")
    for name in generated_datasets:
        log(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

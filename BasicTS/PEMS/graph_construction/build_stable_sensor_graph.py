#!/usr/bin/env python3
"""
基于 2025 年全年 PeMS 元数据快照，筛选稳定存在的 ML / OR / FR 节点，
然后调用本目录下的 graph construction 管线完成匹配，并默认构建
“物理直接连接图”，导出可用于 BasicTS 训练和可视化分析的文件。

默认原始数据目录:
    /data/yuzhang_fei/PEMS

默认输出目录:
    graph_construction/outputs/PEMSD{district}/

示例:
    python build_stable_sensor_graph.py \
      --districts 3 4 5 6 7 8 10 11 12 \
      --stable-year 2025 \
      --presence-ratio 1.0 \
      --sensor-types ML OR FR \
      --shn-shapefile /path/to/State_Highway_Network_Lines.shp \
      --shn-postmiles-geojson /path/to/SHN_Postmiles_Tenth.geojson
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


GRAPH_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/data/yuzhang_fei/PEMS")
DEFAULT_OUTPUT_ROOT = GRAPH_ROOT / "outputs"
PIPELINE_SCRIPT = GRAPH_ROOT / "network_graph_pipeline2.7.py"
BUILD_SCRIPT = GRAPH_ROOT / "build_sensor_graph_direct.py"


def log(message: str) -> None:
    print(message, flush=True)


def resolve_existing_path(path_str: str, desc: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


def find_station_meta_dir(district: int, data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    meta_dir = root / f"PEMSD{district}" / "station_meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"未找到 District {district} 的 station_meta 目录: {meta_dir}")
    return meta_dir


def resolve_latest_metadata(district: int, data_root: Path) -> Path:
    meta_dir = find_station_meta_dir(district, data_root)
    files = sorted(meta_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"元数据目录为空: {meta_dir}")
    return files[-1]


def resolve_year_metadata_files(district: int, data_root: Path, year: int) -> list[Path]:
    meta_dir = find_station_meta_dir(district, data_root)
    pattern = f"d{district:02d}_text_meta_{year}_*.txt"
    files = sorted(meta_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"未找到 {district} 区 {year} 年元数据文件: {meta_dir / pattern}"
        )
    return files


def infer_district_from_metadata(metadata_file: Path) -> str:
    stem = metadata_file.stem.lower()
    for part in stem.split("_"):
        if part.startswith("d") and part[1:].isdigit():
            return f"PEMSD{int(part[1:]):d}"
    return stem


def infer_district_id(label: str, metadata_file: Path) -> int:
    if label.startswith("PEMSD") and label[5:].isdigit():
        return int(label[5:])

    stem = metadata_file.stem.lower()
    for part in stem.split("_"):
        if part.startswith("d") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"无法从标签或文件名推断 District: {label}, {metadata_file}")


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


def prepare_stable_metadata(
    district: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    files = resolve_year_metadata_files(district, args.data_root, args.stable_year)
    required_count = math.ceil(len(files) * args.presence_ratio)
    sensor_types = set(args.sensor_types)

    log(
        f"稳定节点筛选: District {district} | 年份 {args.stable_year} | "
        f"快照数 {len(files)} | 最少出现次数 {required_count} | 类型 {sorted(sensor_types)}"
    )

    presence_counts: dict[int, int] = {}
    latest_rows: dict[int, pd.Series] = {}
    reference_columns: list[str] | None = None

    for meta_file in files:
        df = pd.read_csv(meta_file, sep="\t", encoding="latin-1")
        df = normalize_metadata_columns(df)
        required_cols = {"ID", "Type"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"{meta_file} 缺少列: {sorted(missing_cols)}")

        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
        df = df[df["ID"].notna()].copy()
        df["ID"] = df["ID"].astype(int)
        df = df[df["Type"].isin(sensor_types)].copy()
        df = df.drop_duplicates(subset=["ID"], keep="last")

        if reference_columns is None:
            reference_columns = list(df.columns)

        for sensor_id in df["ID"].tolist():
            presence_counts[sensor_id] = presence_counts.get(sensor_id, 0) + 1

        for _, row in df.iterrows():
            latest_rows[int(row["ID"])] = row

    stable_ids = sorted(
        sensor_id
        for sensor_id, count in presence_counts.items()
        if count >= required_count
    )
    if not stable_ids:
        raise ValueError(
            f"District {district} 在 {args.stable_year} 年没有满足稳定性条件的传感器"
        )

    stable_rows = [latest_rows[sensor_id] for sensor_id in stable_ids]
    stable_df = pd.DataFrame(stable_rows)
    if reference_columns:
        stable_df = stable_df.reindex(columns=reference_columns)
    stable_df = stable_df.sort_values(["Type", "Fwy", "Dir", "ID"]).reset_index(drop=True)

    metadata_dir = output_dir / "_prepared_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"d{district:02d}_stable_meta_{args.stable_year}.txt"
    stable_df.to_csv(metadata_path, sep="\t", index=False, encoding="latin-1")

    summary_rows = []
    for sensor_id in stable_ids:
        row = latest_rows[sensor_id]
        summary_rows.append(
            {
                "ID": sensor_id,
                "Type": row.get("Type"),
                "Fwy": row.get("Fwy"),
                "Dir": row.get("Dir"),
                "presence_count": presence_counts[sensor_id],
                "presence_ratio": presence_counts[sensor_id] / len(files),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["Type", "Fwy", "Dir", "ID"]
    ).reset_index(drop=True)
    summary_path = metadata_dir / f"d{district:02d}_stable_meta_{args.stable_year}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    type_counts = stable_df["Type"].value_counts().to_dict() if "Type" in stable_df.columns else {}
    log(
        f"稳定元数据已生成: {metadata_path} | 传感器数 {len(stable_df)} | "
        f"类型分布 {type_counts}"
    )
    log(f"稳定性统计已保存: {summary_path}")
    return metadata_path


def export_training_and_analysis_artifacts(output_dir: Path, label: str) -> None:
    sensor_graph_dir = output_dir / "sensor_graph"
    adj_path = sensor_graph_dir / "adjacency_matrix.npy"
    dist_path = sensor_graph_dir / "distance_matrix.npy"
    sensor_ids_path = sensor_graph_dir / "sensor_ids.npy"
    sensor_info_path = output_dir / "phase7_sensor_info.csv"

    missing = [path for path in [adj_path, dist_path, sensor_ids_path, sensor_info_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 7 输出不完整，缺少: " + ", ".join(str(path) for path in missing)
        )

    adj_mx = np.load(adj_path)
    dist_mx = np.load(dist_path)
    sensor_ids = np.load(sensor_ids_path, allow_pickle=True)
    sensor_info_df = pd.read_csv(sensor_info_path)

    sensor_ids = [int(x) for x in sensor_ids.tolist()]
    sensor_index_df = pd.DataFrame(
        {
            "node_index": np.arange(len(sensor_ids), dtype=int),
            "sensor_id": sensor_ids,
        }
    )
    if "sensor_id" in sensor_info_df.columns:
        sensor_info_df["sensor_id"] = pd.to_numeric(sensor_info_df["sensor_id"], errors="coerce")
        sensor_info_df = sensor_info_df.dropna(subset=["sensor_id"]).copy()
        sensor_info_df["sensor_id"] = sensor_info_df["sensor_id"].astype(int)
        sensor_index_df = sensor_index_df.merge(
            sensor_info_df,
            on="sensor_id",
            how="left",
        )

    edges = np.argwhere(adj_mx > 0)
    edge_rows = []
    for from_idx, to_idx in edges:
        edge_rows.append(
            {
                "from_index": int(from_idx),
                "to_index": int(to_idx),
                "from_sensor_id": sensor_ids[int(from_idx)],
                "to_sensor_id": sensor_ids[int(to_idx)],
                "distance_m": float(dist_mx[int(from_idx), int(to_idx)]),
            }
        )
    edge_df = pd.DataFrame(edge_rows)

    export_root = output_dir / "exports"
    basicts_dir = export_root / "basicts"
    analysis_dir = export_root / "analysis"
    basicts_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with open(basicts_dir / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj_mx.astype(np.float32), f)
    with open(basicts_dir / "distance_mx.pkl", "wb") as f:
        pickle.dump(dist_mx.astype(np.float32), f)

    np.save(basicts_dir / "adj_mx.npy", adj_mx.astype(np.float32))
    np.save(basicts_dir / "distance_mx.npy", dist_mx.astype(np.float32))
    np.save(basicts_dir / "sensor_ids.npy", np.array(sensor_ids, dtype=np.int64))

    (basicts_dir / "sensor_ids.txt").write_text(
        "\n".join(str(sensor_id) for sensor_id in sensor_ids) + "\n",
        encoding="utf-8",
    )

    sensor_index_df.to_csv(analysis_dir / "sensor_index.csv", index=False)
    edge_df.to_csv(analysis_dir / "directed_edges.csv", index=False)

    summary = {
        "dataset": label,
        "num_nodes": int(adj_mx.shape[0]),
        "num_directed_edges": int(adj_mx.sum()),
        "adjacency_shape": list(adj_mx.shape),
        "distance_shape": list(dist_mx.shape),
        "has_distance_matrix": True,
        "exports": {
            "basicts_adj_pkl": str(basicts_dir / "adj_mx.pkl"),
            "basicts_distance_pkl": str(basicts_dir / "distance_mx.pkl"),
            "sensor_ids_txt": str(basicts_dir / "sensor_ids.txt"),
            "sensor_index_csv": str(analysis_dir / "sensor_index.csv"),
            "directed_edges_csv": str(analysis_dir / "directed_edges.csv"),
        },
    }
    with open(analysis_dir / "graph_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"{label}: BasicTS/分析产物已导出到 {export_root}")


def build_jobs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []

    for district in args.districts:
        metadata_file = resolve_latest_metadata(district, args.data_root)
        label = f"PEMSD{district}"
        output_dir = args.output_root / label
        jobs.append((label, metadata_file, output_dir))

    for metadata_str in args.metadata_file:
        metadata_file = resolve_existing_path(metadata_str, "元数据文件")
        label = infer_district_from_metadata(metadata_file)
        output_dir = args.output_root / label
        jobs.append((label, metadata_file, output_dir))

    if not jobs:
        raise ValueError("请至少提供 `--districts` 或 `--metadata-file`")

    return jobs


def run_command(cmd: list[str], cwd: Path) -> None:
    log(f"$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_job(
    label: str,
    metadata_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stable_year is not None:
        district = infer_district_id(label, metadata_file)
        metadata_file = prepare_stable_metadata(district, output_dir, args)

    log("")
    log("=" * 80)
    log(f"开始处理 {label}")
    log(f"元数据: {metadata_file}")
    log(f"输出目录: {output_dir}")
    log("=" * 80)

    pipeline_cmd = [
        sys.executable,
        str(args.pipeline_script),
        str(metadata_file),
        "-o",
        str(output_dir),
        "--buffer",
        str(args.buffer),
        "--ml-threshold",
        str(args.ml_threshold),
        "--ramp-threshold",
        str(args.ramp_threshold),
        "--shn-shapefile",
        str(args.shn_shapefile),
        "--shn-postmiles-geojson",
        str(args.shn_postmiles_geojson),
        "--max-edge-length",
        str(args.max_edge_length),
        "--shn-pm-tolerance",
        str(args.shn_pm_tolerance),
        "--shn-max-snap-distance",
        str(args.shn_max_snap_distance),
    ]
    run_command(pipeline_cmd, cwd=GRAPH_ROOT)

    if args.pipeline_only:
        log(f"{label}: 已完成 pipeline 阶段，按要求跳过 Phase 7")
        return

    build_cmd = [
        sys.executable,
        str(args.build_script),
        str(output_dir),
        "--cutoff",
        str(args.cutoff),
    ]
    run_command(build_cmd, cwd=GRAPH_ROOT)
    export_training_and_analysis_artifacts(output_dir, label)

    log(f"{label}: ML/OR/FR 传感器图构建完成")
    log(f"关键产物目录: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 PeMS graph_construction 管线并生成 ML/OR/FR 传感器图"
    )
    parser.add_argument(
        "--districts",
        nargs="*",
        type=int,
        default=[],
        help="要处理的 District 编号，如 3 4 5 6 7 8 10 11 12",
    )
    parser.add_argument(
        "--metadata-file",
        action="append",
        default=[],
        help="直接指定元数据文件路径，可重复传入多个",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"PeMS 数据根目录，默认: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"输出根目录，默认: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        default=PIPELINE_SCRIPT,
        help=f"匹配管线脚本，默认: {PIPELINE_SCRIPT}",
    )
    parser.add_argument(
        "--build-script",
        type=Path,
        default=BUILD_SCRIPT,
        help=f"Phase 7 构图脚本，默认: {BUILD_SCRIPT}",
    )
    parser.add_argument(
        "--shn-shapefile",
        required=True,
        help="Caltrans State Highway Network Lines shapefile 路径 (.shp)",
    )
    parser.add_argument(
        "--shn-postmiles-geojson",
        required=True,
        help="Caltrans SHN Postmiles GeoJSON 路径",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.02,
        help="bbox 缓冲，默认 0.02",
    )
    parser.add_argument(
        "--ml-threshold",
        type=float,
        default=1e-4,
        help="ML 匹配阈值，默认 1e-4",
    )
    parser.add_argument(
        "--ramp-threshold",
        type=float,
        default=0.01,
        help="OR/FR 匹配阈值，默认 0.01",
    )
    parser.add_argument(
        "--max-edge-length",
        type=float,
        default=2000,
        help="边最大长度，默认 2000 米",
    )
    parser.add_argument(
        "--shn-pm-tolerance",
        type=float,
        default=1.5,
        help="SHN postmile 容忍范围，默认 1.5 mile",
    )
    parser.add_argument(
        "--shn-max-snap-distance",
        type=float,
        default=500.0,
        help="SHN 修正最大贴线距离，默认 500 米",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5000,
        help="Phase 7 最短路截断距离，默认 5000 米",
    )
    parser.add_argument(
        "--stable-year",
        type=int,
        default=2025,
        help="从该年的所有元数据快照中筛稳定节点，默认 2025",
    )
    parser.add_argument(
        "--presence-ratio",
        type=float,
        default=1.0,
        help="稳定节点最小出现比例，默认 1.0 表示全年快照全部出现",
    )
    parser.add_argument(
        "--sensor-types",
        nargs="+",
        default=["ML", "OR", "FR"],
        help="稳定筛选时保留的传感器类型，默认 ML OR FR",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="只跑 network matching pipeline，不跑 Phase 7 传感器图",
    )

    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.pipeline_script = resolve_existing_path(str(args.pipeline_script), "pipeline 脚本")
    args.build_script = resolve_existing_path(str(args.build_script), "Phase 7 脚本")
    args.shn_shapefile = resolve_existing_path(args.shn_shapefile, "SHN shapefile")
    args.shn_postmiles_geojson = resolve_existing_path(
        args.shn_postmiles_geojson, "SHN postmiles GeoJSON"
    )
    if args.presence_ratio <= 0 or args.presence_ratio > 1:
        raise ValueError("--presence-ratio 必须在 (0, 1] 范围内")
    args.sensor_types = [sensor_type.upper() for sensor_type in args.sensor_types]
    return args


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)

    for label, metadata_file, output_dir in jobs:
        run_job(label, metadata_file, output_dir, args)

    log("")
    log("全部任务完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

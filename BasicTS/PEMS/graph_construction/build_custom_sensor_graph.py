#!/usr/bin/env python3
"""
运行可定制节点集合的 PeMS 传感器图构建流程。

与 build_stable_sensor_graph.py 的区别：
- 不按全年 station_5min 稳定性筛选节点
- 直接使用用户指定的 metadata 文件
- 可通过显式传入的传感器 ID 集合进一步裁剪节点

常见用法：
    python build_custom_sensor_graph.py \
      --metadata-file /path/to/d11_text_meta_2025_01_01.txt \
      --sensor-ids-file /path/to/sd_sensor_ids.txt \
      --output-dir ./outputs/PEMSD11_custom_sd \
      --shn-shapefile /path/to/State_Highway_Network_Lines.shp \
      --shn-postmiles-geojson /path/to/SHN_Postmiles_Tenth.geojson
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

from build_stable_sensor_graph import (
    BUILD_SCRIPT,
    GRAPH_ROOT,
    PIPELINE_SCRIPT,
    export_training_and_analysis_artifacts,
    infer_district_from_metadata,
    log,
    normalize_metadata_columns,
    resolve_existing_path,
    run_command,
)


def parse_sensor_ids(args: argparse.Namespace) -> list[int] | None:
    ids: list[int] = []

    for raw_value in args.sensor_ids:
        for token in raw_value.replace(",", " ").split():
            token = token.strip()
            if token:
                ids.append(int(token))

    if args.sensor_ids_file is not None:
        sensor_ids_file = resolve_existing_path(args.sensor_ids_file, "传感器 ID 文件")
        for line in sensor_ids_file.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                ids.append(int(stripped))

    if not ids:
        return None

    seen = set()
    ordered = []
    for sensor_id in ids:
        if sensor_id not in seen:
            ordered.append(sensor_id)
            seen.add(sensor_id)
    return ordered


def auto_detect_pbf(output_dir: Path) -> Path | None:
    candidates = [
        output_dir.parent / "california-latest.osm.pbf",
        output_dir.parent.parent / "california-latest.osm.pbf",
        GRAPH_ROOT / "california-latest.osm.pbf",
        GRAPH_ROOT.parent / "california-latest.osm.pbf",
        GRAPH_ROOT.parent.parent / "california-latest.osm.pbf",
        Path("/data/yuzhang_fei/PEMS/california-latest.osm.pbf"),
        Path("/data/yuzhang_fei/california-latest.osm.pbf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def detect_csv_sep(path: Path, fallback: str | None) -> str | None:
    if fallback not in (None, "auto"):
        return fallback
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return None


def map_sensor_type(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "ml": "ML",
        "mainline": "ML",
        "main line": "ML",
        "or": "OR",
        "on-ramp": "OR",
        "on ramp": "OR",
        "onramp": "OR",
        "fr": "FR",
        "off-ramp": "FR",
        "off ramp": "FR",
        "offramp": "FR",
        "ff": "FF",
        "freeway connector": "FF",
    }
    return aliases.get(text, str(value).strip().upper())


def parse_fwy_and_dir_from_label(value: object) -> tuple[object, object]:
    text = str(value).strip().upper()
    if not text:
        return pd.NA, pd.NA
    direction_match = re.search(r"[-_ ]([NSEW])B?$", text)
    direction = direction_match.group(1) if direction_match else pd.NA
    route_match = re.search(r"(\d+)", text)
    route = int(route_match.group(1)) if route_match else pd.NA
    return route, direction


def adapt_large_st_style_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    adapted = metadata_df.copy()

    if "Type" in adapted.columns:
        adapted["Type"] = adapted["Type"].map(map_sensor_type)

    if "Dir" in adapted.columns:
        adapted["Dir"] = adapted["Dir"].astype(str).str.strip().str.upper()

    if "Fwy" in adapted.columns:
        parsed = adapted["Fwy"].map(parse_fwy_and_dir_from_label)
        parsed_routes = parsed.map(lambda item: item[0])
        parsed_dirs = parsed.map(lambda item: item[1])
        adapted["Fwy_raw"] = adapted["Fwy"]
        adapted["Fwy"] = pd.to_numeric(parsed_routes, errors="coerce")
        if "Dir" not in adapted.columns:
            adapted["Dir"] = parsed_dirs
        else:
            adapted["Dir"] = adapted["Dir"].where(
                adapted["Dir"].notna() & (adapted["Dir"].astype(str).str.strip() != ""),
                parsed_dirs,
            )

    return adapted


def prepare_custom_metadata(
    metadata_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    separator = detect_csv_sep(metadata_file, args.metadata_sep)
    read_csv_kwargs = {"encoding": args.metadata_encoding}
    if separator is not None:
        read_csv_kwargs["sep"] = separator
    else:
        read_csv_kwargs["sep"] = None
        read_csv_kwargs["engine"] = "python"

    metadata_df = pd.read_csv(metadata_file, **read_csv_kwargs)
    metadata_df = normalize_metadata_columns(metadata_df)
    metadata_df = adapt_large_st_style_metadata(metadata_df)

    required_cols = {"ID", "Type", "Fwy", "Dir", "Latitude", "Longitude"}
    missing_cols = required_cols - set(metadata_df.columns)
    if missing_cols:
        raise ValueError(f"{metadata_file} 缺少列: {sorted(missing_cols)}")

    metadata_df["ID"] = pd.to_numeric(metadata_df["ID"], errors="coerce")
    metadata_df = metadata_df[metadata_df["ID"].notna()].copy()
    metadata_df["ID"] = metadata_df["ID"].astype(int)
    metadata_df["Fwy"] = pd.to_numeric(metadata_df["Fwy"], errors="coerce")
    metadata_df["Latitude"] = pd.to_numeric(metadata_df["Latitude"], errors="coerce")
    metadata_df["Longitude"] = pd.to_numeric(metadata_df["Longitude"], errors="coerce")
    metadata_df["Dir"] = metadata_df["Dir"].astype(str).str.strip().str.upper()
    metadata_df = metadata_df[
        metadata_df["Fwy"].notna()
        & metadata_df["Latitude"].notna()
        & metadata_df["Longitude"].notna()
        & metadata_df["Dir"].isin(["N", "S", "E", "W"])
    ].copy()
    metadata_df["Fwy"] = metadata_df["Fwy"].astype(int)

    if args.sensor_types:
        sensor_types = {sensor_type.upper() for sensor_type in args.sensor_types}
        metadata_df["Type"] = metadata_df["Type"].astype(str).str.upper()
        metadata_df = metadata_df[metadata_df["Type"].isin(sensor_types)].copy()
    else:
        sensor_types = set(metadata_df["Type"].astype(str).str.upper().unique().tolist())

    selected_ids = parse_sensor_ids(args)
    missing_selected_ids: list[int] = []
    if selected_ids is not None:
        selected_id_set = set(selected_ids)
        present_ids = set(metadata_df["ID"].tolist())
        missing_selected_ids = sorted(selected_id_set - present_ids)
        metadata_df = metadata_df[metadata_df["ID"].isin(selected_id_set)].copy()

    if metadata_df.empty:
        raise ValueError("筛选后的 metadata 为空，请检查传感器类型或 sensor id 集合。")

    metadata_df = metadata_df.drop_duplicates(subset=["ID"], keep="last")
    sort_columns = [column for column in ["Type", "Fwy", "Dir", "ID"] if column in metadata_df.columns]
    metadata_df = metadata_df.sort_values(sort_columns).reset_index(drop=True)

    metadata_dir = output_dir / "_prepared_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or infer_district_from_metadata(metadata_file)
    metadata_out = metadata_dir / f"{label}_custom_meta.txt"
    metadata_df.to_csv(metadata_out, sep="\t", index=False, encoding="latin-1")

    summary_df = metadata_df.copy()
    summary_df["selected"] = True
    summary_out = metadata_dir / f"{label}_custom_meta_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    if selected_ids is not None:
        (metadata_dir / f"{label}_selected_sensor_ids.txt").write_text(
            "\n".join(str(sensor_id) for sensor_id in selected_ids) + "\n",
            encoding="utf-8",
        )

    if missing_selected_ids:
        missing_df = pd.DataFrame({"sensor_id": missing_selected_ids})
        missing_df.to_csv(metadata_dir / f"{label}_missing_selected_sensor_ids.csv", index=False)
        log(f"警告: 共有 {len(missing_selected_ids)} 个指定传感器 ID 不在 metadata 中")

    type_counts = metadata_df["Type"].value_counts().to_dict() if "Type" in metadata_df.columns else {}
    log(
        f"定制 metadata 已生成: {metadata_out} | 传感器数 {len(metadata_df)} | "
        f"类型分布 {type_counts}"
    )
    log(f"定制 metadata 统计已保存: {summary_out}")
    return metadata_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行可定制节点集合的 PeMS 传感器图构建流程"
    )
    parser.add_argument(
        "--metadata-file",
        required=True,
        help="输入 PeMS metadata 文件路径（通常为 tab 分隔 txt）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="该任务的输出目录",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="任务标签，用于 prepared metadata 文件命名；默认从 metadata 文件推断",
    )
    parser.add_argument(
        "--sensor-ids",
        action="append",
        default=[],
        help="显式指定传感器 ID，可重复传入，支持逗号或空格分隔",
    )
    parser.add_argument(
        "--sensor-ids-file",
        default=None,
        help="包含传感器 ID 列表的文本文件，每行一个，可带注释",
    )
    parser.add_argument(
        "--sensor-types",
        nargs="+",
        default=["ML", "OR", "FR"],
        help="保留的传感器类型，默认 ML OR FR",
    )
    parser.add_argument(
        "--metadata-sep",
        default="auto",
        help="metadata 分隔符，默认 auto 自动识别",
    )
    parser.add_argument(
        "--metadata-encoding",
        default="latin-1",
        help="metadata 编码，默认 latin-1",
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
        "--pbf-path",
        type=Path,
        default=None,
        help="本地 OSM PBF 文件路径；会透传给 network_graph_pipeline2.7.py",
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
        help="Phase 7 距离参数。direct 图中表示搜索直接邻居的最大距离",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="只跑 network matching pipeline，不跑 Phase 7 传感器图",
    )

    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.pipeline_script = resolve_existing_path(str(args.pipeline_script), "pipeline 脚本")
    args.build_script = resolve_existing_path(str(args.build_script), "Phase 7 脚本")
    args.shn_shapefile = resolve_existing_path(args.shn_shapefile, "SHN shapefile")
    args.shn_postmiles_geojson = resolve_existing_path(
        args.shn_postmiles_geojson, "SHN postmiles GeoJSON"
    )
    args.metadata_file = resolve_existing_path(args.metadata_file, "元数据文件")
    if args.pbf_path is not None:
        args.pbf_path = resolve_existing_path(str(args.pbf_path), "OSM PBF 文件")
    else:
        detected_pbf = auto_detect_pbf(args.output_dir)
        if detected_pbf is not None:
            args.pbf_path = detected_pbf
            log(f"自动检测到 OSM PBF 文件: {args.pbf_path}")
    args.sensor_types = [sensor_type.upper() for sensor_type in args.sensor_types]
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared_metadata = prepare_custom_metadata(args.metadata_file, args.output_dir, args)

    log("")
    log("=" * 80)
    log(f"开始处理 {args.label or infer_district_from_metadata(args.metadata_file)}")
    log(f"原始 metadata: {args.metadata_file}")
    log(f"裁剪后 metadata: {prepared_metadata}")
    log(f"输出目录: {args.output_dir}")
    log("=" * 80)

    pipeline_cmd = [
        sys.executable,
        str(args.pipeline_script),
        str(prepared_metadata),
        "-o",
        str(args.output_dir),
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
    if args.pbf_path is not None:
        pipeline_cmd.extend(["--pbf-path", str(args.pbf_path)])
    run_command(pipeline_cmd, cwd=GRAPH_ROOT)

    if args.pipeline_only:
        log("已完成 pipeline 阶段，按要求跳过 Phase 7")
        return 0

    build_cmd = [
        sys.executable,
        str(args.build_script),
        str(args.output_dir),
    ]
    build_script_name = Path(args.build_script).name
    if build_script_name == "build_sensor_graph_direct.py":
        if args.cutoff is not None:
            build_cmd.extend(["--max-distance", str(args.cutoff)])
    else:
        build_cmd.extend(["--cutoff", str(args.cutoff)])
    run_command(build_cmd, cwd=GRAPH_ROOT)

    label = args.label or infer_district_from_metadata(args.metadata_file)
    export_training_and_analysis_artifacts(args.output_dir, label)

    log(f"{label}: 定制节点集合传感器图构建完成")
    log(f"关键产物目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

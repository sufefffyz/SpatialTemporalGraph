#!/usr/bin/env python3
"""
Analyze stable PeMS sensors with metadata-aware traffic diagnostics.

Inputs:
- station_5min daily files under /data/yuzhang_fei/PEMS/PEMSD{district}_{year}/station_5min/
- stable metadata and graph exports from graph_construction/outputs/PEMSD{district}/

Outputs:
- type weekday flow profiles
- sensor-day summary table
- sensor-level predictability metrics
- type-level predictability summary
- optional graph-degree vs predictability correlations
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_ROOT = Path(__file__).resolve().parent
PEMS_ROOT = SCRIPT_ROOT.parent
GRAPH_OUTPUT_ROOT = PEMS_ROOT / "graph_construction" / "outputs"
DEFAULT_DATA_ROOT = Path("/data/yuzhang_fei/PEMS")
DEFAULT_OUTPUT_ROOT = PEMS_ROOT / "analysis_outputs" / "sensor_diagnostics"

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
TYPE_ORDER = ["ML", "OR", "FR"]
MAX_LANES = 8

BASE_COLUMNS = [
    "Timestamp",
    "Station",
    "District",
    "Freeway",
    "Direction",
    "Lane_Type",
    "Station_Length",
    "Samples",
    "Pct_Observed",
    "Total_Flow",
    "Avg_Occupancy",
    "Avg_Speed",
]
LANE_COLUMNS = []
for i in range(1, MAX_LANES + 1):
    LANE_COLUMNS.extend(
        [
            f"Lane{i}_Samples",
            f"Lane{i}_Flow",
            f"Lane{i}_Avg_Occ",
            f"Lane{i}_Avg_Speed",
            f"Lane{i}_Observed",
        ]
    )
RAW_COLUMNS = BASE_COLUMNS + LANE_COLUMNS


def log(message: str) -> None:
    print(message, flush=True)


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


def resolve_existing_path(path_str: str, desc: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


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


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    xv = x[mask]
    yv = y[mask]
    if np.nanstd(xv) < 1e-8 or np.nanstd(yv) < 1e-8:
        return math.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def load_stable_metadata(district: int, year: int, graph_output_root: Path) -> pd.DataFrame:
    district_dir = graph_output_root / f"PEMSD{district}"
    meta_path = district_dir / "_prepared_metadata" / f"d{district:02d}_stable_meta_{year}.txt"
    summary_path = (
        district_dir
        / "_prepared_metadata"
        / f"d{district:02d}_stable_meta_{year}_summary.csv"
    )
    meta_df = pd.read_csv(meta_path, sep="\t", encoding="latin-1")
    meta_df = normalize_metadata_columns(meta_df)
    meta_df["ID"] = pd.to_numeric(meta_df["ID"], errors="coerce")
    meta_df = meta_df.dropna(subset=["ID"]).copy()
    meta_df["ID"] = meta_df["ID"].astype(int)

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
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
        meta_df = meta_df.merge(summary_df[summary_cols], on="ID", how="left")

    return meta_df


def load_graph_stats(district: int, graph_output_root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    district_dir = graph_output_root / f"PEMSD{district}" / "exports" / "analysis"
    sensor_index_path = district_dir / "sensor_index.csv"
    edges_path = district_dir / "directed_edges.csv"

    sensor_index_df = None
    edge_stats_df = None

    if sensor_index_path.exists():
        sensor_index_df = pd.read_csv(sensor_index_path)
        if "sensor_id" in sensor_index_df.columns:
            sensor_index_df["sensor_id"] = pd.to_numeric(
                sensor_index_df["sensor_id"], errors="coerce"
            )
            sensor_index_df = sensor_index_df.dropna(subset=["sensor_id"]).copy()
            sensor_index_df["sensor_id"] = sensor_index_df["sensor_id"].astype(int)

    if edges_path.exists():
        edges_df = pd.read_csv(edges_path)
        for col in ("from_sensor_id", "to_sensor_id"):
            edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce")
        edges_df = edges_df.dropna(subset=["from_sensor_id", "to_sensor_id"]).copy()
        edges_df["from_sensor_id"] = edges_df["from_sensor_id"].astype(int)
        edges_df["to_sensor_id"] = edges_df["to_sensor_id"].astype(int)

        outdegree = (
            edges_df.groupby("from_sensor_id")
            .size()
            .rename("outdegree")
            .reset_index()
            .rename(columns={"from_sensor_id": "sensor_id"})
        )
        indegree = (
            edges_df.groupby("to_sensor_id")
            .size()
            .rename("indegree")
            .reset_index()
            .rename(columns={"to_sensor_id": "sensor_id"})
        )
        edge_stats_df = outdegree.merge(indegree, on="sensor_id", how="outer").fillna(0)
        edge_stats_df["outdegree"] = edge_stats_df["outdegree"].astype(int)
        edge_stats_df["indegree"] = edge_stats_df["indegree"].astype(int)
        edge_stats_df["degree"] = edge_stats_df["outdegree"] + edge_stats_df["indegree"]

    return sensor_index_df, edge_stats_df


def build_sensor_catalog(
    district: int,
    year: int,
    graph_output_root: Path,
    sensor_scope: str,
) -> pd.DataFrame:
    stable_df = load_stable_metadata(district, year, graph_output_root)
    sensor_index_df, edge_stats_df = load_graph_stats(district, graph_output_root)

    if sensor_scope == "graph":
        if sensor_index_df is None or sensor_index_df.empty:
            raise FileNotFoundError(
                f"PEMSD{district} 缺少 graph sensor_index.csv，无法使用 graph scope"
            )
        base_graph_df = sensor_index_df[[col for col in ("sensor_id", "node_index") if col in sensor_index_df.columns]]
        catalog_df = base_graph_df.rename(columns={"sensor_id": "ID"}).merge(
            stable_df, on="ID", how="left"
        )
    else:
        catalog_df = stable_df.copy()

    if edge_stats_df is not None:
        catalog_df = catalog_df.merge(
            edge_stats_df.rename(columns={"sensor_id": "ID"}), on="ID", how="left"
        )

    for col in ("outdegree", "indegree", "degree"):
        if col in catalog_df.columns:
            catalog_df[col] = catalog_df[col].fillna(0).astype(int)

    if "Type" in catalog_df.columns:
        catalog_df["Type"] = catalog_df["Type"].astype(str).str.upper()
    catalog_df = catalog_df.drop_duplicates(subset=["ID"]).reset_index(drop=True)
    return catalog_df


def load_daily_sensor_data(
    file_path: Path,
    sensor_ids: set[int],
    sensor_catalog: pd.DataFrame,
) -> pd.DataFrame:
    usecols = [
        "Timestamp",
        "Station",
        "Samples",
        "Pct_Observed",
        "Total_Flow",
        "Avg_Occupancy",
        "Avg_Speed",
    ]
    chunks = []
    for chunk in pd.read_csv(
        file_path,
        header=None,
        names=RAW_COLUMNS,
        usecols=usecols,
        compression="infer",
        chunksize=200_000,
        low_memory=False,
    ):
        chunk["Station"] = pd.to_numeric(chunk["Station"], errors="coerce")
        chunk = chunk.dropna(subset=["Station"]).copy()
        chunk["Station"] = chunk["Station"].astype(int)
        chunk = chunk[chunk["Station"].isin(sensor_ids)].copy()
        if chunk.empty:
            continue

        for col in ("Samples", "Pct_Observed", "Total_Flow", "Avg_Occupancy", "Avg_Speed"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    day_df = pd.concat(chunks, ignore_index=True)
    day_df["Timestamp"] = pd.to_datetime(
        day_df["Timestamp"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )
    day_df = day_df.dropna(subset=["Timestamp"]).copy()
    day_df["slot"] = day_df["Timestamp"].dt.hour * 12 + day_df["Timestamp"].dt.minute // 5

    meta_cols = [col for col in ("ID", "Type", "Fwy", "Dir", "Lanes") if col in sensor_catalog.columns]
    meta_df = sensor_catalog[meta_cols].rename(columns={"ID": "Station"})
    day_df = day_df.merge(meta_df, on="Station", how="left")
    day_df = day_df.dropna(subset=["Type"]).copy()
    day_df["Type"] = day_df["Type"].astype(str).str.upper()
    return day_df


def init_sensor_stat() -> dict[str, float]:
    return {
        "value_count": 0,
        "flow_sum": 0.0,
        "flow_sumsq": 0.0,
        "day_count": 0,
        "coverage_sum": 0.0,
        "peak_sum": 0.0,
        "lag1_corr_sum": 0.0,
        "lag1_corr_count": 0,
        "prev_day_corr_sum": 0.0,
        "prev_day_corr_count": 0,
        "prev_week_corr_sum": 0.0,
        "prev_week_corr_count": 0,
    }


def update_type_slot_accumulator(
    accumulator: dict[tuple[str, str, int], dict[str, float]],
    district_label: str,
    weekday: str,
    day_df: pd.DataFrame,
) -> None:
    grouped = (
        day_df.groupby(["Type", "slot"])["Total_Flow"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "flow_sum", "count": "flow_count"})
    )
    for row in grouped.itertuples(index=False):
        key = (district_label, str(row.Type), weekday, int(row.slot))
        stats = accumulator.setdefault(key, {"flow_sum": 0.0, "flow_count": 0})
        stats["flow_sum"] += float(row.flow_sum)
        stats["flow_count"] += int(row.flow_count)


def build_sensor_day_summary(
    district_label: str,
    day: date,
    weekday: str,
    day_df: pd.DataFrame,
    sensor_catalog: pd.DataFrame,
) -> pd.DataFrame:
    agg_df = (
        day_df.groupby("Station")
        .agg(
            observed_slots=("slot", "nunique"),
            num_records=("slot", "size"),
            total_flow=("Total_Flow", "sum"),
            mean_flow=("Total_Flow", "mean"),
            peak_flow=("Total_Flow", "max"),
            std_flow=("Total_Flow", "std"),
            mean_occ=("Avg_Occupancy", "mean"),
            mean_speed=("Avg_Speed", "mean"),
            mean_pct_observed=("Pct_Observed", "mean"),
        )
        .reset_index()
        .rename(columns={"Station": "ID"})
    )
    agg_df["coverage_ratio"] = agg_df["observed_slots"] / 288.0
    agg_df["date"] = day.isoformat()
    agg_df["weekday"] = weekday
    agg_df["district"] = district_label

    meta_cols = [col for col in sensor_catalog.columns if col not in ("node_index",)]
    agg_df = agg_df.merge(sensor_catalog[meta_cols], on="ID", how="left")
    return agg_df


def update_sensor_predictability(
    sensor_stats: dict[int, dict[str, float]],
    recent_profiles: dict[int, deque[tuple[date, np.ndarray]]],
    day: date,
    pivot_df: pd.DataFrame,
) -> None:
    for sensor_id, row in pivot_df.iterrows():
        profile = row.to_numpy(dtype=float)
        valid_mask = np.isfinite(profile)
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            continue

        stats = sensor_stats.setdefault(int(sensor_id), init_sensor_stat())
        stats["value_count"] += valid_count
        stats["flow_sum"] += float(np.nansum(profile))
        stats["flow_sumsq"] += float(np.nansum(profile[valid_mask] ** 2))
        stats["day_count"] += 1
        stats["coverage_sum"] += valid_count / 288.0
        stats["peak_sum"] += float(np.nanmax(profile))

        pair_mask = np.isfinite(profile[:-1]) & np.isfinite(profile[1:])
        if int(pair_mask.sum()) >= 3:
            lag1_corr = safe_corr(profile[:-1][pair_mask], profile[1:][pair_mask])
            if np.isfinite(lag1_corr):
                stats["lag1_corr_sum"] += lag1_corr
                stats["lag1_corr_count"] += 1

        history = recent_profiles.setdefault(int(sensor_id), deque(maxlen=8))
        prev_day_profile = None
        prev_week_profile = None
        for prev_date, prev_profile in history:
            if prev_date == day - timedelta(days=1):
                prev_day_profile = prev_profile
            elif prev_date == day - timedelta(days=7):
                prev_week_profile = prev_profile

        if prev_day_profile is not None:
            corr = safe_corr(profile, prev_day_profile)
            if np.isfinite(corr):
                stats["prev_day_corr_sum"] += corr
                stats["prev_day_corr_count"] += 1

        if prev_week_profile is not None:
            corr = safe_corr(profile, prev_week_profile)
            if np.isfinite(corr):
                stats["prev_week_corr_sum"] += corr
                stats["prev_week_corr_count"] += 1

        history.append((day, profile.copy()))


def finalize_sensor_predictability(
    district_label: str,
    sensor_stats: dict[int, dict[str, float]],
    sensor_catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for sensor_id, stats in sensor_stats.items():
        value_count = int(stats["value_count"])
        day_count = int(stats["day_count"])
        mean_flow = stats["flow_sum"] / value_count if value_count > 0 else math.nan
        variance = stats["flow_sumsq"] / value_count - mean_flow**2 if value_count > 0 else math.nan
        variance = max(variance, 0.0) if np.isfinite(variance) else math.nan
        flow_std = math.sqrt(variance) if np.isfinite(variance) else math.nan
        rows.append(
            {
                "district": district_label,
                "ID": sensor_id,
                "value_count": value_count,
                "day_count": day_count,
                "mean_flow": mean_flow,
                "flow_std": flow_std,
                "flow_cv": (flow_std / mean_flow) if mean_flow and np.isfinite(mean_flow) else math.nan,
                "mean_coverage_ratio": (
                    stats["coverage_sum"] / day_count if day_count > 0 else math.nan
                ),
                "mean_peak_flow": (
                    stats["peak_sum"] / day_count if day_count > 0 else math.nan
                ),
                "mean_daily_lag1_autocorr": (
                    stats["lag1_corr_sum"] / stats["lag1_corr_count"]
                    if stats["lag1_corr_count"] > 0
                    else math.nan
                ),
                "mean_prev_day_profile_corr": (
                    stats["prev_day_corr_sum"] / stats["prev_day_corr_count"]
                    if stats["prev_day_corr_count"] > 0
                    else math.nan
                ),
                "mean_prev_week_profile_corr": (
                    stats["prev_week_corr_sum"] / stats["prev_week_corr_count"]
                    if stats["prev_week_corr_count"] > 0
                    else math.nan
                ),
            }
        )

    predict_df = pd.DataFrame(rows)
    meta_cols = [col for col in sensor_catalog.columns if col not in ("node_index",)]
    predict_df = predict_df.merge(sensor_catalog[meta_cols], on="ID", how="left")
    return predict_df


def summarize_type_weekday_profiles(profile_accumulator: dict[tuple[str, str, str, int], dict[str, float]]) -> pd.DataFrame:
    rows = []
    for (district, sensor_type, weekday, slot), stats in profile_accumulator.items():
        rows.append(
            {
                "district": district,
                "Type": sensor_type,
                "weekday": weekday,
                "slot": slot,
                "mean_flow": stats["flow_sum"] / stats["flow_count"]
                if stats["flow_count"] > 0
                else math.nan,
                "flow_count": int(stats["flow_count"]),
            }
        )
    profile_df = pd.DataFrame(rows)
    if not profile_df.empty:
        profile_df["weekday"] = pd.Categorical(
            profile_df["weekday"], categories=WEEKDAY_ORDER, ordered=True
        )
        profile_df = profile_df.sort_values(["district", "Type", "weekday", "slot"]).reset_index(
            drop=True
        )
    return profile_df


def summarize_sensor_days(sensor_day_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sensor_day_df.groupby(["district", "Type", "weekday"], as_index=False)
        .agg(
            sensor_day_count=("ID", "size"),
            num_sensors=("ID", "nunique"),
            mean_daily_total_flow=("total_flow", "mean"),
            median_daily_total_flow=("total_flow", "median"),
            mean_daily_peak_flow=("peak_flow", "mean"),
            mean_daily_speed=("mean_speed", "mean"),
            mean_daily_occ=("mean_occ", "mean"),
            mean_coverage_ratio=("coverage_ratio", "mean"),
        )
    )
    grouped["weekday"] = pd.Categorical(grouped["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    return grouped.sort_values(["district", "Type", "weekday"]).reset_index(drop=True)


def summarize_predictability(sensor_predict_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        sensor_predict_df.groupby(["district", "Type"], as_index=False)
        .agg(
            num_sensors=("ID", "nunique"),
            mean_flow=("mean_flow", "mean"),
            mean_flow_cv=("flow_cv", "mean"),
            mean_coverage_ratio=("mean_coverage_ratio", "mean"),
            mean_daily_lag1_autocorr=("mean_daily_lag1_autocorr", "mean"),
            mean_prev_day_profile_corr=("mean_prev_day_profile_corr", "mean"),
            mean_prev_week_profile_corr=("mean_prev_week_profile_corr", "mean"),
            mean_peak_flow=("mean_peak_flow", "mean"),
        )
    )
    return summary_df.sort_values(["district", "Type"]).reset_index(drop=True)


def compute_graph_predictability_correlations(sensor_predict_df: pd.DataFrame) -> pd.DataFrame:
    if "degree" not in sensor_predict_df.columns:
        return pd.DataFrame()

    rows = []
    metric_cols = [
        "mean_flow",
        "flow_cv",
        "mean_daily_lag1_autocorr",
        "mean_prev_day_profile_corr",
        "mean_prev_week_profile_corr",
    ]

    for district, district_df in sensor_predict_df.groupby("district"):
        for sensor_type, subset in district_df.groupby("Type"):
            for degree_col in ("indegree", "outdegree", "degree"):
                if degree_col not in subset.columns:
                    continue
                for metric_col in metric_cols:
                    corr = safe_corr(
                        subset[degree_col].to_numpy(dtype=float),
                        subset[metric_col].to_numpy(dtype=float),
                    )
                    rows.append(
                        {
                            "district": district,
                            "Type": sensor_type,
                            "degree_metric": degree_col,
                            "target_metric": metric_col,
                            "pearson_corr": corr,
                            "num_sensors": int(subset["ID"].nunique()),
                        }
                    )

    return pd.DataFrame(rows)


def plot_weekday_profiles(profile_df: pd.DataFrame, output_path: Path, district_label: str) -> None:
    if profile_df.empty:
        return

    available_types = [t for t in TYPE_ORDER if t in set(profile_df["Type"])]
    if not available_types:
        available_types = sorted(profile_df["Type"].unique().tolist())

    fig, axes = plt.subplots(
        len(available_types),
        1,
        figsize=(14, 3.6 * len(available_types)),
        sharex=True,
        squeeze=False,
    )
    color_map = plt.cm.get_cmap("tab10", len(WEEKDAY_ORDER))

    for row_idx, sensor_type in enumerate(available_types):
        ax = axes[row_idx, 0]
        subset = profile_df[profile_df["Type"] == sensor_type]
        for weekday_idx, weekday in enumerate(WEEKDAY_ORDER):
            curve = subset[subset["weekday"] == weekday].sort_values("slot")
            if curve.empty:
                continue
            ax.plot(
                curve["slot"],
                curve["mean_flow"],
                label=weekday,
                linewidth=1.8,
                color=color_map(weekday_idx),
            )
        ax.set_title(f"{district_label} | {sensor_type}")
        ax.set_ylabel("Mean Flow")
        ax.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("5-min Slot")
    axes[0, 0].legend(ncol=4, fontsize=9, loc="upper right")
    fig.suptitle(f"{district_label} Weekday Flow Profiles by Sensor Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_weekday_summary(summary_df: pd.DataFrame, output_path: Path, district_label: str) -> None:
    if summary_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    metrics = [
        ("mean_daily_total_flow", "Mean Daily Total Flow"),
        ("mean_daily_peak_flow", "Mean Daily Peak Flow"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for sensor_type in TYPE_ORDER:
            subset = summary_df[summary_df["Type"] == sensor_type]
            if subset.empty:
                continue
            subset = subset.sort_values("weekday")
            ax.plot(
                subset["weekday"].astype(str),
                subset[metric],
                marker="o",
                linewidth=2,
                label=sensor_type,
            )
        ax.set_title(f"{district_label} | {title}")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Weekday")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_predictability(summary_df: pd.DataFrame, output_path: Path, district_label: str) -> None:
    if summary_df.empty:
        return

    metrics = [
        "mean_daily_lag1_autocorr",
        "mean_prev_day_profile_corr",
        "mean_prev_week_profile_corr",
    ]
    titles = [
        "Mean Daily Lag-1 Autocorr",
        "Mean Prev-Day Profile Corr",
        "Mean Prev-Week Profile Corr",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        plot_df = summary_df.set_index("Type").reindex(TYPE_ORDER).reset_index()
        ax.bar(plot_df["Type"], plot_df[metric], color=["#2E86AB", "#F18F01", "#C73E1D"])
        ax.set_title(f"{district_label} | {title}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_district(
    district: int,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    district_label = f"PEMSD{district}"
    sensor_catalog = build_sensor_catalog(
        district=district,
        year=args.year,
        graph_output_root=args.graph_output_root,
        sensor_scope=args.sensor_scope,
    )
    sensor_catalog = sensor_catalog[sensor_catalog["Type"].isin(args.sensor_types)].copy()
    sensor_catalog = sensor_catalog.drop_duplicates(subset=["ID"]).reset_index(drop=True)
    sensor_ids = set(sensor_catalog["ID"].tolist())

    if not sensor_ids:
        raise ValueError(f"{district_label} 没有可分析的传感器")

    files = resolve_station_5min_files(district, args.data_root, args.year)
    if args.max_days is not None:
        files = files[: args.max_days]

    log(
        f"{district_label}: 传感器数 {len(sensor_catalog)} | "
        f"分析天数 {len(files)} | scope={args.sensor_scope}"
    )

    type_slot_accumulator: dict[tuple[str, str, str, int], dict[str, float]] = {}
    sensor_day_frames: list[pd.DataFrame] = []
    sensor_stats: dict[int, dict[str, float]] = {}
    recent_profiles: dict[int, deque[tuple[date, np.ndarray]]] = {}

    for idx, file_path in enumerate(files, start=1):
        day = extract_date_from_file(file_path)
        weekday = WEEKDAY_ORDER[day.weekday()]
        day_df = load_daily_sensor_data(file_path, sensor_ids, sensor_catalog)
        if day_df.empty:
            continue

        update_type_slot_accumulator(type_slot_accumulator, district_label, weekday, day_df)

        sensor_day_df = build_sensor_day_summary(
            district_label=district_label,
            day=day,
            weekday=weekday,
            day_df=day_df,
            sensor_catalog=sensor_catalog,
        )
        sensor_day_df = sensor_day_df[sensor_day_df["coverage_ratio"] >= args.min_coverage].copy()
        if not sensor_day_df.empty:
            sensor_day_frames.append(sensor_day_df)

        pivot_df = day_df.pivot_table(
            index="Station",
            columns="slot",
            values="Total_Flow",
            aggfunc="mean",
        ).reindex(columns=range(288))
        if not sensor_day_df.empty:
            pivot_df = pivot_df[pivot_df.index.isin(sensor_day_df["ID"].tolist())]
        update_sensor_predictability(sensor_stats, recent_profiles, day, pivot_df)

        if idx == len(files) or idx % 30 == 0:
            log(f"  {district_label}: 已处理 {idx}/{len(files)} 天")

    if sensor_day_frames:
        sensor_day_df = pd.concat(sensor_day_frames, ignore_index=True)
    else:
        sensor_day_df = pd.DataFrame()

    profile_df = summarize_type_weekday_profiles(type_slot_accumulator)
    sensor_predict_df = finalize_sensor_predictability(
        district_label=district_label,
        sensor_stats=sensor_stats,
        sensor_catalog=sensor_catalog,
    )
    type_weekday_summary_df = summarize_sensor_days(sensor_day_df) if not sensor_day_df.empty else pd.DataFrame()
    type_predict_summary_df = (
        summarize_predictability(sensor_predict_df) if not sensor_predict_df.empty else pd.DataFrame()
    )
    graph_corr_df = (
        compute_graph_predictability_correlations(sensor_predict_df)
        if not sensor_predict_df.empty
        else pd.DataFrame()
    )

    return {
        "sensor_catalog": sensor_catalog,
        "sensor_day": sensor_day_df,
        "type_weekday_profile": profile_df,
        "sensor_predictability": sensor_predict_df,
        "type_weekday_summary": type_weekday_summary_df,
        "type_predictability_summary": type_predict_summary_df,
        "graph_predictability_correlation": graph_corr_df,
    }


def write_outputs(
    district_label: str,
    district_results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for name, df in district_results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(csv_dir / f"{name}.csv", index=False)

    plot_weekday_profiles(
        district_results["type_weekday_profile"],
        fig_dir / "weekday_profiles_by_type.png",
        district_label,
    )
    plot_weekday_summary(
        district_results["type_weekday_summary"],
        fig_dir / "weekday_summary_by_type.png",
        district_label,
    )
    plot_predictability(
        district_results["type_predictability_summary"],
        fig_dir / "predictability_by_type.png",
        district_label,
    )

    summary = {
        "district": district_label,
        "num_catalog_sensors": int(district_results["sensor_catalog"]["ID"].nunique()),
        "num_sensor_days": int(len(district_results["sensor_day"])),
        "num_predictability_rows": int(len(district_results["sensor_predictability"])),
        "outputs": {
            "csv_dir": str(csv_dir),
            "figure_dir": str(fig_dir),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stable PeMS sensors by metadata type, weekday profile, and predictability."
    )
    parser.add_argument("--districts", type=int, nargs="+", required=True, help="District IDs, e.g. 3 4")
    parser.add_argument("--year", type=int, default=2025, help="Target year, default 2025")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="PeMS raw data root")
    parser.add_argument(
        "--graph-output-root",
        type=Path,
        default=GRAPH_OUTPUT_ROOT,
        help="graph_construction outputs root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="analysis output root",
    )
    parser.add_argument(
        "--sensor-scope",
        choices=["graph", "stable"],
        default="graph",
        help="Analyze final graph sensors or all stable sensors. Default: graph",
    )
    parser.add_argument(
        "--sensor-types",
        nargs="+",
        default=["ML", "OR", "FR"],
        help="Sensor types to include. Default: ML OR FR",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="Minimum within-day slot coverage ratio for sensor-day summary. Default: 0.9",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Optional cap on number of days, useful for quick smoke tests",
    )
    args = parser.parse_args()
    args.data_root = resolve_existing_path(str(args.data_root), "data root")
    args.graph_output_root = resolve_existing_path(str(args.graph_output_root), "graph output root")
    args.output_root = args.output_root.expanduser().resolve()
    args.sensor_types = [sensor_type.upper() for sensor_type in args.sensor_types]
    return args


def main() -> int:
    args = parse_args()
    combined: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for district in args.districts:
        district_label = f"PEMSD{district}"
        log("")
        log("=" * 80)
        log(f"分析 {district_label}")
        log("=" * 80)
        district_results = analyze_district(district, args)
        district_output_dir = args.output_root / district_label
        write_outputs(district_label, district_results, district_output_dir)
        for name, df in district_results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                combined[name].append(df)
        log(f"{district_label}: 结果已写入 {district_output_dir}")

    combined_dir = args.output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in combined.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(combined_dir / f"{name}.csv", index=False)

    log("")
    log(f"全部分析完成，输出目录: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

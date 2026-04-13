"""
PeMS Sensor to OSM Road Network Matching Pipeline
===================================================
分阶段执行, 每步保存中间结果, 便于逐步检验.

Phase 0: 加载元数据, 自动计算 bbox
Phase 1: 提取 OSM 路网, 构建关联矩阵
Phase 2: ref 标签归属高速公路, SHN 方向判定
Phase 3: ML (主线) 传感器匹配
Phase 4: 添加匝道, OR/FR 传感器匹配
Phase 5: FF 传感器标记
Phase 6: 构建邻接矩阵, 汇总输出

依赖: pip install osmnx geopandas pandas numpy folium shapely
"""

import os
import re
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from shapely.geometry import Point, LineString
from shapely.ops import linemerge
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================================
# OSM ref → PeMS Fwy 编号 特殊映射
# =============================================================================
# 当 OSM 路牌编号与 PeMS 立法路线编号不一致时使用
# 格式: {"OSM relation/way ref 字符串": PeMS_Fwy_编号}
# 通用情况 (如 "I 5" → 5, "CA 99" → 99) 由代码自动提取数字, 不需在此列出
REF_TO_PEMS_FWY = {
    "I 80 Bus": 51,         # SR 51 (unsigned) = Business I-80 (Capital City Fwy)
    "I 80 Business": 51,
    "I 305": 305,            # I-305 (unsigned) in Sacramento
}

# SHN Direction → PeMS Dir 映射
SHN_DIR_TO_PEMS = {"NB": "N", "SB": "S", "EB": "E", "WB": "W"}
PEMS_DIR_TO_SHN = {v: k for k, v in SHN_DIR_TO_PEMS.items()}


# =============================================================================
# 工具函数
# =============================================================================
def log(msg, level="INFO"):
    """带时间戳的日志输出."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def save_intermediate(obj, name, output_dir):
    """保存中间结果 (pickle + csv/html)."""
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(obj, f)
    log(f"  Saved {pkl_path}")

    # 如果是 DataFrame, 同时保存 csv
    if isinstance(obj, pd.DataFrame):
        csv_path = os.path.join(output_dir, f"{name}.csv")
        obj.to_csv(csv_path)
        log(f"  Saved {csv_path} ({obj.shape[0]} rows × {obj.shape[1]} cols)")


def load_intermediate(name, output_dir):
    """加载中间结果."""
    pkl_path = os.path.join(output_dir, f"{name}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"中间结果不存在: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def haversine_deg(lat1, lon1, lat2, lon2):
    """欧氏距离 (度单位, 用于坐标距离计算)."""
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def deg_to_meter(dist_deg):
    """粗略将经纬度差转换为米."""
    return dist_deg * 111320.0


def get_linear_geometry(geometry):
    """规范化线性几何, 便于后续投影或插值."""
    if geometry is None or geometry.is_empty:
        return None

    linear_geom = geometry
    if geometry.geom_type == "MultiLineString":
        try:
            linear_geom = linemerge(geometry)
        except Exception:
            linear_geom = geometry
    return linear_geom


def project_point_to_linear_geometry(point, geometry):
    """将点投影到线性几何上, 返回投影点."""
    linear_geom = get_linear_geometry(geometry)
    if linear_geom is None:
        return None

    try:
        return linear_geom.interpolate(linear_geom.project(point))
    except Exception:
        return None


def interpolate_point_by_postmile(abs_pm, geometry, begin_pm, end_pm):
    """根据段起终点 postmile 在线几何上插值."""
    if pd.isna(abs_pm) or pd.isna(begin_pm) or pd.isna(end_pm):
        return None

    linear_geom = get_linear_geometry(geometry)
    if linear_geom is None:
        return None

    pm_span = end_pm - begin_pm
    if np.isclose(pm_span, 0.0):
        fraction = 0.5
    else:
        fraction = (abs_pm - begin_pm) / pm_span

    fraction = float(np.clip(fraction, 0.0, 1.0))

    try:
        return linear_geom.interpolate(fraction, normalized=True)
    except Exception:
        return None


def expected_align_code(dir_value):
    """根据 PeMS 方向给出 SHN 期望对齐侧."""
    if dir_value in ("N", "E"):
        return "Right"
    if dir_value in ("S", "W"):
        return "Left"
    return None


def build_shn_route_index(shn_gdf):
    """按 Route + Direction 建 SHN 索引."""
    route_index = {}
    for (route_num, direction), group in shn_gdf.groupby(["Route", "Direction"]):
        route_index[(int(route_num), str(direction))] = group.copy()
    return route_index


def interpolate_between_postmile_points(abs_pm, lower_row, upper_row):
    """基于前后 postmile 点做线性插值."""
    lower_pm = pd.to_numeric(lower_row.get("odometer_value"), errors="coerce")
    upper_pm = pd.to_numeric(upper_row.get("odometer_value"), errors="coerce")
    lower_geom = lower_row.geometry
    upper_geom = upper_row.geometry

    if lower_geom is None or upper_geom is None:
        return None

    if pd.isna(lower_pm) or pd.isna(upper_pm):
        return None

    if np.isclose(lower_pm, upper_pm):
        return Point(lower_geom.x, lower_geom.y)

    fraction = (abs_pm - lower_pm) / (upper_pm - lower_pm)
    fraction = float(np.clip(fraction, 0.0, 1.0))

    lon = lower_geom.x + fraction * (upper_geom.x - lower_geom.x)
    lat = lower_geom.y + fraction * (upper_geom.y - lower_geom.y)
    return Point(lon, lat)


def build_postmile_route_index(postmile_gdf):
    """按 Route + 左右对齐侧建 SHN postmile 点索引."""
    route_index = {}
    sort_cols = [col for col in ["odometer_value", "OBJECTID"] if col in postmile_gdf.columns]
    for route_num, group in postmile_gdf.groupby("Route"):
        route_index[int(route_num)] = {
            "right": group[group["align_side"] == "right"].sort_values(sort_cols).reset_index(drop=True),
            "left": group[group["align_side"] == "left"].sort_values(sort_cols).reset_index(drop=True),
        }
    return route_index


def get_postmile_candidate(abs_pm, pm_group, pm_tolerance):
    """在同一对齐侧的 postmile 点序列中为一个传感器找到前后点."""
    if pd.isna(abs_pm) or len(pm_group) == 0:
        return None

    sorted_group = pm_group.sort_values(["odometer_value", "OBJECTID"]).reset_index(drop=True)
    odometer_values = sorted_group["odometer_value"].to_numpy(dtype=float)

    nearest_idx = int(np.argmin(np.abs(odometer_values - abs_pm)))
    nearest_gap = float(abs(odometer_values[nearest_idx] - abs_pm))
    if nearest_gap <= max(pm_tolerance, 0.05):
        row = sorted_group.iloc[nearest_idx]
        point = Point(row.geometry.x, row.geometry.y)
        return {
            "point": point,
            "lower_row": row,
            "upper_row": row,
            "lower_pm": float(row["odometer_value"]),
            "upper_pm": float(row["odometer_value"]),
            "pm_in_range": True,
            "pm_gap": nearest_gap,
            "mode": "pm_exact",
        }

    insert_idx = int(np.searchsorted(odometer_values, abs_pm))
    if insert_idx == 0:
        row = sorted_group.iloc[0]
        point = Point(row.geometry.x, row.geometry.y)
        return {
            "point": point,
            "lower_row": row,
            "upper_row": row,
            "lower_pm": float(row["odometer_value"]),
            "upper_pm": float(row["odometer_value"]),
            "pm_in_range": False,
            "pm_gap": float(abs(row["odometer_value"] - abs_pm)),
            "mode": "before_start",
        }
    if insert_idx >= len(odometer_values):
        row = sorted_group.iloc[-1]
        point = Point(row.geometry.x, row.geometry.y)
        return {
            "point": point,
            "lower_row": row,
            "upper_row": row,
            "lower_pm": float(row["odometer_value"]),
            "upper_pm": float(row["odometer_value"]),
            "pm_in_range": False,
            "pm_gap": float(abs(row["odometer_value"] - abs_pm)),
            "mode": "after_end",
        }

    lower_row = sorted_group.iloc[insert_idx - 1]
    upper_row = sorted_group.iloc[insert_idx]
    lower_pm = float(lower_row["odometer_value"])
    upper_pm = float(upper_row["odometer_value"])

    if lower_pm > abs_pm or upper_pm < abs_pm:
        return None
    if (abs_pm - lower_pm) > pm_tolerance or (upper_pm - abs_pm) > pm_tolerance:
        return None

    point = interpolate_between_postmile_points(abs_pm, lower_row, upper_row)
    if point is None:
        return None

    return {
        "point": point,
        "lower_row": lower_row,
        "upper_row": upper_row,
        "lower_pm": lower_pm,
        "upper_pm": upper_pm,
        "pm_in_range": True,
        "pm_gap": max(abs_pm - lower_pm, upper_pm - abs_pm),
        "mode": "pm_interpolate",
    }


def save_phase0_correction_map(sensors_df, correction_details_df, output_dir, postmile_gdf=None):
    """保存 Phase 0 原始点与修正结果的对比地图."""
    if len(sensors_df) == 0:
        return

    center_lat = float(sensors_df["raw_lat"].mean())
    center_lon = float(sensors_df["raw_lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    fg_shift = folium.FeatureGroup(name="位移连线", show=True)
    fg_raw = folium.FeatureGroup(name="原始点", show=False)
    fg_corrected = folium.FeatureGroup(name="已修正点", show=True)
    fg_target_only = folium.FeatureGroup(name="候选目标点(未应用)", show=False)
    fg_postmile = folium.FeatureGroup(name="SHN 锚点里程桩", show=False)

    details_by_id = correction_details_df.set_index("sensor_id")

    def detail_value(sensor_id, column, default=np.nan):
        if column not in details_by_id.columns or sensor_id not in details_by_id.index:
            return default
        value = details_by_id.loc[sensor_id, column]
        if isinstance(value, pd.Series):
            return value.iloc[0]
        return value

    if postmile_gdf is not None and len(postmile_gdf) > 0:
        anchor_rows = []
        for _, detail in correction_details_df.iterrows():
            for side in ("lower", "upper"):
                lat = pd.to_numeric(detail.get(f"{side}_anchor_lat"), errors="coerce")
                lon = pd.to_numeric(detail.get(f"{side}_anchor_lon"), errors="coerce")
                if pd.isna(lat) or pd.isna(lon):
                    continue
                anchor_rows.append({
                    "sensor_id": int(detail["sensor_id"]),
                    "anchor_side": side,
                    "lat": float(lat),
                    "lon": float(lon),
                    "route": detail.get("shn_route"),
                    "direction": detail.get("shn_direction"),
                    "align": detail.get("shn_align_code"),
                    "pmrouteid": detail.get("shn_pmrouteid"),
                    "pm": detail.get(f"{side}_anchor_pm"),
                    "odometer": detail.get(f"{side}_anchor_odometer"),
                })

        anchor_df = pd.DataFrame(anchor_rows).drop_duplicates(
            subset=["anchor_side", "lat", "lon", "route", "pmrouteid", "odometer"]
        ) if anchor_rows else pd.DataFrame()

        for _, pm in anchor_df.iterrows():
            radius = 5 if pm["anchor_side"] == "lower" else 4
            color = "#1565C0" if pm["anchor_side"] == "lower" else "#00ACC1"
            popup = (
                f"<b>Anchor:</b> {pm['anchor_side']}<br>"
                f"<b>Route:</b> {pm.get('route', '')}<br>"
                f"<b>Direction:</b> {pm.get('direction', '')}<br>"
                f"<b>Align:</b> {pm.get('align', '')}<br>"
                f"<b>PM:</b> {pm.get('pm', 'NA')}<br>"
                f"<b>Odometer:</b> {pm.get('odometer', 'NA')}<br>"
                f"<b>PMRouteID:</b> {pm.get('pmrouteid', '')}<br>"
                f"<b>Coords:</b> ({pm['lat']:.6f}, {pm['lon']:.6f})<br>"
            )
            tooltip = f"{pm['anchor_side']} R{pm.get('route', '')} Odo={pm.get('odometer', 'NA')}"

            folium.CircleMarker(
                location=[pm["lat"], pm["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup, max_width=350),
                tooltip=tooltip,
            ).add_to(fg_postmile)

    for _, sensor in sensors_df.iterrows():
        sensor_id = int(sensor["ID"])
        raw_lat = float(sensor["raw_lat"])
        raw_lon = float(sensor["raw_lon"])
        corrected_lat = float(sensor["corrected_lat"])
        corrected_lon = float(sensor["corrected_lon"])
        target_lat = sensor.get("shn_target_lat")
        target_lon = sensor.get("shn_target_lon")
        correction_method = sensor.get("correction_method", "raw")
        correction_dist_m = sensor.get("correction_dist_m")
        status = "unknown"
        if sensor_id in details_by_id.index:
            status = detail_value(sensor_id, "status", "unknown")
            lower_anchor_lat = pd.to_numeric(detail_value(sensor_id, "lower_anchor_lat"), errors="coerce")
            lower_anchor_lon = pd.to_numeric(detail_value(sensor_id, "lower_anchor_lon"), errors="coerce")
            upper_anchor_lat = pd.to_numeric(detail_value(sensor_id, "upper_anchor_lat"), errors="coerce")
            upper_anchor_lon = pd.to_numeric(detail_value(sensor_id, "upper_anchor_lon"), errors="coerce")
        else:
            lower_anchor_lat = np.nan
            lower_anchor_lon = np.nan
            upper_anchor_lat = np.nan
            upper_anchor_lon = np.nan

        popup = (
            f"<b>ID:</b> {sensor_id}<br>"
            f"<b>Type:</b> {sensor.get('Type', '')}<br>"
            f"<b>Fwy/Dir:</b> {sensor.get('Fwy', '')} {sensor.get('Dir', '')}<br>"
            f"<b>Name:</b> {sensor.get('Name', '')}<br>"
            f"<b>Status:</b> {status}<br>"
            f"<b>Method:</b> {correction_method}<br>"
            f"<b>Shift:</b> {correction_dist_m if pd.notna(correction_dist_m) else 'NA'} m<br>"
            f"<b>Raw:</b> ({raw_lat:.6f}, {raw_lon:.6f})<br>"
            f"<b>Applied:</b> ({corrected_lat:.6f}, {corrected_lon:.6f})<br>"
        )

        folium.CircleMarker(
            location=[raw_lat, raw_lon],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.75,
            popup=folium.Popup(popup, max_width=350),
            tooltip=f"raw {sensor_id}",
        ).add_to(fg_raw)

        if str(correction_method).startswith("route_dir_pm"):
            folium.CircleMarker(
                location=[corrected_lat, corrected_lon],
                radius=4,
                color="#2ca02c",
                fill=True,
                fill_color="#2ca02c",
                fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=350),
                tooltip=f"corrected {sensor_id}",
            ).add_to(fg_corrected)

            folium.PolyLine(
                locations=[[raw_lat, raw_lon], [corrected_lat, corrected_lon]],
                color="#ff7f0e",
                weight=1.5,
                opacity=0.8,
                popup=folium.Popup(popup, max_width=350),
            ).add_to(fg_shift)
            if pd.notna(lower_anchor_lat) and pd.notna(lower_anchor_lon):
                folium.PolyLine(
                    locations=[[float(lower_anchor_lat), float(lower_anchor_lon)], [corrected_lat, corrected_lon]],
                    color="#1565C0",
                    weight=1.0,
                    opacity=0.45,
                ).add_to(fg_shift)
            if pd.notna(upper_anchor_lat) and pd.notna(upper_anchor_lon):
                folium.PolyLine(
                    locations=[[float(upper_anchor_lat), float(upper_anchor_lon)], [corrected_lat, corrected_lon]],
                    color="#00ACC1",
                    weight=1.0,
                    opacity=0.45,
                ).add_to(fg_shift)
        elif pd.notna(target_lat) and pd.notna(target_lon):
            target_popup = popup + (
                f"<b>Target:</b> ({float(target_lat):.6f}, {float(target_lon):.6f})<br>"
            )
            folium.CircleMarker(
                location=[float(target_lat), float(target_lon)],
                radius=4,
                color="#d62728",
                fill=True,
                fill_color="#d62728",
                fill_opacity=0.9,
                popup=folium.Popup(target_popup, max_width=350),
                tooltip=f"target {sensor_id}",
            ).add_to(fg_target_only)

            folium.PolyLine(
                locations=[[raw_lat, raw_lon], [float(target_lat), float(target_lon)]],
                color="#d62728",
                weight=1.2,
                opacity=0.5,
                popup=folium.Popup(target_popup, max_width=350),
            ).add_to(fg_shift)

    fg_shift.add_to(m)
    fg_raw.add_to(m)
    fg_corrected.add_to(m)
    fg_target_only.add_to(m)
    fg_postmile.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    map_path = os.path.join(output_dir, "phase0_correction_map.html")
    m.save(map_path)
    log(f"  Phase 0 修正地图已保存: {map_path}")


def correct_sensor_coordinates_with_postmiles(
    sensors_df,
    shn_postmiles_geojson,
    output_dir,
    pm_tolerance=1.5,
    max_snap_distance_m=500.0,
):
    """
    基于 SHN_Postmiles_Tenth.geojson 对传感器坐标做受限修正.

    约束:
    1. 先按 Fwy + Dir 筛选 SHN 候选
    2. 在同一 PMRouteID 内, 用前后 postmile 点对做线性插值
    3. 不再依赖 SHN_Lines 线段几何做投影回退
    """
    if not shn_postmiles_geojson:
        raise ValueError("必须提供 shn_postmiles_geojson 用于 2.7 坐标修正")
    if not os.path.exists(shn_postmiles_geojson):
        raise FileNotFoundError(f"找不到 SHN Postmiles 文件: {shn_postmiles_geojson}")

    log("Step 0.1: 基于 SHN_Postmiles_Tenth.geojson 做传感器坐标修正...")
    log(f"  SHN Postmiles: {shn_postmiles_geojson}")

    postmile_gdf = gpd.read_file(shn_postmiles_geojson).to_crs(epsg=4326)
    log(f"  SHN Postmile 点数: {len(postmile_gdf)}")

    numeric_cols = ["Route", "PM", "PMoffset", "Odometer", "District", "OBJECTID"]
    for col in numeric_cols:
        if col in postmile_gdf.columns:
            postmile_gdf[col] = pd.to_numeric(postmile_gdf[col], errors="coerce")

    postmile_gdf["odometer_value"] = pd.to_numeric(postmile_gdf["Odometer"], errors="coerce")
    postmile_gdf["align_side"] = "unknown"
    align_code_series = postmile_gdf.get("AlignCode", pd.Series(index=postmile_gdf.index, dtype=object)).fillna("").astype(str)
    postmile_gdf.loc[align_code_series.str.contains("right", case=False, na=False), "align_side"] = "right"
    postmile_gdf.loc[align_code_series.str.contains("left", case=False, na=False), "align_side"] = "left"

    postmile_gdf = postmile_gdf[
        postmile_gdf["Route"].notna()
        & postmile_gdf["odometer_value"].notna()
        & postmile_gdf["align_side"].isin(["left", "right"])
        & postmile_gdf.geometry.notna()
        & (~postmile_gdf.geometry.is_empty)
    ].copy()

    route_index = build_postmile_route_index(postmile_gdf)
    corrected_df = sensors_df.copy()
    corrected_df["raw_lat"] = corrected_df["Latitude"]
    corrected_df["raw_lon"] = corrected_df["Longitude"]
    corrected_df["corrected_lat"] = corrected_df["Latitude"]
    corrected_df["corrected_lon"] = corrected_df["Longitude"]
    corrected_df["correction_dist_m"] = np.nan
    corrected_df["correction_method"] = "raw"
    corrected_df["shn_route"] = pd.NA
    corrected_df["shn_direction"] = pd.NA
    corrected_df["shn_align_code"] = pd.NA
    corrected_df["shn_pm_match"] = False
    corrected_df["shn_projection_ok"] = False
    corrected_df["shn_correction_applied"] = False
    corrected_df["shn_target_lat"] = np.nan
    corrected_df["shn_target_lon"] = np.nan
    corrected_df["shn_pmrouteid"] = pd.NA

    records = []

    for idx, sensor in corrected_df.iterrows():
        sensor_id = int(sensor["ID"])
        route_num = pd.to_numeric(sensor.get("Fwy"), errors="coerce")
        dir_value = str(sensor.get("Dir", "")).strip()
        abs_pm = pd.to_numeric(sensor.get("Abs_PM"), errors="coerce")
        point = Point(sensor["raw_lon"], sensor["raw_lat"])

        record = {
            "sensor_id": sensor_id,
            "fwy": sensor.get("Fwy"),
            "dir": dir_value,
            "raw_lat": sensor["raw_lat"],
            "raw_lon": sensor["raw_lon"],
            "corrected_lat": sensor["raw_lat"],
            "corrected_lon": sensor["raw_lon"],
            "correction_dist_m": np.nan,
            "correction_method": "raw",
            "shn_route": pd.NA,
            "shn_direction": pd.NA,
            "shn_align_code": pd.NA,
            "pm_in_range": False,
            "n_candidates": 0,
            "shn_target_lat": np.nan,
            "shn_target_lon": np.nan,
            "shn_pmrouteid": pd.NA,
            "status": "no_route_or_dir",
        }

        if pd.isna(route_num) or dir_value not in PEMS_DIR_TO_SHN:
            records.append(record)
            continue
        if pd.isna(abs_pm):
            record["status"] = "no_abs_pm"
            records.append(record)
            continue

        shn_direction = PEMS_DIR_TO_SHN[dir_value]
        route_candidates = route_index.get(int(route_num))
        if route_candidates is None:
            record["status"] = "no_shn_candidates"
            records.append(record)
            continue
        align_side = "right" if dir_value in ("N", "E") else "left"
        align_candidates = route_candidates.get(align_side)
        if align_candidates is None or len(align_candidates) == 0:
            record["status"] = "no_align_candidates"
            records.append(record)
            continue

        candidate = get_postmile_candidate(abs_pm, align_candidates, pm_tolerance)
        record["n_candidates"] = int(len(align_candidates))
        record["shn_route"] = int(route_num)
        record["shn_direction"] = shn_direction

        if candidate is None:
            record["status"] = "no_postmile_bracket"
            records.append(record)
            continue

        correction_point = candidate["point"]
        distance_m = deg_to_meter(point.distance(correction_point))
        record["correction_dist_m"] = float(distance_m)
        record["shn_align_code"] = candidate["lower_row"].get("AlignCode", pd.NA)
        record["pm_in_range"] = bool(candidate["pm_in_range"])
        record["shn_target_lat"] = correction_point.y
        record["shn_target_lon"] = correction_point.x
        record["shn_pmrouteid"] = candidate["lower_row"].get("PMRouteID", pd.NA)
        record["lower_anchor_lat"] = candidate["lower_row"].geometry.y
        record["lower_anchor_lon"] = candidate["lower_row"].geometry.x
        record["upper_anchor_lat"] = candidate["upper_row"].geometry.y
        record["upper_anchor_lon"] = candidate["upper_row"].geometry.x
        record["lower_anchor_pm"] = candidate["lower_row"].get("PM", pd.NA)
        record["upper_anchor_pm"] = candidate["upper_row"].get("PM", pd.NA)
        record["lower_anchor_odometer"] = candidate["lower_row"].get("odometer_value", pd.NA)
        record["upper_anchor_odometer"] = candidate["upper_row"].get("odometer_value", pd.NA)
        corrected_df.at[idx, "shn_target_lat"] = correction_point.y
        corrected_df.at[idx, "shn_target_lon"] = correction_point.x
        corrected_df.at[idx, "shn_pmrouteid"] = candidate["lower_row"].get("PMRouteID", pd.NA)

        if distance_m <= max_snap_distance_m:
            corrected_df.at[idx, "Latitude"] = correction_point.y
            corrected_df.at[idx, "Longitude"] = correction_point.x
            corrected_df.at[idx, "corrected_lat"] = correction_point.y
            corrected_df.at[idx, "corrected_lon"] = correction_point.x
            corrected_df.at[idx, "correction_dist_m"] = distance_m
            corrected_df.at[idx, "correction_method"] = (
                "route_dir_pm_exact"
                if candidate["mode"] == "pm_exact"
                else "route_dir_pm_interpolate"
            )
            corrected_df.at[idx, "shn_route"] = int(route_num)
            corrected_df.at[idx, "shn_direction"] = shn_direction
            corrected_df.at[idx, "shn_align_code"] = candidate["lower_row"].get("AlignCode", pd.NA)
            corrected_df.at[idx, "shn_pm_match"] = bool(candidate["pm_in_range"])
            corrected_df.at[idx, "shn_projection_ok"] = False
            corrected_df.at[idx, "shn_correction_applied"] = True

            record["corrected_lat"] = correction_point.y
            record["corrected_lon"] = correction_point.x
            record["correction_method"] = corrected_df.at[idx, "correction_method"]
            record["status"] = "corrected"
        else:
            corrected_df.at[idx, "correction_dist_m"] = distance_m
            corrected_df.at[idx, "correction_method"] = "raw_distance_too_large"
            corrected_df.at[idx, "shn_route"] = int(route_num)
            corrected_df.at[idx, "shn_direction"] = shn_direction
            corrected_df.at[idx, "shn_align_code"] = candidate["lower_row"].get("AlignCode", pd.NA)
            corrected_df.at[idx, "shn_pm_match"] = bool(candidate["pm_in_range"])
            corrected_df.at[idx, "shn_projection_ok"] = False
            record["correction_method"] = "raw_distance_too_large"
            record["status"] = "distance_too_large"

        records.append(record)

    correction_details_df = pd.DataFrame(records)
    correction_summary_df = (
        correction_details_df.groupby(["status", "correction_method"], dropna=False)
        .agg(count=("sensor_id", "count"),
             mean_dist_m=("correction_dist_m", "mean"),
             median_dist_m=("correction_dist_m", "median"))
        .reset_index()
        .sort_values(["status", "correction_method"])
    )

    save_intermediate(correction_details_df, "phase0_shn_corrections", output_dir)
    save_intermediate(correction_summary_df, "phase0_shn_correction_summary", output_dir)
    save_phase0_correction_map(corrected_df, correction_details_df, output_dir, postmile_gdf)

    n_corrected = int((correction_details_df["status"] == "corrected").sum())
    log(f"  SHN postmile 修正成功: {n_corrected}/{len(corrected_df)}")
    return corrected_df


def split_long_edges(G, max_length_m=2000):
    """
    拆分超长边: 对 length > max_length_m 的边, 沿几何等分插入虚拟节点.

    虚拟节点使用负 ID (从 -1 递减), 避免与 OSM 节点冲突.
    子边继承原边的所有属性 (highway, ref, osmid 等).
    """
    from shapely.ops import substring

    edges_to_split = []
    for u, v, key, data in G.edges(keys=True, data=True):
        length = data.get("length", 0)
        if length > max_length_m:
            edges_to_split.append((u, v, key, dict(data)))

    if not edges_to_split:
        log(f"  无超过 {max_length_m}m 的边, 无需拆分")
        return G

    log(f"  需要拆分的超长边: {len(edges_to_split)} 条")

    # 虚拟节点 ID 计数器 (负数, 避免冲突)
    virtual_id = -1

    split_count = 0
    new_node_count = 0

    for u, v, key, data in edges_to_split:
        length = data["length"]
        n_segs = int(np.ceil(length / max_length_m))
        if n_segs < 2:
            continue

        # 获取几何 (无 geometry 属性时用端点直线)
        geom = data.get("geometry", None)
        if geom is None:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            geom = LineString([
                (u_data["x"], u_data["y"]),
                (v_data["x"], v_data["y"]),
            ])

        # 沿几何等分, 插入 n_segs-1 个虚拟节点
        chain = [u]
        sub_geoms = []

        for i in range(1, n_segs):
            frac = i / n_segs
            pt = geom.interpolate(frac, normalized=True)
            vid = virtual_id
            virtual_id -= 1
            G.add_node(vid, x=pt.x, y=pt.y, street_count=0)
            chain.append(vid)
            new_node_count += 1

            # 子段几何
            sub_line = substring(geom, (i - 1) / n_segs, i / n_segs,
                                 normalized=True)
            sub_geoms.append(sub_line)

        chain.append(v)
        # 最后一段
        sub_geoms.append(
            substring(geom, (n_segs - 1) / n_segs, 1.0, normalized=True)
        )

        # 继承原边属性 (去除 length 和 geometry, 重新计算)
        base_attrs = {k: val for k, val in data.items()
                      if k not in ("length", "geometry")}

        # 删除原边
        G.remove_edge(u, v, key)

        # 添加子边
        for i in range(n_segs):
            su, sv = chain[i], chain[i + 1]
            sub_g = sub_geoms[i]
            sub_len = length / n_segs  # 等分长度 (米)
            attrs = dict(base_attrs)
            attrs["length"] = sub_len
            if sub_g is not None and not sub_g.is_empty:
                attrs["geometry"] = sub_g
            G.add_edge(su, sv, **attrs)

        split_count += 1

    log(f"  拆分完成: {split_count} 条边 → 新增 {new_node_count} 个虚拟节点")
    return G


# =============================================================================
# Phase 0: 加载元数据, 自动计算 bbox
# =============================================================================
def phase0_load_metadata(
    metadata_file,
    output_dir,
    buffer_deg=0.02,
    shn_postmiles_geojson=None,
    shn_pm_tolerance=1.5,
    shn_max_snap_distance_m=500.0,
):
    """
    加载 PeMS 元数据, 根据传感器坐标自动计算 bounding box.

    参数:
        metadata_file: PeMS 元数据文件 (tab 分隔)
        output_dir: 输出目录
        buffer_deg: bbox 外扩缓冲 (度), 默认 0.02 ≈ 2km

    输出:
        sensors_df: 传感器 DataFrame
        bbox_dict: {north, south, east, west}
    """
    log("=" * 70)
    log("Phase 0: 加载 PeMS 元数据 & 计算 Bounding Box")
    log("=" * 70)

    # --- 加载元数据 ---
    log(f"读取文件: {metadata_file}")
    df = pd.read_csv(metadata_file, sep="\t", encoding="latin-1")

    # 标准化列名
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "id":
            col_map[col] = "ID"
        elif cl == "fwy":
            col_map[col] = "Fwy"
        elif cl in ("dir", "direction"):
            col_map[col] = "Dir"
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
        elif cl == "length":
            col_map[col] = "Length"
    df = df.rename(columns=col_map)

    # 过滤无效坐标
    valid_mask = (
        df["Latitude"].notna()
        & df["Longitude"].notna()
        & (df["Latitude"] != 0)
        & (df["Longitude"] != 0)
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        log(f"  过滤掉 {invalid_count} 条无效坐标记录", "WARN")
    sensors_df = df[valid_mask].copy()
    sensors_df["ID"] = sensors_df["ID"].astype(int)

    if "Abs_PM" in sensors_df.columns:
        sensors_df["Abs_PM"] = pd.to_numeric(sensors_df["Abs_PM"], errors="coerce")
    if "Length" in sensors_df.columns:
        sensors_df["Length"] = pd.to_numeric(sensors_df["Length"], errors="coerce")

    log(f"  加载 {len(sensors_df)} 条有效传感器记录")

    # --- 传感器类型统计 ---
    type_counts = sensors_df["Type"].value_counts()
    log(f"  传感器类型分布:")
    for t, c in type_counts.items():
        log(f"    {t}: {c}")

    # --- 高速公路统计 ---
    fwy_counts = sensors_df["Fwy"].value_counts().sort_index()
    log(f"  涉及 {len(fwy_counts)} 条高速公路: {sorted(fwy_counts.index.tolist())}")

    # --- 方向统计 ---
    dir_counts = sensors_df["Dir"].value_counts()
    log(f"  方向分布: {dict(dir_counts)}")

    # --- SHN 线路坐标修正 ---
    sensors_df = correct_sensor_coordinates_with_postmiles(
        sensors_df,
        shn_postmiles_geojson=shn_postmiles_geojson,
        output_dir=output_dir,
        pm_tolerance=shn_pm_tolerance,
        max_snap_distance_m=shn_max_snap_distance_m,
    )

    # --- 计算 bbox ---
    lat_min = sensors_df["Latitude"].min()
    lat_max = sensors_df["Latitude"].max()
    lon_min = sensors_df["Longitude"].min()
    lon_max = sensors_df["Longitude"].max()

    bbox_dict = {
        "north": lat_max + buffer_deg,
        "south": lat_min - buffer_deg,
        "east": lon_max + buffer_deg,
        "west": lon_min - buffer_deg,
    }

    raw_bbox_dict = {
        "north": sensors_df["raw_lat"].max() + buffer_deg,
        "south": sensors_df["raw_lat"].min() - buffer_deg,
        "east": sensors_df["raw_lon"].max() + buffer_deg,
        "west": sensors_df["raw_lon"].min() - buffer_deg,
    }

    log(f"  传感器坐标范围:")
    log(f"    纬度: [{lat_min:.4f}, {lat_max:.4f}]")
    log(f"    经度: [{lon_min:.4f}, {lon_max:.4f}]")
    log(f"  外扩 {buffer_deg}° 后的 Bounding Box:")
    log(f"    North: {bbox_dict['north']:.4f}")
    log(f"    South: {bbox_dict['south']:.4f}")
    log(f"    East:  {bbox_dict['east']:.4f}")
    log(f"    West:  {bbox_dict['west']:.4f}")

    # --- 保存 ---
    save_intermediate(sensors_df, "phase0_sensors", output_dir)
    save_intermediate(bbox_dict, "phase0_bbox", output_dir)
    save_intermediate(raw_bbox_dict, "phase0_raw_bbox", output_dir)

    # 每条高速公路的传感器分布保存为诊断 csv
    fwy_summary = (
        sensors_df.groupby(["Fwy", "Dir", "Type"])
        .agg(count=("ID", "count"),
             lat_min=("Latitude", "min"),
             lat_max=("Latitude", "max"),
             lon_min=("Longitude", "min"),
               lon_max=("Longitude", "max"),
             correction_rate=("shn_correction_applied", "mean"))
        .reset_index()
    )
    save_intermediate(fwy_summary, "phase0_fwy_summary", output_dir)

    log("Phase 0 完成\n")
    return sensors_df, bbox_dict


# =============================================================================
# Phase 1: 提取 OSM 路网, 构建关联矩阵
# =============================================================================
def phase1_extract_network(bbox_dict, output_dir, max_edge_length_m=2000, pbf_path=None):
    """
    从 OSM 提取 motorway + motorway_link + trunk + trunk_link 网络, 构建关联矩阵.
    trunk/trunk_link 覆盖部分在 OSM 中未标为 motorway 但在 PeMS 中有传感器的高速公路.

    对应 R 代码 L12-72, L203-232.

    输出:
        G: osmnx 图对象
        nodes_gdf: 节点 GeoDataFrame
        edges_gdf: 边 GeoDataFrame
        incidence_df: 关联矩阵 DataFrame
        edge_highway_types: {osmid: highway_type} 映射
    """
    log("=" * 70)
    log("Phase 1: 提取 OSM 路网 & 构建关联矩阵")
    log("=" * 70)

# --- Step 1.1: 从本地 pbf 提取高速路网 ---
    log("Step 1.1: 从本地 PBF 文件提取高速路网...")
    log("  (包含 motorway/motorway_link/trunk/trunk_link)")
    import osmium
    import sys

    north = bbox_dict["north"]
    south = bbox_dict["south"]
    east = bbox_dict["east"]
    west = bbox_dict["west"]
    log(f"  区域范围: ({south:.4f},{west:.4f})-({north:.4f},{east:.4f})")

    # PBF 文件路径
    if pbf_path is None:
        pbf_path = os.path.join(os.path.dirname(output_dir), "california-latest.osm.pbf")
    if not os.path.exists(pbf_path):
        raise FileNotFoundError(f"找不到 PBF 文件: {pbf_path}")

    pbf_size_mb = os.path.getsize(pbf_path) / 1024 / 1024
    log(f"  使用本地文件: {pbf_path} ({pbf_size_mb:.0f} MB)")

    # 提取的 highway 类型 (motorway + trunk 覆盖所有高速/快速路)
    HIGHWAY_TYPES = {"motorway", "motorway_link", "trunk", "trunk_link"}

    # --- 第一遍: 收集 bbox 内的 way ---
    class WayCollector(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []
            self.needed_nodes = set()
            self._count = 0
            self._hit = 0
            self._last_print = 0

        def way(self, w):
            self._count += 1
            if self._count - self._last_print >= 500_000:
                self._last_print = self._count
                sys.stdout.write(
                    f"\r  [Pass 1] 已扫描 {self._count:,} ways, "
                    f"命中 {self._hit} 条..."
                )
                sys.stdout.flush()

            hw = w.tags.get("highway", "")
            if hw not in HIGHWAY_TYPES:
                return
            nodes = []
            in_bbox = False
            for n in w.nodes:
                nodes.append(n.ref)
                if (south <= n.location.lat <= north and
                    west <= n.location.lon <= east):
                    in_bbox = True
            if in_bbox:
                self._hit += 1
                self.ways.append({
                    "id": w.id,
                    "tags": dict(w.tags),
                    "node_refs": nodes,
                })
                self.needed_nodes.update(nodes)

    log("  第一遍扫描: 收集 motorway + trunk way...")
    t0 = time.time()
    wc = WayCollector()
    wc.apply_file(pbf_path, locations=True)
    sys.stdout.write("\n")
    log(f"  找到 {len(wc.ways)} 条 way, 涉及 {len(wc.needed_nodes)} 个节点 "
        f"(耗时 {time.time()-t0:.0f}s)")

    # 统计各 highway 类型的数量
    hw_counts = defaultdict(int)
    for w in wc.ways:
        hw_counts[w["tags"].get("highway", "unknown")] += 1
    for hw, cnt in sorted(hw_counts.items()):
        log(f"    {hw}: {cnt} ways")

    # --- 第二遍: 收集节点坐标 ---
    class NodeCollector(osmium.SimpleHandler):
        def __init__(self, needed):
            super().__init__()
            self.needed = needed
            self.coords = {}
            self._count = 0
            self._hit = 0
            self._last_print = 0
            self._total_needed = len(needed)

        def node(self, n):
            self._count += 1
            if self._count - self._last_print >= 2_000_000:
                self._last_print = self._count
                pct = 100 * self._hit / self._total_needed if self._total_needed else 0
                sys.stdout.write(
                    f"\r  [Pass 2] 已扫描 {self._count:,} nodes, "
                    f"已收集 {self._hit}/{self._total_needed} ({pct:.1f}%)"
                )
                sys.stdout.flush()

            if n.id in self.needed:
                self.coords[n.id] = (n.location.lat, n.location.lon)
                self._hit += 1

    log("  第二遍扫描: 收集节点坐标...")
    t0 = time.time()
    nc = NodeCollector(wc.needed_nodes)
    nc.apply_file(pbf_path)
    sys.stdout.write("\n")
    log(f"  获取到 {len(nc.coords)} 个节点坐标 (耗时 {time.time()-t0:.0f}s)")

    # --- 第三遍: 提取 route relation (获取方向和额外 ref) ---
    extracted_way_ids = set(w["id"] for w in wc.ways)

    class RelationCollector(osmium.SimpleHandler):
        """扫描 route=road 类型的 relation, 收集与已提取 way 相关的路线信息."""
        def __init__(self, target_ways):
            super().__init__()
            self.target_ways = target_ways
            self.way_to_rels = defaultdict(list)  # way_id -> [rel_info, ...]
            self._count = 0
            self._hit = 0
            self._last_print = 0

        def relation(self, r):
            self._count += 1
            if self._count - self._last_print >= 100_000:
                self._last_print = self._count
                sys.stdout.write(
                    f"\r  [Pass 3] 已扫描 {self._count:,} relations, "
                    f"命中 {self._hit} 条路线..."
                )
                sys.stdout.flush()

            # 只处理 type=route, route=road 的 relation
            if r.tags.get("type") != "route" or r.tags.get("route") != "road":
                return

            ref = r.tags.get("ref", "")
            direction = r.tags.get("direction", "")
            network = r.tags.get("network", "")
            name = r.tags.get("name", "")

            # 找出属于我们提取的 way 的成员
            member_way_ids = []
            for m in r.members:
                if m.type == "w" and m.ref in self.target_ways:
                    member_way_ids.append(m.ref)

            if not member_way_ids:
                return

            self._hit += 1
            rel_info = {
                "rel_id": r.id,
                "ref": ref,
                "direction": direction,
                "network": network,
                "name": name,
            }
            for wid in member_way_ids:
                self.way_to_rels[wid].append(rel_info)

    log("  第三遍扫描: 提取 route relation (方向 + ref)...")
    t0 = time.time()
    rc = RelationCollector(extracted_way_ids)
    rc.apply_file(pbf_path)
    sys.stdout.write("\n")
    log(f"  找到 {rc._hit} 条相关路线 relation, "
        f"覆盖 {len(rc.way_to_rels)} 条 way (耗时 {time.time()-t0:.0f}s)")

    # 统计有方向标签的 relation
    dir_rels = sum(1 for wid in rc.way_to_rels
                   for r in rc.way_to_rels[wid] if r["direction"])
    log(f"  其中有 direction 标签的 relation-way 对: {dir_rels}")

    # 输出示例
    sample_count = 0
    for wid, rels in list(rc.way_to_rels.items())[:5]:
        for r in rels:
            if r["direction"]:
                log(f"    way {wid}: ref={r['ref']}, "
                    f"dir={r['direction']}, name={r['name'][:50]}")
                sample_count += 1
                if sample_count >= 5:
                    break
        if sample_count >= 5:
            break

    way_to_rels = dict(rc.way_to_rels)

    # --- 写出 .osm XML 文件 ---
    osm_path = os.path.join(output_dir, "d3_motorway.osm")
    log(f"  写出 .osm 文件: {osm_path}")

    with open(osm_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<osm version="0.6" generator="pbf_extract">\n')

        # 写节点
        for nid, (lat, lon) in nc.coords.items():
            f.write(f'  <node id="{nid}" lat="{lat}" lon="{lon}" />\n')

        # 写 way
        for w in wc.ways:
            f.write(f'  <way id="{w["id"]}">\n')
            for nref in w["node_refs"]:
                f.write(f'    <nd ref="{nref}" />\n')
            for k, v in w["tags"].items():
                v_esc = v.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
                f.write(f'    <tag k="{k}" v="{v_esc}" />\n')
            f.write(f'  </way>\n')

        f.write('</osm>\n')

    osm_size = os.path.getsize(osm_path) / 1024 / 1024
    log(f"  .osm 文件大小: {osm_size:.1f} MB")

    # --- 用 osmnx 读取 ---
    log("  用 osmnx 加载图...")
    # 确保 osmnx 保留 unsigned_ref 标签
    if "unsigned_ref" not in ox.settings.useful_tags_way:
        ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + ["unsigned_ref"]

    # Step 1: 不简化加载
    G_raw = ox.graph_from_xml(osm_path, simplify=False)
    log(f"  原始图: {G_raw.number_of_nodes()} 节点, {G_raw.number_of_edges()} 边")

    # Step 2: 自定义简化 - 保留不同 osmid 的边界节点
    G = ox.simplify_graph(G_raw, edge_attrs_differ=["osmid"])
    log(f"  简化后 (保留 osmid 边界): {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # Step 3: 拆分超长边
    log(f"  拆分超长边 (阈值 {max_edge_length_m}m)...")
    G = split_long_edges(G, max_length_m=max_edge_length_m)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    log(f"  最终: {len(nodes_gdf)} 个节点, {len(edges_gdf)} 条边")

    # --- Step 1.2: 统计 highway 类型 ---
    log("Step 1.2: 统计边的 highway 类型...")
    highway_types = edges_gdf["highway"].apply(
        lambda x: x if isinstance(x, str) else str(x)
    ).value_counts()
    for ht, cnt in highway_types.items():
        log(f"    {ht}: {cnt}")

    # --- Step 1.3: 提取每条边的详细信息 ---
    log("Step 1.3: 提取边的标签信息 (ref, name, highway type)...")
    edge_info = []
    edge_highway_map = {}  # edge_key -> highway_type

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_key = f"{u}_{v}_{key}"

        hw_type = data.get("highway", "")
        if isinstance(hw_type, list):
            hw_type = hw_type[0]
        ref = data.get("ref", "")
        if isinstance(ref, list):
            ref = ";".join(ref)
        name = data.get("name", "")
        if isinstance(name, list):
            name = ";".join(name)
        unsigned_ref = data.get("unsigned_ref", "")
        if isinstance(unsigned_ref, list):
            unsigned_ref = ";".join(unsigned_ref)

        # osmid 仅保留用于参考, 不作为 link ID
        osmid = data.get("osmid", "")
        if isinstance(osmid, list):
            osmid_str = ";".join(str(o) for o in osmid)
        else:
            osmid_str = str(osmid)

        edge_highway_map[edge_key] = hw_type
        edge_info.append({
            "edge_key": edge_key,
            "osmid": osmid_str,
            "u": u,
            "v": v,
            "highway": hw_type,
            "ref": ref,
            "unsigned_ref": unsigned_ref,
            "name": name,
            "oneway": data.get("oneway", ""),
            "lanes": data.get("lanes", ""),
        })

    edge_info_df = pd.DataFrame(edge_info)
    log(f"  共 {len(edge_info_df)} 条边 (每条 edge 唯一 edge_key)")

    # ref 标签覆盖率
    # 分类型统计 ref 覆盖率
    for hw_type in ["motorway", "trunk"]:
        hw_edges = edge_info_df[edge_info_df["highway"] == hw_type]
        if len(hw_edges) == 0:
            continue
        has_ref = hw_edges["ref"].apply(lambda x: x != "" and pd.notna(x))
        has_name = hw_edges["name"].apply(lambda x: x != "" and pd.notna(x))
        log(f"  {hw_type} 边: ref 覆盖率 {has_ref.sum()}/{len(hw_edges)} "
            f"({100*has_ref.mean():.1f}%), "
            f"name 覆盖率 {has_name.sum()}/{len(hw_edges)} "
            f"({100*has_name.mean():.1f}%)")
        no_ref_no_name = hw_edges[~has_ref & ~has_name]
        if len(no_ref_no_name) > 0:
            log(f"  WARNING: {len(no_ref_no_name)} 条 {hw_type} 边既无 ref 也无 name!", "WARN")

    # --- Step 1.4: 构建关联矩阵 ---
    log("Step 1.4: 构建关联矩阵 (incidence matrix)...")

    # 收集所有拓扑节点
    topo_nodes = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        topo_nodes.add(u)
        topo_nodes.add(v)

    topo_node_list = sorted(list(topo_nodes))
    node_to_idx = {n: i for i, n in enumerate(topo_node_list)}
    log(f"  拓扑节点数: {len(topo_node_list)}")

    # 收集所有唯一边 (用 edge_key 作为唯一标识)
    link_records = []
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_key = f"{u}_{v}_{key}"
        link_records.append((edge_key, u, v))

    link_ids = [r[0] for r in link_records]
    n_nodes = len(topo_node_list)
    n_links = len(link_records)

    log(f"  构建 {n_nodes} × {n_links} 关联矩阵...")
    mat = np.zeros((n_nodes, n_links), dtype=np.int8)
    links_dict = {}  # edge_key -> (tail, head)

    for j, (lid, u, v) in enumerate(link_records):
        links_dict[lid] = (u, v)
        if u in node_to_idx:
            mat[node_to_idx[u], j] = -1  # tail (起点)
        if v in node_to_idx:
            mat[node_to_idx[v], j] = +1  # head (终点)

    incidence_df = pd.DataFrame(
        mat,
        index=[str(n) for n in topo_node_list],
        columns=link_ids,
    )

    # 验证: 每列应恰好有一个 -1 和一个 +1
    col_sums = incidence_df.sum(axis=0)
    bad_cols = col_sums[col_sums != 0]
    if len(bad_cols) > 0:
        log(f"  WARNING: {len(bad_cols)} 列的 +1/-1 不匹配", "WARN")

    log(f"  关联矩阵形状: {incidence_df.shape}")

    # --- 保存 ---
    save_intermediate(G, "phase1_graph", output_dir)
    save_intermediate(nodes_gdf, "phase1_nodes_gdf", output_dir)
    save_intermediate(edges_gdf, "phase1_edges_gdf", output_dir)
    save_intermediate(incidence_df, "phase1_incidence", output_dir)
    save_intermediate(edge_info_df, "phase1_edge_info", output_dir)
    save_intermediate(edge_highway_map, "phase1_edge_highway_map", output_dir)
    save_intermediate(links_dict, "phase1_links_dict", output_dir)
    save_intermediate(way_to_rels, "phase1_way_relations", output_dir)

    # 输出关联矩阵的前 10 行 5 列作为示例
    log("  关联矩阵示例 (前10行×5列):")
    sample = incidence_df.iloc[:10, :5]
    for idx, row in sample.iterrows():
        vals = "  ".join([f"{v:+d}" if v != 0 else " 0" for v in row])
        log(f"    [{idx}] {vals}")

    log("Phase 1 完成\n")
    return (G, nodes_gdf, edges_gdf, incidence_df, edge_info_df,
            edge_highway_map, links_dict, way_to_rels)


# =============================================================================
# Phase 2: ref 标签归属高速公路 + SHN 方向判定
# =============================================================================
def phase2_assign_freeways_ref_only(
    sensors_df, edge_info_df, incidence_df, nodes_gdf, links_dict,
    way_to_rels, edges_gdf, shn_shapefile_path, output_dir
):
    """
    使用 OSM way ref 标签 + route relation 将主线边分配到高速公路,
    使用 Caltrans SHN Lines 判定方向.

    方向判定仅依赖 SHN: 计算 OSM edge 几何中点到同 Route 最近 SHN 线段的距离,
    读取该 SHN 线段的 Direction 字段. 匹配失败则方向留 None.

    输出:
        fwy_links: {fwy_number: [{id, dir, ref_tag}, ...]}
        ref_coverage_report: 无 ref 的 motorway 边列表
    """
    log("=" * 70)
    log("Phase 2: ref 归属高速公路 + SHN 方向判定")
    log("=" * 70)

    # --- Step 2.1: 从 way ref + relation ref 提取高速公路编号 ---
    log("Step 2.1: 解析 ref 标签 (way 级 + relation 级)...")
    log(f"  特殊映射表: {REF_TO_PEMS_FWY}")

    mainline_edges = edge_info_df[
        edge_info_df["highway"].isin(["motorway", "trunk"])
    ].copy()
    log(f"  主线边 (motorway + trunk): {len(mainline_edges)}")

    pems_fwy_numbers = set(int(f) for f in sensors_df["Fwy"].dropna().unique())
    log(f"  PeMS 中的高速编号: {sorted(pems_fwy_numbers)}")

    # --- 辅助函数: 从 ref 字符串解析 PeMS fwy 编号 ---
    def ref_to_fwy_num(ref_str):
        """将一个 ref 字符串转为 PeMS fwy 编号, 优先查特殊映射."""
        ref_str = ref_str.strip()
        if ref_str in REF_TO_PEMS_FWY:
            return REF_TO_PEMS_FWY[ref_str]
        numbers = re.findall(r"\d+", ref_str)
        return int(numbers[0]) if numbers else None

    # --- 辅助函数: 获取 edge 的所有 relation ---
    def get_edge_relations(osmid_str):
        """通过 osmid 查找 edge 所属的所有 route relation."""
        rels = []
        for oid_str in str(osmid_str).split(";"):
            oid_str = oid_str.strip()
            try:
                oid = int(oid_str)
            except ValueError:
                continue
            rels.extend(way_to_rels.get(oid, []))
        return rels

    # --- 为每条主线边收集所有 ref 来源, 确定 fwy 编号 ---
    edge_to_fwy_nums = defaultdict(set)    # edge_key -> {fwy_num, ...}
    edge_to_refs = defaultdict(list)        # edge_key -> [ref_str, ...]
    edge_relations = {}                     # edge_key -> [rel_info, ...]
    no_ref_links = []
    ambiguous_ref_links = []

    for _, row in mainline_edges.iterrows():
        ek = row["edge_key"]
        all_refs = []

        # 来源 1: way 级 ref 标签
        way_ref = str(row["ref"]).strip() if pd.notna(row["ref"]) else ""
        if way_ref:
            all_refs.extend([r.strip() for r in way_ref.split(";")])

        # 来源 2: relation 级 ref (通过 osmid 查找)
        rels = get_edge_relations(row["osmid"])
        edge_relations[ek] = rels
        for rel in rels:
            if rel["ref"]:
                rel_ref = rel["ref"].strip()
                if rel_ref not in all_refs:
                    all_refs.append(rel_ref)

        # 来源 3: way 级 unsigned_ref 标签 (如 unsigned_ref="CA 51")
        u_ref = str(row["unsigned_ref"]).strip() if pd.notna(row.get("unsigned_ref")) else ""
        if u_ref:
            for r in u_ref.split(";"):
                r = r.strip()
                if r and r not in all_refs:
                    all_refs.append(r)

        if not all_refs:
            no_ref_links.append({
                "edge_key": ek,
                "name": row["name"],
                "u": row["u"],
                "v": row["v"],
            })
            continue

        edge_to_refs[ek] = all_refs
        if len(all_refs) > 1:
            ambiguous_ref_links.append({
                "edge_key": ek,
                "refs": ";".join(all_refs),
                "name": row["name"],
            })

        # 从所有 ref 解析 fwy 编号
        for ref_str in all_refs:
            fwy_num = ref_to_fwy_num(ref_str)
            if fwy_num is not None:
                edge_to_fwy_nums[ek].add(fwy_num)

        if not edge_to_fwy_nums[ek]:
            no_ref_links.append({
                "edge_key": ek,
                "refs": ";".join(all_refs),
                "name": row["name"],
                "u": row["u"],
                "v": row["v"],
            })

    # 按高速公路编号分组 (一条 edge 可以属于多条高速)
    fwy_groups = defaultdict(list)
    for ek, fwy_nums in edge_to_fwy_nums.items():
        for fwy_num in fwy_nums:
            fwy_groups[fwy_num].append({
                "id": ek,
                "dir": None,
                "ref": ";".join(edge_to_refs.get(ek, [])),
            })

    log(f"  通过 ref 识别了 {len(edge_to_fwy_nums)} 条主线边")
    log(f"  归属到 {len(fwy_groups)} 条高速公路: {sorted(fwy_groups.keys())}")
    log(f"  无 ref 的主线边: {len(no_ref_links)}")
    log(f"  ref 含多条路的边: {len(ambiguous_ref_links)}")

    # 统计 ref 来源
    from_way_only = 0
    from_rel_only = 0
    from_both = 0
    for ek in edge_to_fwy_nums:
        row = mainline_edges[mainline_edges["edge_key"] == ek].iloc[0]
        has_way_ref = bool(str(row["ref"]).strip()) if pd.notna(row["ref"]) else False
        has_rel_ref = any(r["ref"] for r in edge_relations.get(ek, []))
        if has_way_ref and has_rel_ref:
            from_both += 1
        elif has_way_ref:
            from_way_only += 1
        elif has_rel_ref:
            from_rel_only += 1
    log(f"  ref 来源: way only={from_way_only}, relation only={from_rel_only}, both={from_both}")

    # --- Step 2.2: SHN 空间匹配 (方向判定 + 补充归属) ---
    log("\nStep 2.2: SHN 空间匹配 (方向判定 + 补充归属)...")
    log(f"  SHN shapefile: {shn_shapefile_path}")

    # 加载 SHN Lines
    shn_gdf = gpd.read_file(shn_shapefile_path)
    shn_gdf = shn_gdf.to_crs(epsg=4326)  # 转 WGS84 与 OSM 一致
    log(f"  SHN 全州记录数: {len(shn_gdf)}")

    # 按 Route 建索引 (覆盖所有 PeMS 高速编号, 而非仅 ref 识别到的)
    shn_by_route = {}
    for route_num in pems_fwy_numbers:
        route_lines = shn_gdf[shn_gdf["Route"] == route_num].copy()
        if len(route_lines) > 0:
            shn_by_route[route_num] = route_lines
    log(f"  SHN 中匹配到的 PeMS 高速: {sorted(shn_by_route.keys())}")
    missing_shn = pems_fwy_numbers - set(shn_by_route.keys())
    if missing_shn:
        log(f"  WARNING: 以下高速无 SHN 数据: {sorted(missing_shn)}", "WARN")

    # 构建 edge_key -> geometry 映射 (所有主线边)
    edge_geom = {}
    for (u, v, key), row in edges_gdf.iterrows():
        ek = f"{u}_{v}_{key}"
        edge_geom[ek] = row.geometry

    # 记录 ref 阶段每条 edge 已归入的 fwy 集合
    edge_in_fwy = defaultdict(set)  # edge_key -> {fwy_num, ...}
    for fwy_num, links in fwy_groups.items():
        for link_info in links:
            edge_in_fwy[link_info["id"]].add(fwy_num)

    # SHN 匹配阈值 (度): 0.0003° ≈ 33m, 覆盖分离式高速的车道间距
    SHN_MATCH_THRESHOLD = 0.0003
    log(f"  SHN 匹配阈值: {SHN_MATCH_THRESHOLD}° ≈ {SHN_MATCH_THRESHOLD * 111000:.0f}m")

    # 对每条主线边, 匹配所有 PeMS 高速的 SHN 线段
    edge_shn_dir = defaultdict(dict)  # edge_key -> {fwy_num -> pems_dir}
    shn_supplement_count = 0

    for _, row in mainline_edges.iterrows():
        ek = row["edge_key"]
        geom = edge_geom.get(ek)
        if geom is None or geom.is_empty:
            continue

        centroid = geom.interpolate(0.5, normalized=True)

        for route_num, route_shn in shn_by_route.items():
            dists = route_shn.geometry.distance(centroid)
            nearest_idx = dists.idxmin()
            nearest_dist = dists[nearest_idx]

            if nearest_dist > SHN_MATCH_THRESHOLD:
                continue

            # 读取方向
            shn_dir = route_shn.loc[nearest_idx, "Direction"]
            pems_dir = SHN_DIR_TO_PEMS.get(shn_dir)
            if not pems_dir:
                continue

            # 记录该 edge 在该 route 上的方向
            edge_shn_dir[ek][route_num] = pems_dir

            # 如果 ref 未将此 edge 归入该 route, 则补充归入
            if route_num not in edge_in_fwy[ek]:
                fwy_groups[route_num].append({
                    "id": ek,
                    "dir": pems_dir,
                    "ref": f"SHN:{route_num}",
                })
                edge_in_fwy[ek].add(route_num)
                shn_supplement_count += 1

    log(f"  SHN 补充归属: {shn_supplement_count} 条 edge 被补充到高速分组")

    # 为 fwy_groups 中所有 edge 填充方向 (ref 阶段的 dir 为 None)
    dir_stats = {"from_shn": 0, "no_geometry": 0, "shn_match_failed": 0}

    for fwy_num, links in fwy_groups.items():
        for link_info in links:
            if link_info["dir"] is not None:
                # SHN 补充时已设置方向
                dir_stats["from_shn"] += 1
                continue

            ek = link_info["id"]
            shn_dir = edge_shn_dir.get(ek, {}).get(fwy_num)
            if shn_dir:
                link_info["dir"] = shn_dir
                dir_stats["from_shn"] += 1
            else:
                # 该 edge 不在 SHN 匹配范围内
                geom = edge_geom.get(ek)
                if geom is None or geom.is_empty:
                    dir_stats["no_geometry"] += 1
                else:
                    dir_stats["shn_match_failed"] += 1

    log(f"  方向判定: SHN={dir_stats['from_shn']}, "
        f"无几何={dir_stats['no_geometry']}, "
        f"SHN匹配失败={dir_stats['shn_match_failed']}")

    # 过滤: 只保留 PeMS 中存在的高速编号
    extra_fwy = set(fwy_groups.keys()) - pems_fwy_numbers
    if extra_fwy:
        log(f"  OSM 中有但 PeMS 中无的高速编号 (将过滤): {sorted(extra_fwy)}")
        for f in extra_fwy:
            del fwy_groups[f]

    missing_fwy = pems_fwy_numbers - set(fwy_groups.keys())
    if missing_fwy:
        log(f"  WARNING: PeMS 中有但未找到的高速编号: {sorted(missing_fwy)}", "WARN")

    log(f"  最终保留 {len(fwy_groups)} 条高速公路")

    # 各高速各方向的 link 数 (区分来源)
    for fwy_num in sorted(fwy_groups.keys()):
        dir_counts = defaultdict(int)
        ref_count = 0
        shn_count = 0
        for l in fwy_groups[fwy_num]:
            d = l["dir"] if l["dir"] else "unknown"
            dir_counts[d] += 1
            if l.get("ref", "").startswith("SHN:"):
                shn_count += 1
            else:
                ref_count += 1
        log(f"    Fwy {fwy_num}: {dict(dir_counts)} (ref={ref_count}, shn补充={shn_count})")

    # --- 保存 ---
    fwy_links = dict(fwy_groups)
    save_intermediate(fwy_links, "phase2_fwy_links", output_dir)

    no_ref_df = pd.DataFrame(no_ref_links) if no_ref_links else pd.DataFrame()
    save_intermediate(no_ref_df, "phase2_no_ref_links", output_dir)

    ambig_df = pd.DataFrame(ambiguous_ref_links) if ambiguous_ref_links else pd.DataFrame()
    save_intermediate(ambig_df, "phase2_ambiguous_ref", output_dir)

    if len(no_ref_links) > 0:
        log(f"\n  === 无 ref 标签的主线边 (需要检查) ===")
        for item in no_ref_links[:20]:
            log(f"    edge_key={item['edge_key']}, name={item.get('name', 'N/A')}")
        if len(no_ref_links) > 20:
            log(f"    ... 共 {len(no_ref_links)} 条, 完整列表见 phase2_no_ref_links.csv")

    log("Phase 2 完成\n")
    return fwy_links, no_ref_df


# =============================================================================
# Phase 3: ML (主线) 传感器匹配
# =============================================================================
def phase3_match_ml_sensors(
    sensors_df, fwy_links, incidence_df, nodes_gdf, edges_gdf, output_dir,
    threshold=1e-4
):
    """
    将 ML 传感器匹配到 motorway link (点到几何线距离算法).

    计算传感器坐标点到 edge 几何线的最短距离 (度单位),
    取同高速同方向中距离最小的 edge 作为匹配结果.

    对应 R 代码 L236-283.
    """
    log("=" * 70)
    log("Phase 3: ML (主线) 传感器匹配")
    log("=" * 70)

    ml_sensors = sensors_df[sensors_df["Type"] == "ML"].copy()
    log(f"  ML 传感器总数: {len(ml_sensors)}")

    # 构建 edge_key -> geometry 映射
    edge_geom = {}
    for (u, v, key), row in edges_gdf.iterrows():
        ek = f"{u}_{v}_{key}"
        if row.geometry and not row.geometry.is_empty:
            edge_geom[ek] = row.geometry
        else:
            # 无几何属性时用端点构造直线
            try:
                u_node = nodes_gdf.loc[u]
                v_node = nodes_gdf.loc[v]
                edge_geom[ek] = LineString([
                    (u_node["x"], u_node["y"]),
                    (v_node["x"], v_node["y"]),
                ])
            except KeyError:
                pass

    log(f"  edge 几何缓存: {len(edge_geom)} 条")

    ml_mapping = []  # [{sensor_id, link_id, dist, fwy, dir}, ...]
    ml_unmatched = []

    for idx, sensor in ml_sensors.iterrows():
        sid = int(sensor["ID"])
        slat = sensor["Latitude"]
        slon = sensor["Longitude"]
        sfwy = sensor["Fwy"]
        sdir = sensor["Dir"]

        sensor_pt = Point(slon, slat)  # shapely: (x=lon, y=lat)

        # 获取该高速该方向上的所有 link
        links = fwy_links.get(sfwy, [])
        candidates = [l for l in links if l["dir"] == sdir]

        best_dist = float("inf")
        best_link = None

        for link_info in candidates:
            lid = link_info["id"]
            geom = edge_geom.get(lid)
            if geom is None:
                continue

            dist = sensor_pt.distance(geom)

            if dist < best_dist:
                best_dist = dist
                best_link = lid

        if best_link is not None and best_dist < threshold:
            ml_mapping.append({
                "sensor_id": sid,
                "link_id": best_link,
                "dist": best_dist,
                "fwy": sfwy,
                "dir": sdir,
                "lat": slat,
                "lon": slon,
            })
        else:
            ml_unmatched.append({
                "sensor_id": sid,
                "fwy": sfwy,
                "dir": sdir,
                "lat": slat,
                "lon": slon,
                "best_dist": best_dist,
                "best_link": best_link,
                "reason": "no_candidates" if not candidates else "dist_exceeds_threshold",
                "n_candidates": len(candidates),
            })

    ml_mapping_df = pd.DataFrame(ml_mapping)
    ml_unmatched_df = pd.DataFrame(ml_unmatched)

    log(f"\n  === ML 匹配结果 ===")
    log(f"  匹配成功: {len(ml_mapping_df)}/{len(ml_sensors)} "
        f"({100*len(ml_mapping_df)/max(1,len(ml_sensors)):.1f}%)")
    log(f"  未匹配:   {len(ml_unmatched_df)}")

    if len(ml_mapping_df) > 0:
        log(f"  dist 统计 (度):")
        log(f"    mean: {ml_mapping_df['dist'].mean():.6f}")
        log(f"    max:  {ml_mapping_df['dist'].max():.6f}")
        log(f"    min:  {ml_mapping_df['dist'].min():.6f}")

    # 按高速分组统计
    log(f"\n  各高速匹配情况:")
    ml_fwy_groups = ml_sensors.groupby("Fwy")["ID"].count()
    if len(ml_mapping_df) > 0:
        matched_fwy = ml_mapping_df.groupby("fwy")["sensor_id"].count()
    else:
        matched_fwy = pd.Series(dtype=int)

    for fwy_num in sorted(ml_fwy_groups.index):
        total = ml_fwy_groups[fwy_num]
        matched = matched_fwy.get(fwy_num, 0)
        log(f"    Fwy {fwy_num}: {matched}/{total}")

    # 未匹配原因分布
    if len(ml_unmatched_df) > 0:
        log(f"\n  未匹配原因:")
        reasons = ml_unmatched_df["reason"].value_counts()
        for r, c in reasons.items():
            log(f"    {r}: {c}")

    # --- 保存 ---
    save_intermediate(ml_mapping_df, "phase3_ml_mapping", output_dir)
    save_intermediate(ml_unmatched_df, "phase3_ml_unmatched", output_dir)

    log("Phase 3 完成\n")
    return ml_mapping_df, ml_unmatched_df


# =============================================================================
# Phase 4: 添加匝道 + OR/FR 匹配
# =============================================================================
def phase4_ramp_matching(
    sensors_df, fwy_links, incidence_df, nodes_gdf, edge_highway_map,
    output_dir, coord_threshold=0.01
):
    """
    1. 识别主线节点上的 motorway_link (匝道)
    2. 区分 OR (on-ramp) 和 FR (off-ramp)
    3. 匹配 OR/FR 传感器到匝道 link

    对应 R 代码 L155-982.
    """
    log("=" * 70)
    log("Phase 4: 匝道识别 + OR/FR 传感器匹配")
    log("=" * 70)

    or_sensors = sensors_df[sensors_df["Type"] == "OR"]
    fr_sensors = sensors_df[sensors_df["Type"] == "FR"]
    log(f"  OR 传感器总数: {len(or_sensors)}")
    log(f"  FR 传感器总数: {len(fr_sensors)}")

    # --- Step 4.1: 识别匝道 link IDs (motorway_link + trunk_link) ---
    log("\nStep 4.1: 识别匝道 (motorway_link + trunk_link) ...")
    ramp_link_ids = set()
    for lid, hw_type in edge_highway_map.items():
        hw_str = str(hw_type)
        if "motorway_link" in hw_str or "trunk_link" in hw_str:
            ramp_link_ids.add(lid)
    log(f"  匝道 link 总数: {len(ramp_link_ids)}")

    # --- Step 4.2: 为每条高速找到与主线相连的匝道 ---
    log("Step 4.2: 为每条高速识别匝道 (OR/FR) ...")

    # 结构: {fwy: {"OR": [{node, link, lat, lon, dir}, ...], "FR": [...]}}
    ramp_data = {}

    for fwy_num, links in fwy_links.items():
        or_ramps = []
        fr_ramps = []

        mainline_ids = {l["id"] for l in links}

        for link_info in links:
            lid = link_info["id"]
            if lid not in incidence_df.columns:
                continue

            col = incidence_df[lid]
            # 获取该 link 的所有节点
            connected_nodes = col[col != 0].index.tolist()

            for node_str in connected_nodes:
                row = incidence_df.loc[node_str]
                connected_links = row[row != 0].index.tolist()

                for cl in connected_links:
                    if cl in mainline_ids or cl not in ramp_link_ids:
                        continue

                    try:
                        node_coord = nodes_gdf.loc[int(node_str)]
                    except KeyError:
                        continue

                    # 关联矩阵中该节点对该匝道 link 的值:
                    # +1 = 该节点是该 link 的 head (流入该节点) → FR (从主线驶出)
                    # -1 = 该节点是该 link 的 tail (从该节点流出) → OR (驶入匝道)
                    # 注意: 这里的语义是从主线节点角度看
                    val = int(incidence_df.loc[node_str, cl])

                    ramp_info = {
                        "node_id": node_str,
                        "link_id": cl,
                        "lat": node_coord["y"],
                        "lon": node_coord["x"],
                        "dir": link_info["dir"],
                        "incidence_val": val,
                    }

                    # -1: 从该节点出发 → 车辆驶出主线 → off-ramp (FR)
                    # +1: 到达该节点 → 车辆驶入主线 → on-ramp (OR)
                    # 但要注意方向约定: 在 R 代码中 L520-528:
                    # plus_ramp → OR, minus_ramp → FR
                    # +1 在关联矩阵中表示 head, 即匝道终点在该节点 → 车从匝道来 → OR
                    # -1 表示 tail, 即匝道起点在该节点 → 车去往匝道 → FR
                    if val == +1:
                        or_ramps.append(ramp_info)
                    elif val == -1:
                        fr_ramps.append(ramp_info)

        # 去重
        or_seen = set()
        or_unique = []
        for r in or_ramps:
            if r["link_id"] not in or_seen:
                or_seen.add(r["link_id"])
                or_unique.append(r)

        fr_seen = set()
        fr_unique = []
        for r in fr_ramps:
            if r["link_id"] not in fr_seen:
                fr_seen.add(r["link_id"])
                fr_unique.append(r)

        ramp_data[fwy_num] = {"OR": or_unique, "FR": fr_unique}
        log(f"    Fwy {fwy_num}: OR匝道 {len(or_unique)}, FR匝道 {len(fr_unique)}")

    # --- Step 4.3: OR/FR 传感器匹配 (最近距离) ---
    log("\nStep 4.3: OR/FR 传感器匹配...")

    or_mapping = []
    or_unmatched = []
    fr_mapping = []
    fr_unmatched = []

    def match_ramp_sensors(sensor_df, ramp_type, mapping_list, unmatched_list):
        for _, sensor in sensor_df.iterrows():
            sid = int(sensor["ID"])
            slat = sensor["Latitude"]
            slon = sensor["Longitude"]
            sfwy = sensor["Fwy"]
            sdir = sensor["Dir"]

            ramps = ramp_data.get(sfwy, {}).get(ramp_type, [])
            dir_ramps = [r for r in ramps if r["dir"] == sdir]

            if not dir_ramps:
                unmatched_list.append({
                    "sensor_id": sid, "fwy": sfwy, "dir": sdir,
                    "lat": slat, "lon": slon,
                    "reason": "no_ramps_in_direction",
                    "n_ramps_total": len(ramps),
                })
                continue

            best_dist = float("inf")
            best_link = None
            best_node = None

            for ramp in dir_ramps:
                dist = haversine_deg(slat, slon, ramp["lat"], ramp["lon"])
                if dist < best_dist:
                    best_dist = dist
                    best_link = ramp["link_id"]
                    best_node = ramp["node_id"]

            if best_dist < coord_threshold:
                mapping_list.append({
                    "sensor_id": sid, "link_id": best_link,
                    "node_id": best_node, "dist": best_dist,
                    "fwy": sfwy, "dir": sdir, "lat": slat, "lon": slon,
                })
            else:
                unmatched_list.append({
                    "sensor_id": sid, "fwy": sfwy, "dir": sdir,
                    "lat": slat, "lon": slon,
                    "best_dist": best_dist, "best_link": best_link,
                    "reason": "distance_exceeds_threshold",
                    "n_candidates": len(dir_ramps),
                })

    match_ramp_sensors(or_sensors, "OR", or_mapping, or_unmatched)
    match_ramp_sensors(fr_sensors, "FR", fr_mapping, fr_unmatched)

    or_mapping_df = pd.DataFrame(or_mapping)
    or_unmatched_df = pd.DataFrame(or_unmatched)
    fr_mapping_df = pd.DataFrame(fr_mapping)
    fr_unmatched_df = pd.DataFrame(fr_unmatched)

    log(f"\n  === OR 匹配结果 ===")
    log(f"  匹配成功: {len(or_mapping_df)}/{len(or_sensors)} "
        f"({100*len(or_mapping_df)/max(1,len(or_sensors)):.1f}%)")
    log(f"  未匹配:   {len(or_unmatched_df)}")

    log(f"\n  === FR 匹配结果 ===")
    log(f"  匹配成功: {len(fr_mapping_df)}/{len(fr_sensors)} "
        f"({100*len(fr_mapping_df)/max(1,len(fr_sensors)):.1f}%)")
    log(f"  未匹配:   {len(fr_unmatched_df)}")

    # 未匹配原因
    for label, udf in [("OR", or_unmatched_df), ("FR", fr_unmatched_df)]:
        if len(udf) > 0:
            reasons = udf["reason"].value_counts()
            log(f"  {label} 未匹配原因: {dict(reasons)}")

    # --- 保存 ---
    save_intermediate(ramp_data, "phase4_ramp_data", output_dir)
    save_intermediate(or_mapping_df, "phase4_or_mapping", output_dir)
    save_intermediate(or_unmatched_df, "phase4_or_unmatched", output_dir)
    save_intermediate(fr_mapping_df, "phase4_fr_mapping", output_dir)
    save_intermediate(fr_unmatched_df, "phase4_fr_unmatched", output_dir)

    log("Phase 4 完成\n")
    return or_mapping_df, or_unmatched_df, fr_mapping_df, fr_unmatched_df, ramp_data


# =============================================================================
# Phase 5: FF 传感器标记
# =============================================================================
def phase5_ff_sensors(sensors_df, output_dir):
    """
    FF (Freeway-Freeway) 传感器需要手动匹配, 这里输出待匹配列表.
    """
    log("=" * 70)
    log("Phase 5: FF 传感器标记")
    log("=" * 70)

    ff_sensors = sensors_df[sensors_df["Type"] == "FF"].copy()
    log(f"  FF 传感器总数: {len(ff_sensors)}")

    if len(ff_sensors) == 0:
        log("  无 FF 传感器, 跳过")
        ff_df = pd.DataFrame()
        save_intermediate(ff_df, "phase5_ff_to_manual", output_dir)
        log("Phase 5 完成\n")
        return ff_df

    ff_list = []
    for _, sensor in ff_sensors.iterrows():
        ff_list.append({
            "sensor_id": int(sensor["ID"]),
            "fwy": sensor["Fwy"],
            "dir": sensor["Dir"],
            "lat": sensor["Latitude"],
            "lon": sensor["Longitude"],
            "name": sensor.get("Name", ""),
            "status": "pending_manual",
            "assigned_link": "",
        })

    ff_df = pd.DataFrame(ff_list)

    # 各高速的 FF 传感器数
    log("  各高速 FF 传感器:")
    for fwy, group in ff_df.groupby("fwy"):
        dirs = group["dir"].value_counts().to_dict()
        log(f"    Fwy {fwy}: {len(group)} 个, 方向 {dirs}")

    save_intermediate(ff_df, "phase5_ff_to_manual", output_dir)

    log(f"\n  FF 传感器列表已输出到 phase5_ff_to_manual.csv")
    log(f"  请手动填写 assigned_link 列后用于后续分析")

    log("Phase 5 完成\n")
    return ff_df


# =============================================================================
# Phase 6: 构建邻接矩阵 + 最终汇总
# =============================================================================
def phase6_final_summary(
    incidence_df, ml_mapping_df, or_mapping_df, fr_mapping_df, ff_df,
    sensors_df, nodes_gdf, edges_gdf, output_dir
):
    """
    构建邻接矩阵, 生成可视化地图, 输出最终汇总.
    """
    log("=" * 70)
    log("Phase 6: 构建邻接矩阵 + 最终汇总")
    log("=" * 70)

    # --- Step 6.1: 构建邻接矩阵 ---
    log("Step 6.1: 从关联矩阵构建邻接矩阵...")

    node_ids = list(incidence_df.index)
    n = len(node_ids)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    adj = np.zeros((n, n), dtype=np.int8)
    for col_name in incidence_df.columns:
        col = incidence_df[col_name]
        tails = col[col == -1].index.tolist()
        heads = col[col == +1].index.tolist()
        for t in tails:
            for h in heads:
                ti, hi = node_to_idx[t], node_to_idx[h]
                adj[ti][hi] = 1
                adj[hi][ti] = 1

    adjacency_df = pd.DataFrame(adj, index=node_ids, columns=node_ids)
    log(f"  邻接矩阵形状: {adjacency_df.shape}")
    log(f"  非零元素: {np.count_nonzero(adj)} ({100*np.count_nonzero(adj)/adj.size:.4f}%)")

    save_intermediate(adjacency_df, "phase6_adjacency", output_dir)

    # --- Step 6.2: 合并所有匹配结果 ---
    log("\nStep 6.2: 合并匹配结果...")

    all_records = []
    if len(ml_mapping_df) > 0:
        for _, r in ml_mapping_df.iterrows():
            all_records.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": "ML",
                "fwy": r["fwy"],
                "dir": r["dir"],
            })

    if len(or_mapping_df) > 0:
        for _, r in or_mapping_df.iterrows():
            all_records.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": "OR",
                "fwy": r["fwy"],
                "dir": r["dir"],
            })

    if len(fr_mapping_df) > 0:
        for _, r in fr_mapping_df.iterrows():
            all_records.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": "FR",
                "fwy": r["fwy"],
                "dir": r["dir"],
            })

    all_mapping_df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    save_intermediate(all_mapping_df, "phase6_all_mapping", output_dir)

# --- Step 6.3: 生成交互式地图 ---
    log("\nStep 6.3: 生成交互式地图...")

    center_lat = sensors_df["Latitude"].mean()
    center_lon = sensors_df["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # --- 构建辅助查询结构 ---

    # sensor_id -> 匹配信息 (link_id, dist, type)
    matched_info = {}
    if len(ml_mapping_df) > 0:
        for _, r in ml_mapping_df.iterrows():
            matched_info[int(r["sensor_id"])] = {
                "link_id": r["link_id"], "dist": r.get("dist", None),
                "type": "ML", "fwy": r["fwy"], "dir": r["dir"],
            }
    if len(or_mapping_df) > 0:
        for _, r in or_mapping_df.iterrows():
            matched_info[int(r["sensor_id"])] = {
                "link_id": r["link_id"], "dist": r.get("dist", None),
                "type": "OR", "fwy": r["fwy"], "dir": r["dir"],
            }
    if len(fr_mapping_df) > 0:
        for _, r in fr_mapping_df.iterrows():
            matched_info[int(r["sensor_id"])] = {
                "link_id": r["link_id"], "dist": r.get("dist", None),
                "type": "FR", "fwy": r["fwy"], "dir": r["dir"],
            }

    # sensor_id -> 未匹配原因
    unmatched_info = {}
    for phase_file in ["phase3_ml_unmatched", "phase4_or_unmatched", "phase4_fr_unmatched"]:
        try:
            udf = load_intermediate(phase_file, output_dir)
            if udf is not None and len(udf) > 0:
                for _, r in udf.iterrows():
                    unmatched_info[int(r["sensor_id"])] = {
                        "reason": r.get("reason", "unknown"),
                        "best_dist": r.get("best_dist", None),
                        "best_link": r.get("best_link", None),
                        "n_candidates": r.get("n_candidates", r.get("n_ramps_total", None)),
                    }
        except Exception:
            pass

    # --- 构建 edge → fwy/dir 查询 (从 Phase 2 结果) ---
    try:
        fwy_links = load_intermediate("phase2_fwy_links", output_dir)
    except Exception:
        fwy_links = {}
    edge_fwy_info = defaultdict(list)  # edge_key -> [(fwy, dir, source), ...]
    for fwy_num, links in fwy_links.items():
        for l in links:
            src = "SHN" if l.get("ref", "").startswith("SHN:") else "ref"
            edge_fwy_info[l["id"]].append({
                "fwy": fwy_num,
                "dir": l["dir"],
                "source": src,
            })

    # edge_key -> edge geometry (用于高亮匹配路段, edge_key 保证唯一)
    link_geom = {}
    link_nodes = {}  # edge_key -> (u, v)
    for (u, v, key), edge in edges_gdf.iterrows():
        edge_key = f"{u}_{v}_{key}"
        if edge.geometry and edge.geometry.geom_type == "LineString":
            link_geom[edge_key] = edge.geometry
        link_nodes[edge_key] = (u, v)

    # 计算 link 长度 (米)
    def link_length_m(geom):
        if geom is None:
            return 0
        coords = list(geom.coords)
        total = 0
        for i in range(len(coords) - 1):
            total += haversine_deg(coords[i][1], coords[i][0],
                                   coords[i+1][1], coords[i+1][0])
        return total * 111320  # 粗略转米

    # 已被传感器匹配的 edge
    sensor_matched_edges = set()
    for sid, info in matched_info.items():
        sensor_matched_edges.add(str(info["link_id"]))

    # --- 绘制路网边 (分三层: 未归属 / 已归属未匹配 / 传感器匹配) ---

    # 层 1: 未归属到任何高速的主线边 (暗黄色)
    fg_unassigned = folium.FeatureGroup(name="未归属主线边 (黄)", show=True)
    # 层 2: 已归属但无传感器匹配的边 (粉色)
    fg_assigned = folium.FeatureGroup(name="已归属未匹配 (粉)", show=True)
    # 层 3: 传感器匹配到的边 (蓝/绿/红)
    fg_highlight = folium.FeatureGroup(name="传感器匹配路段", show=True)
    highlight_colors = {"ML": "blue", "OR": "green", "FR": "red"}

    # 收集所有节点坐标 (用于绘制分界点)
    junction_nodes = set()

    for (u, v, key), edge in edges_gdf.iterrows():
        if not (edge.geometry and edge.geometry.geom_type == "LineString"):
            continue
        ek = f"{u}_{v}_{key}"
        coords = [(c[1], c[0]) for c in edge.geometry.coords]
        length_m = link_length_m(edge.geometry)
        fwy_info = edge_fwy_info.get(ek, [])

        if fwy_info:
            fwy_str = ", ".join(
                f"Fwy{f['fwy']}-{f['dir'] or '?'}({f['source']})"
                for f in fwy_info
            )
        else:
            fwy_str = "未归属"

        popup_html = (
            f"<b>Edge:</b> {ek}<br>"
            f"<b>长度:</b> {length_m:.0f}m<br>"
            f"<b>归属:</b> {fwy_str}<br>"
        )

        if ek in sensor_matched_edges:
            # 层 3: 传感器匹配
            # 找匹配类型
            match_type = "ML"
            for sid, info in matched_info.items():
                if str(info["link_id"]) == ek:
                    match_type = info["type"]
                    break
            color = highlight_colors.get(match_type, "cyan")
            folium.PolyLine(
                coords, weight=5, color=color, opacity=0.8,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{ek} | {fwy_str} | {match_type}",
            ).add_to(fg_highlight)
        elif fwy_info:
            # 层 2: 已归属未匹配 (粉色)
            folium.PolyLine(
                coords, weight=4, color="hotpink", opacity=0.5,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{ek} | {fwy_str}",
            ).add_to(fg_assigned)
        else:
            # 层 1: 未归属 (暗黄色)
            folium.PolyLine(
                coords, weight=3, color="#CC9900", opacity=0.4,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{ek} | 未归属",
            ).add_to(fg_unassigned)

        junction_nodes.add(u)
        junction_nodes.add(v)

    fg_unassigned.add_to(m)
    fg_assigned.add_to(m)
    fg_highlight.add_to(m)

    # --- 绘制节点分界点 (黑边小圆) ---
    fg_nodes = folium.FeatureGroup(name="节点分界点", show=False)
    for nid in junction_nodes:
        try:
            node = nodes_gdf.loc[nid]
        except KeyError:
            continue
        nlat, nlon = node["y"], node["x"]
        folium.CircleMarker(
            location=[nlat, nlon], radius=3,
            color="black", weight=1.5,
            fill=True, fill_color="white", fill_opacity=0.8,
            tooltip=f"node {nid}",
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)

    # --- 绘制传感器点 ---
    colors = {"ML": "blue", "OR": "green", "FR": "red", "FF": "orange", "HV": "purple"}

    for sensor_type, color in colors.items():
        type_df = sensors_df[sensors_df["Type"] == sensor_type]
        fg = folium.FeatureGroup(
            name=f"{sensor_type} ({len(type_df)})", show=(sensor_type == "ML")
        )
        for _, s in type_df.iterrows():
            sid = int(s["ID"])
            slat, slon = s["Latitude"], s["Longitude"]

            if sid in matched_info:
                # === 已匹配 ===
                info = matched_info[sid]
                lid = str(info["link_id"])
                geom = link_geom.get(lid)
                length_m = link_length_m(geom) if geom else 0
                dist_val = info.get("dist", None)
                dist_str = f"{dist_val:.6f}" if dist_val is not None else "N/A"

                popup = (
                    f"<b>ID: {sid}</b> ✅ 已匹配<br>"
                    f"Type: {s['Type']} | Fwy: {s.get('Fwy','')} {s.get('Dir','')}<br>"
                    f"Name: {s.get('Name','')}<br>"
                    f"坐标: ({slat:.6f}, {slon:.6f})<br>"
                    f"<hr>"
                    f"<b>匹配路段:</b> {lid}<br>"
                    f"路段长度: {length_m:.1f} m<br>"
                    f"匹配距离 (dist): {dist_str}<br>"
                )
                folium.CircleMarker(
                    location=[slat, slon], radius=5,
                    color=color, fill=True, fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=350),
                    tooltip=f"{sid} ✓ link={lid}",
                ).add_to(fg)

            elif sid in unmatched_info:
                # === 未匹配 (有诊断信息) ===
                uinfo = unmatched_info[sid]
                reason = uinfo["reason"]
                best_dist_val = uinfo.get("best_dist")
                best_link = uinfo.get("best_link")
                n_cand = uinfo.get("n_candidates")
                dist_str = f"{best_dist_val:.6f}" if best_dist_val is not None and best_dist_val != float("inf") else "∞"

                popup = (
                    f"<b>ID: {sid}</b> ❌ 未匹配<br>"
                    f"Type: {s['Type']} | Fwy: {s.get('Fwy','')} {s.get('Dir','')}<br>"
                    f"Name: {s.get('Name','')}<br>"
                    f"坐标: ({slat:.6f}, {slon:.6f})<br>"
                    f"<hr>"
                    f"<b>原因:</b> {reason}<br>"
                    f"候选数: {n_cand}<br>"
                    f"最佳匹配距离: {dist_str}<br>"
                    f"最佳候选link: {best_link}<br>"
                )
                folium.CircleMarker(
                    location=[slat, slon], radius=5,
                    color="black", fill=True, fill_color="red",
                    fill_opacity=0.5,
                    popup=folium.Popup(popup, max_width=350),
                    tooltip=f"{sid} ✗ {reason}",
                ).add_to(fg)

            else:
                # === 未处理 (FF 等) ===
                popup = (
                    f"<b>ID: {sid}</b> ⏳ 待处理<br>"
                    f"Type: {s['Type']} | Fwy: {s.get('Fwy','')} {s.get('Dir','')}<br>"
                    f"Name: {s.get('Name','')}<br>"
                    f"坐标: ({slat:.6f}, {slon:.6f})<br>"
                )
                folium.CircleMarker(
                    location=[slat, slon], radius=4,
                    color="gray", fill=True, fill_opacity=0.3,
                    popup=folium.Popup(popup, max_width=350),
                    tooltip=f"{sid} ⏳ {s['Type']}",
                ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_path = os.path.join(output_dir, "network_map.html")
    m.save(map_path)
    log(f"  地图已保存: {map_path}")

    # --- Step 6.4: 最终统计 ---
    log("\n" + "=" * 70)
    log("最终汇总")
    log("=" * 70)

    total_sensors = len(sensors_df)
    type_totals = sensors_df["Type"].value_counts()

    ml_total = type_totals.get("ML", 0)
    or_total = type_totals.get("OR", 0)
    fr_total = type_totals.get("FR", 0)
    ff_total = type_totals.get("FF", 0)

    ml_matched = len(ml_mapping_df)
    or_matched = len(or_mapping_df)
    fr_matched = len(fr_mapping_df)

    log(f"  传感器总数:     {total_sensors}")
    log(f"  ML 匹配:        {ml_matched}/{ml_total} "
        f"({100*ml_matched/max(1,ml_total):.1f}%)")
    log(f"  OR 匹配:        {or_matched}/{or_total} "
        f"({100*or_matched/max(1,or_total):.1f}%)")
    log(f"  FR 匹配:        {fr_matched}/{fr_total} "
        f"({100*fr_matched/max(1,fr_total):.1f}%)")
    log(f"  FF 待手动标注:  {ff_total}")
    log(f"  总匹配数:       {ml_matched + or_matched + fr_matched}")
    log(f"  关联矩阵:       {incidence_df.shape}")
    log(f"  邻接矩阵:       {adjacency_df.shape}")

    log(f"\n  === 输出文件列表 ===")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size/1024/1024:.1f}MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size}B"
        log(f"    {f} ({size_str})")

    log("\n全部阶段完成!")


# =============================================================================
# 主流程
# =============================================================================
def run_pipeline(metadata_file, output_dir="output", buffer_deg=0.02,
                 ml_threshold=1e-4, ramp_threshold=0.01,
                 shn_shapefile=None, shn_postmiles_geojson=None,
                 max_edge_length_m=2000, shn_pm_tolerance=1.5,
                 shn_max_snap_distance_m=500.0, pbf_path=None):
    """
    执行完整 pipeline.

    参数:
        metadata_file: PeMS 元数据文件路径
        output_dir: 输出目录
        buffer_deg: bbox 外扩缓冲 (度)
        ml_threshold: ML 三角不等式阈值
        ramp_threshold: OR/FR 匝道匹配坐标阈值 (度)
        shn_shapefile: Caltrans SHN Lines shapefile 路径
        shn_postmiles_geojson: SHN_Postmiles_Tenth.geojson 路径, 用于 2.7 坐标修正
        max_edge_length_m: 边最大长度 (米), 超过则拆分
    """
    if shn_shapefile is None:
        raise ValueError("必须提供 --shn-shapefile 参数")
    if shn_postmiles_geojson is None:
        raise ValueError("必须提供 --shn-postmiles-geojson 参数")

    t0 = time.time()

    # Phase 0
    sensors_df, bbox_dict = phase0_load_metadata(
        metadata_file,
        output_dir,
        buffer_deg,
        shn_postmiles_geojson=shn_postmiles_geojson,
        shn_pm_tolerance=shn_pm_tolerance,
        shn_max_snap_distance_m=shn_max_snap_distance_m,
    )

    # Phase 1
    (G, nodes_gdf, edges_gdf, incidence_df, edge_info_df,
     edge_hw_map, links_dict, way_to_rels) = \
        phase1_extract_network(
            bbox_dict,
            output_dir,
            max_edge_length_m,
            pbf_path=pbf_path,
        )

    # Phase 2
    fwy_links, no_ref_df = phase2_assign_freeways_ref_only(
        sensors_df, edge_info_df, incidence_df, nodes_gdf, links_dict,
        way_to_rels, edges_gdf, shn_shapefile, output_dir
    )

    # Phase 3
    ml_mapping_df, ml_unmatched_df = phase3_match_ml_sensors(
        sensors_df, fwy_links, incidence_df, nodes_gdf, edges_gdf,
        output_dir, ml_threshold
    )

    # Phase 4
    or_mapping_df, or_unmatched_df, fr_mapping_df, fr_unmatched_df, ramp_data = \
        phase4_ramp_matching(
            sensors_df, fwy_links, incidence_df, nodes_gdf, edge_hw_map,
            output_dir, ramp_threshold
        )

    # Phase 5
    ff_df = phase5_ff_sensors(sensors_df, output_dir)

    # Phase 6
    phase6_final_summary(
        incidence_df, ml_mapping_df, or_mapping_df, fr_mapping_df, ff_df,
        sensors_df, nodes_gdf, edges_gdf, output_dir
    )

    elapsed = time.time() - t0
    log(f"\n总耗时: {elapsed:.1f} 秒")


def run_single_phase(phase_num, output_dir="output", **kwargs):
    """
    单独运行某一阶段 (从中间结果恢复).

    用法:
        run_single_phase(3, output_dir="output", ml_threshold=1e-3)
    """
    if phase_num == 0:
        return phase0_load_metadata(
            kwargs["metadata_file"],
            output_dir,
            kwargs.get("buffer_deg", 0.02),
            shn_postmiles_geojson=kwargs.get("shn_postmiles_geojson"),
            shn_pm_tolerance=kwargs.get("shn_pm_tolerance", 1.5),
            shn_max_snap_distance_m=kwargs.get("shn_max_snap_distance_m", 500.0),
        )

    elif phase_num == 1:
        bbox_dict = load_intermediate("phase0_bbox", output_dir)
        max_edge_length_m = kwargs.get("max_edge_length_m", 2000)
        return phase1_extract_network(
            bbox_dict,
            output_dir,
            max_edge_length_m,
            pbf_path=kwargs.get("pbf_path"),
        )

    elif phase_num == 2:
        sensors_df = load_intermediate("phase0_sensors", output_dir)
        edge_info_df = load_intermediate("phase1_edge_info", output_dir)
        incidence_df = load_intermediate("phase1_incidence", output_dir)
        nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)
        links_dict = load_intermediate("phase1_links_dict", output_dir)
        way_to_rels = load_intermediate("phase1_way_relations", output_dir)
        edges_gdf = load_intermediate("phase1_edges_gdf", output_dir)
        shn_shapefile = kwargs.get("shn_shapefile")
        if shn_shapefile is None:
            raise ValueError("Phase 2 需要 shn_shapefile 参数")
        return phase2_assign_freeways_ref_only(
            sensors_df, edge_info_df, incidence_df, nodes_gdf, links_dict,
            way_to_rels, edges_gdf, shn_shapefile, output_dir
        )

    elif phase_num == 3:
        sensors_df = load_intermediate("phase0_sensors", output_dir)
        fwy_links = load_intermediate("phase2_fwy_links", output_dir)
        incidence_df = load_intermediate("phase1_incidence", output_dir)
        nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)
        edges_gdf = load_intermediate("phase1_edges_gdf", output_dir)
        threshold = kwargs.get("ml_threshold", 1e-4)
        return phase3_match_ml_sensors(
            sensors_df, fwy_links, incidence_df, nodes_gdf, edges_gdf,
            output_dir, threshold
        )

    elif phase_num == 4:
        sensors_df = load_intermediate("phase0_sensors", output_dir)
        fwy_links = load_intermediate("phase2_fwy_links", output_dir)
        incidence_df = load_intermediate("phase1_incidence", output_dir)
        nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)
        edge_hw_map = load_intermediate("phase1_edge_highway_map", output_dir)
        threshold = kwargs.get("ramp_threshold", 0.01)
        return phase4_ramp_matching(
            sensors_df, fwy_links, incidence_df, nodes_gdf, edge_hw_map,
            output_dir, threshold
        )

    elif phase_num == 5:
        sensors_df = load_intermediate("phase0_sensors", output_dir)
        return phase5_ff_sensors(sensors_df, output_dir)

    elif phase_num == 6:
        incidence_df = load_intermediate("phase1_incidence", output_dir)
        ml_mapping_df = load_intermediate("phase3_ml_mapping", output_dir)
        or_mapping_df = load_intermediate("phase4_or_mapping", output_dir)
        fr_mapping_df = load_intermediate("phase4_fr_mapping", output_dir)
        ff_df = load_intermediate("phase5_ff_to_manual", output_dir)
        sensors_df = load_intermediate("phase0_sensors", output_dir)
        nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)
        edges_gdf = load_intermediate("phase1_edges_gdf", output_dir)
        return phase6_final_summary(
            incidence_df, ml_mapping_df, or_mapping_df, fr_mapping_df, ff_df,
            sensors_df, nodes_gdf, edges_gdf, output_dir
        )


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PeMS Sensor to OSM Network Matching Pipeline"
    )
    parser.add_argument(
        "metadata_file",
        help="PeMS 元数据文件路径 (tab 分隔 .txt)",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="输出目录 (默认: output)",
    )
    parser.add_argument(
        "--buffer", type=float, default=0.02,
        help="Bbox 缓冲 (度, 默认: 0.02)",
    )
    parser.add_argument(
        "--ml-threshold", type=float, default=1e-4,
        help="ML 三角不等式阈值 (默认: 1e-4)",
    )
    parser.add_argument(
        "--ramp-threshold", type=float, default=0.01,
        help="OR/FR 坐标匹配阈值 (度, 默认: 0.01)",
    )
    parser.add_argument(
        "--shn-shapefile", type=str, required=True,
        help="Caltrans SHN Lines shapefile 路径 (.shp)",
    )
    parser.add_argument(
        "--shn-postmiles-geojson", "--shn-lines-geojson",
        dest="shn_postmiles_geojson", type=str, required=True,
        help="SHN_Postmiles_Tenth.geojson 路径, 用于 2.7 传感器坐标修正",
    )
    parser.add_argument(
        "--max-edge-length", type=float, default=2000,
        help="边最大长度 (米), 超过则拆分 (默认: 2000)",
    )
    parser.add_argument(
        "--pbf-path", type=str, default=None,
        help="本地 OSM PBF 文件路径; 不指定时默认使用 output_dir 上级目录中的 california-latest.osm.pbf",
    )
    parser.add_argument(
        "--shn-pm-tolerance", type=float, default=1.5,
        help="SHN 里程候选筛选容忍范围 (mile, 默认: 1.5)",
    )
    parser.add_argument(
        "--shn-max-snap-distance", type=float, default=500.0,
        help="SHN 坐标修正最大贴线距离 (米, 默认: 500)",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="只运行指定阶段 (0-6), 不指定则运行全部",
    )

    args = parser.parse_args()

    if args.phase is not None:
        run_single_phase(
            args.phase,
            output_dir=args.output,
            metadata_file=args.metadata_file,
            buffer_deg=args.buffer,
            ml_threshold=args.ml_threshold,
            ramp_threshold=args.ramp_threshold,
            shn_shapefile=args.shn_shapefile,
            shn_postmiles_geojson=args.shn_postmiles_geojson,
            max_edge_length_m=args.max_edge_length,
            pbf_path=args.pbf_path,
            shn_pm_tolerance=args.shn_pm_tolerance,
            shn_max_snap_distance_m=args.shn_max_snap_distance,
        )
    else:
        run_pipeline(
            args.metadata_file,
            output_dir=args.output,
            buffer_deg=args.buffer,
            ml_threshold=args.ml_threshold,
            ramp_threshold=args.ramp_threshold,
            shn_shapefile=args.shn_shapefile,
            shn_postmiles_geojson=args.shn_postmiles_geojson,
            max_edge_length_m=args.max_edge_length,
            pbf_path=args.pbf_path,
            shn_pm_tolerance=args.shn_pm_tolerance,
            shn_max_snap_distance_m=args.shn_max_snap_distance,
        )

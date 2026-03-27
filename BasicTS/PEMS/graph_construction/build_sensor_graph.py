"""
Phase 7: 构建传感器图 (Sensor Graph)
======================================
标准流程:
1. 保留已匹配的目标类型传感器作为图节点 (默认 ML/OR/FR)
2. 先按几何距离阈值筛选候选节点对
3. 分别用两种方式计算边权:
   - local: 基于提取出的高速路子图计算最短路长度
   - osrm: 通过 OSRM 双向 route 查询计算最短路长度
4. 用 a->b 与 b->a 的最短路长度比较边方向
5. 若存在 a->b, b->c, a->c, 则删除三者中最长的边
6. 生成有向图、边表、距离矩阵、邻接矩阵

默认输出:
    phase7_local_*   - 基于本地提取路网的图
    phase7_osrm_*    - 基于 OSRM 的图
    phase7_*         - 兼容旧流程, 指向 local 版本

用法:
    python build_sensor_graph.py output_d03_2.7_full --pair-threshold 5000
"""

import json
import math
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import folium
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring as shp_substring


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def load_intermediate(name, output_dir):
    pkl_path = os.path.join(output_dir, f"{name}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到中间结果: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_intermediate(obj, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(obj, f)
    log(f"  Saved {pkl_path}")
    if isinstance(obj, pd.DataFrame):
        csv_path = os.path.join(output_dir, f"{name}.csv")
        obj.to_csv(csv_path, index=False)
        log(f"  Saved {csv_path} ({obj.shape[0]} rows × {obj.shape[1]} cols)")


def parse_edge_key(edge_key):
    parts = edge_key.split("_")
    key = int(parts[-1])
    v = int(parts[-2])
    u = int("_".join(parts[:-2]))
    return u, v, key


def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def query_osrm_distance(lat1, lon1, lat2, lon2, server, timeout=10.0):
    coords = f"{lon1},{lat1};{lon2},{lat2}"
    params = urlencode({
        "overview": "full",
        "steps": "false",
        "annotations": "false",
        "geometries": "geojson",
    })
    url = f"{server.rstrip('/')}/route/v1/driving/{coords}?{params}"

    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "distance_m": np.nan,
            "duration_s": np.nan,
            "error": str(exc),
        }

    if payload.get("code") != "Ok" or not payload.get("routes"):
        return {
            "status": payload.get("code", "error"),
            "distance_m": np.nan,
            "duration_s": np.nan,
            "error": payload.get("message", "OSRM 无可用路线"),
        }

    route = payload["routes"][0]
    geometry = route.get("geometry", {})
    geometry_coords = []
    if geometry.get("type") == "LineString":
        geometry_coords = [
            [float(lat), float(lon)]
            for lon, lat in geometry.get("coordinates", [])
        ]
    return {
        "status": "ok",
        "distance_m": float(route.get("distance", np.nan)),
        "duration_s": float(route.get("duration", np.nan)),
        "geometry_coords": geometry_coords,
        "error": "",
    }


def collect_sensors(output_dir, sensor_types):
    sensor_types = [str(t).upper() for t in sensor_types if str(t).strip()]
    sensor_types = sorted(set(sensor_types))
    if not sensor_types:
        raise ValueError("sensor_types 不能为空")

    log(f"Step 7.1: 收集传感器 (types={','.join(sensor_types)})...")
    all_mapping_df = load_intermediate("phase6_all_mapping", output_dir)
    sensors_df = load_intermediate("phase0_sensors", output_dir)

    all_mapping_df = all_mapping_df.copy()
    all_mapping_df["type"] = all_mapping_df["type"].astype(str).str.upper()
    selected_mapping = all_mapping_df[all_mapping_df["type"].isin(sensor_types)].copy()

    sensors_meta = sensors_df.copy()
    sensors_meta["Type"] = sensors_meta["Type"].astype(str).str.upper()
    sensors_meta = sensors_meta[sensors_meta["Type"].isin(sensor_types)]
    meta_by_id = {
        int(row["ID"]): row.to_dict()
        for _, row in sensors_meta.iterrows()
    }

    sensors = []
    for _, row in selected_mapping.iterrows():
        sensor_id = int(row["sensor_id"])
        sensor_type = str(row.get("type", "")).upper()
        meta = meta_by_id.get(sensor_id, {})

        lat = meta.get("corrected_lat", meta.get("Latitude", np.nan))
        lon = meta.get("corrected_lon", meta.get("Longitude", np.nan))
        if pd.isna(lat) or pd.isna(lon):
            continue

        fwy_val = row.get("fwy", meta.get("Fwy", np.nan))
        dir_val = row.get("dir", meta.get("Dir", ""))
        if pd.isna(fwy_val) or pd.isna(dir_val):
            continue

        sensors.append({
            "sensor_id": sensor_id,
            "link_id": row["link_id"],
            "type": sensor_type,
            "fwy": int(fwy_val),
            "dir": str(dir_val),
            "lat": float(lat),
            "lon": float(lon),
            "name": meta.get("Name", ""),
            "abs_pm": meta.get("Abs_PM", np.nan),
            "raw_lat": meta.get("raw_lat", lat),
            "raw_lon": meta.get("raw_lon", lon),
            "corrected_lat": meta.get("corrected_lat", lat),
            "corrected_lon": meta.get("corrected_lon", lon),
            "correction_method": meta.get("correction_method", ""),
            "correction_dist_m": meta.get("correction_dist_m", np.nan),
        })

    if sensors:
        type_counts = pd.Series([s["type"] for s in sensors]).value_counts().to_dict()
        log(f"  传感器节点数: {len(sensors)} | 分类型: {type_counts}")
    else:
        log("  传感器节点数: 0", "WARN")
    return sensors


def embed_sensors_into_graph(graph, edges_gdf, nodes_gdf, sensors):
    log("Step 7.2: 将传感器嵌入图...")

    edge_geom = {}
    for (u, v, key), row in edges_gdf.iterrows():
        edge_key = f"{u}_{v}_{key}"
        if row.geometry and not row.geometry.is_empty:
            edge_geom[edge_key] = row.geometry
        else:
            try:
                u_node = nodes_gdf.loc[u]
                v_node = nodes_gdf.loc[v]
                edge_geom[edge_key] = LineString([
                    (u_node["x"], u_node["y"]),
                    (v_node["x"], v_node["y"]),
                ])
            except KeyError:
                continue

    edge_sensors = defaultdict(list)
    for sensor in sensors:
        edge_sensors[sensor["link_id"]].append(sensor)

    sensor_node_map = {}
    embedded = 0
    skipped = 0

    for edge_key, sensor_list in edge_sensors.items():
        geom = edge_geom.get(edge_key)
        if geom is None or geom.length == 0:
            skipped += len(sensor_list)
            continue

        try:
            u, v, key = parse_edge_key(edge_key)
        except (ValueError, IndexError):
            skipped += len(sensor_list)
            continue

        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            skipped += len(sensor_list)
            continue
        if key not in edge_data:
            key = list(edge_data.keys())[0]

        original_data = dict(edge_data[key])
        edge_length_m = float(original_data.get("length", 0.0))
        if edge_length_m <= 0:
            skipped += len(sensor_list)
            continue

        for sensor in sensor_list:
            point = Point(sensor["lon"], sensor["lat"])
            proj_ratio = geom.project(point, normalized=True)
            sensor["_proj_ratio"] = max(0.001, min(0.999, proj_ratio))

        sensor_list.sort(key=lambda item: item["_proj_ratio"])

        graph.remove_edge(u, v, key)
        chain = [u]
        ratios = [0.0]

        for sensor in sensor_list:
            vnode = f"S_{sensor['sensor_id']}"
            proj_pt = geom.interpolate(sensor["_proj_ratio"], normalized=True)
            graph.add_node(
                vnode,
                x=proj_pt.x,
                y=proj_pt.y,
                sensor_id=sensor["sensor_id"],
                sensor_type="ML",
            )
            sensor_node_map[sensor["sensor_id"]] = vnode
            sensor["proj_lon"] = proj_pt.x
            sensor["proj_lat"] = proj_pt.y
            chain.append(vnode)
            ratios.append(sensor["_proj_ratio"])

        chain.append(v)
        ratios.append(1.0)

        base_attrs = {
            key_name: value for key_name, value in original_data.items()
            if key_name not in ("length", "geometry")
        }
        has_geometry = original_data.get("geometry") is not None
        for idx in range(len(chain) - 1):
            seg_u = chain[idx]
            seg_v = chain[idx + 1]
            seg_ratio = ratios[idx + 1] - ratios[idx]
            attrs = dict(base_attrs)
            attrs["length"] = edge_length_m * seg_ratio
            if has_geometry:
                try:
                    sub_geom = shp_substring(geom, ratios[idx], ratios[idx + 1], normalized=True)
                    if sub_geom is not None and not sub_geom.is_empty:
                        attrs["geometry"] = sub_geom
                except Exception:
                    pass
            graph.add_edge(seg_u, seg_v, **attrs)

        embedded += len(sensor_list)

    log(f"  嵌入成功: {embedded}")
    if skipped:
        log(f"  跳过: {skipped}", "WARN")
    return graph, sensor_node_map


def build_sensor_info_df(sensors, sensor_node_map, embedded_graph):
    records = []
    for sensor in sensors:
        sensor_id = sensor["sensor_id"]
        if sensor_id not in sensor_node_map:
            continue
        vnode = sensor_node_map[sensor_id]
        node = embedded_graph.nodes[vnode]
        records.append({
            "sensor_id": sensor_id,
            "name": sensor.get("name", ""),
            "type": sensor.get("type", ""),
            "fwy": sensor["fwy"],
            "dir": sensor["dir"],
            "link_id": sensor["link_id"],
            "graph_node": vnode,
            "lat": sensor["lat"],
            "lon": sensor["lon"],
            "proj_lat": float(node.get("y", sensor.get("proj_lat", sensor["lat"]))),
            "proj_lon": float(node.get("x", sensor.get("proj_lon", sensor["lon"]))),
            "raw_lat": sensor.get("raw_lat", sensor["lat"]),
            "raw_lon": sensor.get("raw_lon", sensor["lon"]),
            "corrected_lat": sensor.get("corrected_lat", sensor["lat"]),
            "corrected_lon": sensor.get("corrected_lon", sensor["lon"]),
            "abs_pm": sensor.get("abs_pm", np.nan),
            "correction_method": sensor.get("correction_method", ""),
            "correction_dist_m": sensor.get("correction_dist_m", np.nan),
        })
    sensor_info_df = pd.DataFrame(records)
    sensor_info_df = sensor_info_df.sort_values(["fwy", "dir", "sensor_id"]).reset_index(drop=True)
    log(f"  生成 sensor_info: {len(sensor_info_df)} 行")
    return sensor_info_df


def build_candidate_pairs(sensor_info_df, pair_threshold_m):
    log(f"Step 7.3: 按几何距离筛选候选对 (threshold={pair_threshold_m:.0f}m)...")
    records = []
    sensor_records = sensor_info_df.to_dict("records")
    for left, right in combinations(sensor_records, 2):
        geo_dist_m = haversine_m(
            left["proj_lat"], left["proj_lon"], right["proj_lat"], right["proj_lon"]
        )
        if geo_dist_m > pair_threshold_m:
            continue
        same_fwy = int(left["fwy"]) == int(right["fwy"])
        same_dir = str(left["dir"]) == str(right["dir"])
        records.append({
            "source_id": int(left["sensor_id"]),
            "target_id": int(right["sensor_id"]),
            "fwy": int(left["fwy"]),
            "dir": str(left["dir"]),
            "source_fwy": int(left["fwy"]),
            "source_dir": str(left["dir"]),
            "target_fwy": int(right["fwy"]),
            "target_dir": str(right["dir"]),
            "same_fwy": bool(same_fwy),
            "same_dir": bool(same_dir),
            "geo_dist_m": float(geo_dist_m),
            "source_abs_pm": left.get("abs_pm", np.nan),
            "target_abs_pm": right.get("abs_pm", np.nan),
        })
    candidate_df = pd.DataFrame(records)
    log(f"  候选无向对数: {len(candidate_df)}")
    return candidate_df


def compute_local_pair_paths(embedded_graph, sensor_node_map, candidate_df, route_cutoff_m=None):
    log("Step 7.4: 计算本地高速路最短路权重...")
    pair_lookup = defaultdict(set)
    for _, row in candidate_df.iterrows():
        left = int(row["source_id"])
        right = int(row["target_id"])
        pair_lookup[left].add(right)
        pair_lookup[right].add(left)

    distance_lookup = {}
    path_lookup = {}
    sensor_ids = sorted(pair_lookup.keys())
    started = time.time()

    for index, source_id in enumerate(sensor_ids, start=1):
        source_node = sensor_node_map.get(source_id)
        if source_node not in embedded_graph:
            continue
        try:
            lengths, paths = nx.single_source_dijkstra(
                embedded_graph,
                source_node,
                cutoff=route_cutoff_m,
                weight="length",
            )
        except nx.NetworkXError:
            continue

        for target_id in pair_lookup[source_id]:
            target_node = sensor_node_map.get(target_id)
            if target_node in lengths:
                distance_lookup[(source_id, target_id)] = float(lengths[target_node])
                path_lookup[(source_id, target_id)] = paths[target_node]

        if index % 100 == 0:
            log(f"    本地最短路进度: {index}/{len(sensor_ids)} ({time.time()-started:.1f}s)")

    log(f"  已得到本地有向最短路: {len(distance_lookup)} 条")
    return distance_lookup, path_lookup


def choose_direction(forward_dist, backward_dist, tie_tolerance_m=1.0):
    forward_ok = pd.notna(forward_dist) and np.isfinite(forward_dist)
    backward_ok = pd.notna(backward_dist) and np.isfinite(backward_dist)

    if not forward_ok and not backward_ok:
        return None, np.nan, "unreachable"
    if forward_ok and not backward_ok:
        return "forward", float(forward_dist), "forward_only"
    if backward_ok and not forward_ok:
        return "backward", float(backward_dist), "backward_only"
    if abs(float(forward_dist) - float(backward_dist)) <= tie_tolerance_m:
        return None, float(min(forward_dist, backward_dist)), "tie"
    if float(forward_dist) < float(backward_dist):
        return "forward", float(forward_dist), "forward_shorter"
    return "backward", float(backward_dist), "backward_shorter"


def build_directed_edges_from_pairs(candidate_df, distance_lookup, method_name,
                                    max_route_distance_m=None, extra_lookup=None):
    records = []
    for _, row in candidate_df.iterrows():
        left = int(row["source_id"])
        right = int(row["target_id"])
        dist_forward = distance_lookup.get((left, right), np.nan)
        dist_backward = distance_lookup.get((right, left), np.nan)
        direction, weight_m, status = choose_direction(dist_forward, dist_backward)

        if (
            direction is not None
            and max_route_distance_m is not None
            and pd.notna(weight_m)
            and np.isfinite(weight_m)
            and float(weight_m) > float(max_route_distance_m)
        ):
            direction = None
            status = "path_exceeds_threshold"

        record = {
            "pair_source_id": left,
            "pair_target_id": right,
            "fwy": int(row["fwy"]),
            "dir": row["dir"],
            "source_fwy": row.get("source_fwy", np.nan),
            "source_dir": row.get("source_dir", ""),
            "target_fwy": row.get("target_fwy", np.nan),
            "target_dir": row.get("target_dir", ""),
            "same_fwy": row.get("same_fwy", False),
            "same_dir": row.get("same_dir", False),
            "geo_dist_m": float(row["geo_dist_m"]),
            f"{method_name}_forward_m": dist_forward,
            f"{method_name}_backward_m": dist_backward,
            "chosen_direction": direction,
            "weight_m": weight_m,
            "status": status,
        }
        if direction == "forward":
            record["source_id"] = left
            record["target_id"] = right
        elif direction == "backward":
            record["source_id"] = right
            record["target_id"] = left
        else:
            record["source_id"] = np.nan
            record["target_id"] = np.nan

        if extra_lookup is not None:
            extra = extra_lookup.get((left, right), {})
            for key, value in extra.items():
                record[key] = value
        records.append(record)

    pair_results_df = pd.DataFrame(records)
    edge_df = pair_results_df[pair_results_df["chosen_direction"].isin(["forward", "backward"])].copy()
    if not edge_df.empty:
        edge_df["source_id"] = edge_df["source_id"].astype(int)
        edge_df["target_id"] = edge_df["target_id"].astype(int)
        edge_df["edge_id"] = edge_df.apply(
            lambda row: f"{int(row['source_id'])}->{int(row['target_id'])}", axis=1
        )
        edge_df = edge_df.drop_duplicates(subset=["source_id", "target_id"]).reset_index(drop=True)
    return pair_results_df, edge_df


def apply_osrm_highway_mask(osrm_pair_results_df, osrm_raw_edge_df, local_raw_edge_df):
    log("Step 7.6b: 按本地 highway 图过滤 OSRM 连边...")
    if osrm_raw_edge_df.empty or local_raw_edge_df.empty:
        log("  OSRM 或 local 原始边为空, 跳过 highway 过滤", "WARN")
        return osrm_pair_results_df, osrm_raw_edge_df, 0

    local_allowed = {
        (int(row["source_id"]), int(row["target_id"]))
        for _, row in local_raw_edge_df.iterrows()
    }

    keep_rows = []
    removed_directed = set()
    for _, row in osrm_raw_edge_df.iterrows():
        key = (int(row["source_id"]), int(row["target_id"]))
        if key in local_allowed:
            keep_rows.append(row.to_dict())
        else:
            removed_directed.add(key)

    filtered_raw = pd.DataFrame(keep_rows)
    if not filtered_raw.empty:
        filtered_raw["source_id"] = filtered_raw["source_id"].astype(int)
        filtered_raw["target_id"] = filtered_raw["target_id"].astype(int)
        filtered_raw = filtered_raw.reset_index(drop=True)

    updated_pair = osrm_pair_results_df.copy()
    removed_count = 0
    for idx, row in updated_pair.iterrows():
        chosen = row.get("chosen_direction")
        if chosen not in ("forward", "backward"):
            continue
        left = int(row["pair_source_id"])
        right = int(row["pair_target_id"])
        directed = (left, right) if chosen == "forward" else (right, left)
        if directed in removed_directed:
            updated_pair.at[idx, "chosen_direction"] = None
            updated_pair.at[idx, "source_id"] = np.nan
            updated_pair.at[idx, "target_id"] = np.nan
            updated_pair.at[idx, "status"] = "masked_by_local_highway"
            removed_count += 1

    log(f"  OSRM 原始边: {len(osrm_raw_edge_df)} -> 过滤后: {len(filtered_raw)}")
    log(f"  被 highway 过滤的候选对: {removed_count}")
    return updated_pair, filtered_raw, removed_count


def prune_transitive_longest_edges(edge_df):
    log("Step 7.5: 删除 a->b, b->c, a->c 中最长边...")
    if edge_df.empty:
        return edge_df.copy(), pd.DataFrame()

    edge_rows = {
        (int(row["source_id"]), int(row["target_id"])): row.to_dict()
        for _, row in edge_df.iterrows()
    }
    removed = []

    changed = True
    while changed:
        changed = False
        out_neighbors = defaultdict(set)
        for source_id, target_id in edge_rows:
            out_neighbors[source_id].add(target_id)

        for source_id, middle_id in list(edge_rows.keys()):
            if (source_id, middle_id) not in edge_rows:
                continue
            for target_id in list(out_neighbors.get(middle_id, set())):
                if target_id == source_id:
                    continue
                if (source_id, target_id) not in edge_rows:
                    continue
                tri_edges = [
                    (source_id, middle_id),
                    (middle_id, target_id),
                    (source_id, target_id),
                ]
                longest = max(
                    tri_edges,
                    key=lambda item: float(edge_rows[item]["weight_m"]),
                )
                removed.append({
                    "u": source_id,
                    "v": middle_id,
                    "w": target_id,
                    "removed_edge": f"{longest[0]}->{longest[1]}",
                    "removed_weight_m": float(edge_rows[longest]["weight_m"]),
                })
                del edge_rows[longest]
                changed = True
                break
            if changed:
                break

    pruned_df = pd.DataFrame(edge_rows.values())
    if not pruned_df.empty:
        pruned_df = pruned_df.sort_values(["source_id", "target_id"]).reset_index(drop=True)
    removed_df = pd.DataFrame(removed)
    log(f"  删边数: {len(removed_df)}")
    log(f"  保留边数: {len(pruned_df)}")
    return pruned_df, removed_df


def build_graph_and_matrices(sensor_info_df, edge_df, weight_column="weight_m"):
    sensor_ids = sorted(sensor_info_df["sensor_id"].astype(int).tolist())
    graph = nx.DiGraph()
    for _, row in sensor_info_df.iterrows():
        graph.add_node(int(row["sensor_id"]), **row.to_dict())

    dist_mat = np.full((len(sensor_ids), len(sensor_ids)), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0.0)
    adj_mat = np.zeros((len(sensor_ids), len(sensor_ids)), dtype=np.int8)
    sid_to_idx = {sid: idx for idx, sid in enumerate(sensor_ids)}

    for _, row in edge_df.iterrows():
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        weight_m = float(row[weight_column])
        graph.add_edge(source_id, target_id, **row.to_dict())
        dist_mat[sid_to_idx[source_id], sid_to_idx[target_id]] = weight_m
        adj_mat[sid_to_idx[source_id], sid_to_idx[target_id]] = 1

    dist_df = pd.DataFrame(dist_mat, index=sensor_ids, columns=sensor_ids)
    adj_df = pd.DataFrame(adj_mat, index=sensor_ids, columns=sensor_ids)
    return graph, dist_df, adj_df


def edge_geometry_coords(graph, source_node, target_node):
    edge_data = graph.get_edge_data(source_node, target_node)
    if not edge_data:
        return []
    for _, attrs in edge_data.items():
        geom = attrs.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            return [[float(lat), float(lon)] for lon, lat in geom.coords]
        source_attrs = graph.nodes.get(source_node, {})
        target_attrs = graph.nodes.get(target_node, {})
        if "y" in source_attrs and "x" in source_attrs and "y" in target_attrs and "x" in target_attrs:
            return [
                [float(source_attrs["y"]), float(source_attrs["x"])],
                [float(target_attrs["y"]), float(target_attrs["x"])],
            ]
    return []


def path_nodes_to_coords(graph, path_nodes):
    if not path_nodes or len(path_nodes) < 2:
        return []
    coords = []
    for idx in range(len(path_nodes) - 1):
        segment = edge_geometry_coords(graph, path_nodes[idx], path_nodes[idx + 1])
        if not segment:
            continue
        if coords and coords[-1] == segment[0]:
            segment = segment[1:]
        coords.extend(segment)
    return coords


def attach_local_edge_geometry(edge_df, local_paths, sensor_node_map, embedded_graph):
    if edge_df.empty:
        edge_df = edge_df.copy()
        edge_df["path_geometry"] = []
        return edge_df
    rows = []
    for _, row in edge_df.iterrows():
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        path_nodes = local_paths.get((source_id, target_id))
        if path_nodes is None:
            source_node = sensor_node_map.get(source_id)
            target_node = sensor_node_map.get(target_id)
            path_nodes = [source_node, target_node] if source_node and target_node else []
        coords = path_nodes_to_coords(embedded_graph, path_nodes)
        row_dict = row.to_dict()
        row_dict["path_geometry"] = json.dumps(coords)
        rows.append(row_dict)
    return pd.DataFrame(rows)


def attach_osrm_edge_geometry(edge_df):
    if edge_df.empty:
        edge_df = edge_df.copy()
        edge_df["path_geometry"] = []
        return edge_df
    rows = []
    for _, row in edge_df.iterrows():
        row_dict = row.to_dict()
        if row_dict.get("chosen_direction") == "forward":
            coords = row_dict.get("osrm_forward_geometry", [])
        else:
            coords = row_dict.get("osrm_backward_geometry", [])
        row_dict["path_geometry"] = json.dumps(coords if isinstance(coords, list) else [])
        rows.append(row_dict)
    return pd.DataFrame(rows)


def draw_sensor_graph_map(output_dir, method_name, sensor_info_df, edge_df, title):
    log(f"Step 7.{7 if method_name == 'local' else 8}: 绘制 {method_name} 传感器图地图...")
    if sensor_info_df.empty:
        log("  无 sensor, 跳过地图", "WARN")
        return None

    center_lat = float(sensor_info_df["proj_lat"].mean())
    center_lon = float(sensor_info_df["proj_lon"].mean())
    graph_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    edge_layer = folium.FeatureGroup(name=f"{method_name} edges", show=True)

    sensor_styles = {
        "ML": {"color": "#1f77b4", "radius": 4, "label": "ML sensors"},
        "OR": {"color": "#d62728", "radius": 5, "label": "OR sensors"},
        "FR": {"color": "#2ca02c", "radius": 5, "label": "FR sensors"},
    }
    sensor_layers = {}
    present_types = []
    for sensor_type in sensor_info_df["type"].fillna("UNKNOWN").astype(str).str.upper().unique().tolist():
        style = sensor_styles.get(sensor_type, {"color": "#7f7f7f", "radius": 4, "label": f"{sensor_type} sensors"})
        sensor_layers[sensor_type] = folium.FeatureGroup(name=style["label"], show=True)
        present_types.append(sensor_type)

    weight_series = edge_df["weight_m"].dropna() if not edge_df.empty else pd.Series(dtype=float)
    min_weight = float(weight_series.min()) if not weight_series.empty else 0.0
    max_weight = float(weight_series.max()) if not weight_series.empty else 1.0

    def weight_to_color(weight_m):
        if max_weight <= min_weight:
            ratio = 0.5
        else:
            ratio = (float(weight_m) - min_weight) / (max_weight - min_weight)
        ratio = max(0.0, min(1.0, ratio))
        red = int(255 * ratio)
        green = int(255 * (1.0 - ratio))
        return f"#{red:02x}{green:02x}00"

    for _, row in edge_df.iterrows():
        try:
            coords = json.loads(row.get("path_geometry", "[]"))
        except json.JSONDecodeError:
            coords = []
        if len(coords) < 2:
            continue
        popup = (
            f"<b>{int(row['source_id'])}→{int(row['target_id'])}</b><br>"
            f"Weight: {float(row['weight_m']):.1f} m<br>"
            f"Fwy: {row['fwy']} {row['dir']}<br>"
            f"Method: {method_name}"
        )
        folium.PolyLine(
            locations=coords,
            color=weight_to_color(row["weight_m"]),
            weight=3,
            opacity=0.8,
            popup=folium.Popup(popup, max_width=260),
            tooltip=f"{int(row['source_id'])}→{int(row['target_id'])}: {float(row['weight_m']):.0f}m",
        ).add_to(edge_layer)

    for _, row in sensor_info_df.iterrows():
        sensor_type = str(row.get("type", "UNKNOWN")).upper()
        style = sensor_styles.get(sensor_type, {"color": "#7f7f7f", "radius": 4, "label": f"{sensor_type} sensors"})
        sensor_layer = sensor_layers[sensor_type]
        popup = (
            f"<b>Sensor {int(row['sensor_id'])}</b><br>"
            f"Type: {sensor_type}<br>"
            f"{row['name']}<br>"
            f"Fwy {row['fwy']}{row['dir']}<br>"
            f"Abs_PM: {row['abs_pm']}"
        )
        folium.CircleMarker(
            location=[float(row["proj_lat"]), float(row["proj_lon"])],
            radius=style["radius"],
            color=style["color"],
            fill=True,
            fill_color=style["color"],
            fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=260),
        ).add_to(sensor_layer)

    edge_layer.add_to(graph_map)
    for sensor_type in present_types:
        sensor_layers[sensor_type].add_to(graph_map)
    folium.LayerControl(collapsed=False).add_to(graph_map)

    title_html = f"<h4 style='position: fixed; top: 10px; left: 50px; z-index:9999; background: white; padding: 6px 10px;'>{title}</h4>"
    graph_map.get_root().html.add_child(folium.Element(title_html))

    legend_items = []
    for sensor_type in present_types:
        style = sensor_styles.get(sensor_type, {"color": "#7f7f7f"})
        legend_items.append(
            f"<div><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{style['color']};margin-right:6px;'></span>{sensor_type}</div>"
        )
    legend_html = (
        "<div style='position: fixed; bottom: 30px; left: 20px; z-index:9999; background: white; "
        "padding: 8px 10px; border: 1px solid #ccc; font-size: 12px;'>"
        "<div style='font-weight: 700; margin-bottom: 4px;'>Sensor Types</div>"
        + "".join(legend_items)
        + "</div>"
    )
    graph_map.get_root().html.add_child(folium.Element(legend_html))

    map_path = os.path.join(output_dir, f"phase7_{method_name}_graph_map.html")
    graph_map.save(map_path)
    log(f"  地图已保存: {map_path}")
    return map_path


def compute_osrm_pair_metrics(candidate_df, sensor_info_df, server, timeout, sleep_s):
    log(f"Step 7.6: 计算 OSRM 最短路权重 (server={server})...")
    sensor_lookup = {
        int(row["sensor_id"]): row.to_dict()
        for _, row in sensor_info_df.iterrows()
    }
    distance_lookup = {}
    extra_lookup = {}
    started = time.time()

    for index, row in enumerate(candidate_df.to_dict("records"), start=1):
        left = int(row["source_id"])
        right = int(row["target_id"])
        left_info = sensor_lookup[left]
        right_info = sensor_lookup[right]

        forward = query_osrm_distance(
            left_info["proj_lat"], left_info["proj_lon"],
            right_info["proj_lat"], right_info["proj_lon"],
            server=server,
            timeout=timeout,
        )
        backward = query_osrm_distance(
            right_info["proj_lat"], right_info["proj_lon"],
            left_info["proj_lat"], left_info["proj_lon"],
            server=server,
            timeout=timeout,
        )

        if forward["status"] == "ok":
            distance_lookup[(left, right)] = forward["distance_m"]
        if backward["status"] == "ok":
            distance_lookup[(right, left)] = backward["distance_m"]

        extra_lookup[(left, right)] = {
            "osrm_forward_status": forward["status"],
            "osrm_backward_status": backward["status"],
            "osrm_forward_duration_s": forward["duration_s"],
            "osrm_backward_duration_s": backward["duration_s"],
            "osrm_forward_geometry": forward.get("geometry_coords", []),
            "osrm_backward_geometry": backward.get("geometry_coords", []),
            "osrm_forward_error": forward["error"],
            "osrm_backward_error": backward["error"],
        }

        if sleep_s > 0:
            time.sleep(sleep_s)
        if index % 100 == 0:
            log(f"    OSRM 进度: {index}/{len(candidate_df)} ({time.time()-started:.1f}s)")

    log(f"  OSRM 成功有向路由数: {len(distance_lookup)}")
    return distance_lookup, extra_lookup


def save_method_outputs(output_dir, method_name, sensor_info_df, pair_results_df,
                        raw_edge_df, pruned_edge_df, removed_df,
                        dist_df, adj_df, graph_obj):
    prefix = f"phase7_{method_name}"
    save_intermediate(pair_results_df, f"{prefix}_pair_results", output_dir)
    save_intermediate(raw_edge_df, f"{prefix}_edges_raw", output_dir)
    save_intermediate(pruned_edge_df, f"{prefix}_edges", output_dir)
    save_intermediate(removed_df, f"{prefix}_triangle_removed", output_dir)
    save_intermediate(dist_df, f"{prefix}_dist_matrix", output_dir)
    save_intermediate(adj_df, f"{prefix}_adj_matrix", output_dir)
    save_intermediate(sensor_info_df, f"{prefix}_sensor_info", output_dir)
    save_intermediate(graph_obj, f"{prefix}_graph", output_dir)

    np_dir = os.path.join(output_dir, method_name)
    os.makedirs(np_dir, exist_ok=True)
    np.save(os.path.join(np_dir, "distance_matrix.npy"), dist_df.values)
    np.save(os.path.join(np_dir, "adjacency_matrix.npy"), adj_df.values)
    np.save(os.path.join(np_dir, "sensor_ids.npy"), np.array(dist_df.index.tolist(), dtype=np.int64))
    log(f"  {method_name} numpy 输出: {np_dir}")


def save_default_local_alias(output_dir, sensor_info_df, dist_df, adj_df, graph_obj):
    save_intermediate(dist_df, "phase7_dist_matrix", output_dir)
    save_intermediate(adj_df, "phase7_adj_matrix", output_dir)
    save_intermediate(sensor_info_df, "phase7_sensor_info", output_dir)
    save_intermediate(graph_obj, "phase7_graph", output_dir)

    np_dir = output_dir
    os.makedirs(np_dir, exist_ok=True)
    np.save(os.path.join(np_dir, "distance_matrix.npy"), dist_df.values)
    np.save(os.path.join(np_dir, "adjacency_matrix.npy"), adj_df.values)
    np.save(os.path.join(np_dir, "sensor_ids.npy"), np.array(dist_df.index.tolist(), dtype=np.int64))
    log(f"  兼容输出已写入: {np_dir}")


def save_default_local_map_alias(output_dir):
    local_map_path = os.path.join(output_dir, "phase7_local_graph_map.html")
    default_map_path = os.path.join(output_dir, "phase7_graph_map.html")
    if os.path.exists(local_map_path):
        with open(local_map_path, "r", encoding="utf-8") as src:
            content = src.read()
        with open(default_map_path, "w", encoding="utf-8") as dst:
            dst.write(content)
        log(f"  兼容地图已写入: {default_map_path}")


def summarize_method(method_name, pair_results_df, edge_df, removed_df, adj_df):
    if pair_results_df.empty:
        log(f"\n  === {method_name.upper()} 图摘要 ===")
        log("  无候选边")
        return
    n_forward = int((pair_results_df["chosen_direction"] == "forward").sum())
    n_backward = int((pair_results_df["chosen_direction"] == "backward").sum())
    n_tie = int((pair_results_df["status"] == "tie").sum())
    n_unreachable = int((pair_results_df["status"] == "unreachable").sum())
    n_path_over = int((pair_results_df["status"] == "path_exceeds_threshold").sum())
    n_masked = int((pair_results_df["status"] == "masked_by_local_highway").sum())
    log(f"\n  === {method_name.upper()} 图摘要 ===")
    log(f"  定向边: {len(edge_df)} (forward={n_forward}, backward={n_backward})")
    log(f"  tie: {n_tie}, unreachable: {n_unreachable}, path_over: {n_path_over}, masked: {n_masked}")
    log(f"  三角删边: {len(removed_df)}")
    log(f"  邻接矩阵形状: {adj_df.shape}, 有向边数: {int(adj_df.values.sum())}")


def build_sensor_graph(output_dir="output", pair_threshold_m=5000,
                       route_cutoff_m=None, osrm_server="http://localhost:5000",
                       osrm_timeout=10.0, osrm_sleep_s=0.0,
                       max_route_distance_m=10000.0,
                       osrm_highway_mask=True,
                       sensor_types=("ML", "OR", "FR"),
                       save_subdir="sensor_graph"):
    started = time.time()
    log("=" * 70)
    log("Phase 7: 构建 ML 传感器图")
    log("=" * 70)

    graph = load_intermediate("phase1_graph", output_dir)
    edges_gdf = load_intermediate("phase1_edges_gdf", output_dir)
    nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)
    log(f"加载图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")

    save_output_dir = output_dir
    if save_subdir:
        save_output_dir = os.path.join(output_dir, save_subdir)
    os.makedirs(save_output_dir, exist_ok=True)
    log(f"结果输出目录: {save_output_dir}")

    sensors = collect_sensors(output_dir, sensor_types=sensor_types)
    if not sensors:
        raise RuntimeError("未找到任何已匹配的目标类型传感器")

    embedded_graph = graph.copy()
    embedded_graph, sensor_node_map = embed_sensors_into_graph(
        embedded_graph, edges_gdf, nodes_gdf, sensors
    )
    sensor_info_df = build_sensor_info_df(sensors, sensor_node_map, embedded_graph)
    candidate_df = build_candidate_pairs(sensor_info_df, pair_threshold_m)
    save_intermediate(candidate_df, "phase7_ml_candidate_pairs", save_output_dir)

    local_lookup, local_paths = compute_local_pair_paths(
        embedded_graph,
        sensor_node_map,
        candidate_df,
        route_cutoff_m=route_cutoff_m,
    )
    local_pair_results_df, local_raw_edge_df = build_directed_edges_from_pairs(
        candidate_df,
        local_lookup,
        method_name="local",
        max_route_distance_m=max_route_distance_m,
    )
    local_pruned_edge_df, local_removed_df = prune_transitive_longest_edges(local_raw_edge_df)
    local_pruned_edge_df = attach_local_edge_geometry(
        local_pruned_edge_df,
        local_paths,
        sensor_node_map,
        embedded_graph,
    )
    local_graph, local_dist_df, local_adj_df = build_graph_and_matrices(
        sensor_info_df,
        local_pruned_edge_df,
    )
    save_method_outputs(
        save_output_dir,
        "local",
        sensor_info_df,
        local_pair_results_df,
        local_raw_edge_df,
        local_pruned_edge_df,
        local_removed_df,
        local_dist_df,
        local_adj_df,
        local_graph,
    )
    save_default_local_alias(save_output_dir, sensor_info_df, local_dist_df, local_adj_df, local_graph)
    save_intermediate(local_paths, "phase7_local_paths", save_output_dir)
    draw_sensor_graph_map(save_output_dir, "local", sensor_info_df, local_pruned_edge_df, "Phase 7 Local Sensor Graph")
    save_default_local_map_alias(save_output_dir)
    summarize_method("local", local_pair_results_df, local_pruned_edge_df, local_removed_df, local_adj_df)

    osrm_lookup, osrm_extra = compute_osrm_pair_metrics(
        candidate_df,
        sensor_info_df,
        server=osrm_server,
        timeout=osrm_timeout,
        sleep_s=osrm_sleep_s,
    )
    osrm_pair_results_df, osrm_raw_edge_df = build_directed_edges_from_pairs(
        candidate_df,
        osrm_lookup,
        method_name="osrm",
        max_route_distance_m=max_route_distance_m,
        extra_lookup=osrm_extra,
    )
    if osrm_highway_mask:
        osrm_pair_results_df, osrm_raw_edge_df, _ = apply_osrm_highway_mask(
            osrm_pair_results_df,
            osrm_raw_edge_df,
            local_raw_edge_df,
        )
    osrm_pruned_edge_df, osrm_removed_df = prune_transitive_longest_edges(osrm_raw_edge_df)
    osrm_pruned_edge_df = attach_osrm_edge_geometry(osrm_pruned_edge_df)
    osrm_graph, osrm_dist_df, osrm_adj_df = build_graph_and_matrices(
        sensor_info_df,
        osrm_pruned_edge_df,
    )
    save_method_outputs(
        save_output_dir,
        "osrm",
        sensor_info_df,
        osrm_pair_results_df,
        osrm_raw_edge_df,
        osrm_pruned_edge_df,
        osrm_removed_df,
        osrm_dist_df,
        osrm_adj_df,
        osrm_graph,
    )
    draw_sensor_graph_map(save_output_dir, "osrm", sensor_info_df, osrm_pruned_edge_df, "Phase 7 OSRM Sensor Graph")
    summarize_method("osrm", osrm_pair_results_df, osrm_pruned_edge_df, osrm_removed_df, osrm_adj_df)

    log(f"\nPhase 7 完成, 总耗时: {time.time() - started:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建传感器有向图")
    parser.add_argument("output_dir", help="pipeline 输出目录")
    parser.add_argument(
        "--pair-threshold", type=float, default=5000.0,
        help="候选节点对的几何距离阈值, 单位米",
    )
    parser.add_argument(
        "--route-cutoff", type=float, default=None,
        help="本地最短路 Dijkstra 截断距离, 单位米; 默认不限",
    )
    parser.add_argument(
        "--osrm-server", type=str, default="http://localhost:5000",
        help="OSRM 服务地址",
    )
    parser.add_argument(
        "--osrm-timeout", type=float, default=10.0,
        help="单次 OSRM 请求超时, 秒",
    )
    parser.add_argument(
        "--osrm-sleep", type=float, default=0.0,
        help="OSRM 请求间隔, 秒",
    )
    parser.add_argument(
        "--max-route-distance", type=float, default=10000.0,
        help="最短路距离阈值(米), 超过该阈值则删除该边 (默认: 10000)",
    )
    parser.add_argument(
        "--osrm-highway-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否要求 OSRM 连边必须在本地 highway 图中可达同向边 (默认: 开启)",
    )
    parser.add_argument(
        "--sensor-types", type=str, default="ML,OR,FR",
        help="纳入构图的传感器类型, 逗号分隔 (默认: ML,OR,FR)",
    )
    parser.add_argument(
        "--save-subdir", type=str, default="sensor_graph",
        help="将 phase7 输出写入 output_dir 下的子目录 (默认: sensor_graph)",
    )

    args = parser.parse_args()
    sensor_types = [item.strip().upper() for item in args.sensor_types.split(",") if item.strip()]
    if not sensor_types:
        raise ValueError("--sensor-types 不能为空")

    build_sensor_graph(
        output_dir=args.output_dir,
        pair_threshold_m=args.pair_threshold,
        route_cutoff_m=args.route_cutoff,
        osrm_server=args.osrm_server,
        osrm_timeout=args.osrm_timeout,
        osrm_sleep_s=args.osrm_sleep,
        max_route_distance_m=args.max_route_distance,
        osrm_highway_mask=args.osrm_highway_mask,
        sensor_types=sensor_types,
        save_subdir=args.save_subdir,
    )

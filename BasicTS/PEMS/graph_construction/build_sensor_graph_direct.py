"""
Phase 7-direct: 构建传感器物理直接连接图
======================================
基于 Phase 1-4 的匹配结果，将已匹配的 ML/OR/FR 传感器嵌入 OSM 路网图，
然后为每个传感器搜索“沿有向路网向外遇到的第一个传感器”，构建物理直接连接图。

与 build_sensor_graph1.1.py 的区别:
- 1.1: cutoff 范围内所有可达传感器都连边
- direct: 只保留没有其他传感器夹在中间的直接物理邻居

输入:
- Phase 1: graph / nodes_gdf / edges_gdf
- Phase 3/4: ML / OR / FR mapping 结果

输出:
- phase7_dist_matrix / phase7_adj_matrix / phase7_sensor_info
- sensor_graph/*.npy
- sensor_graph_map.html

用法:
    python build_sensor_graph_direct.py output_dir --max-distance 20000
"""

import heapq
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import folium
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


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
        obj.to_csv(csv_path)
        log(f"  Saved {csv_path} ({obj.shape[0]} rows × {obj.shape[1]} cols)")


def parse_edge_key(edge_key):
    parts = edge_key.split("_")
    key = int(parts[-1])
    v = int(parts[-2])
    u = int("_".join(parts[:-2]))
    return u, v, key


def collect_matched_sensors(output_dir):
    log("Step 7.1: 收集已匹配传感器...")

    sensors = []

    try:
        ml_df = load_intermediate("phase3_ml_mapping", output_dir)
        for _, r in ml_df.iterrows():
            sensors.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": "ML",
                "fwy": r["fwy"],
                "dir": r["dir"],
                "lat": r["lat"],
                "lon": r["lon"],
            })
        log(f"  ML: {len(ml_df)} 个")
    except FileNotFoundError:
        log("  ML mapping 未找到", "WARN")

    try:
        or_df = load_intermediate("phase4_or_mapping", output_dir)
        for _, r in or_df.iterrows():
            sensors.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": "OR",
                "fwy": r["fwy"],
                "dir": r["dir"],
                "lat": r["lat"],
                "lon": r["lon"],
            })
        log(f"  OR: {len(or_df)} 个")
    except FileNotFoundError:
        log("  OR mapping 未找到", "WARN")

    try:
        fr_df = load_intermediate("phase4_fr_mapping", output_dir)
        for _, r in fr_df.iterrows():
            sensors.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": "FR",
                "fwy": r["fwy"],
                "dir": r["dir"],
                "lat": r["lat"],
                "lon": r["lon"],
            })
        log(f"  FR: {len(fr_df)} 个")
    except FileNotFoundError:
        log("  FR mapping 未找到", "WARN")

    log(f"  合计: {len(sensors)} 个已匹配传感器")
    return sensors


def embed_sensors_into_graph(G, edges_gdf, nodes_gdf, sensors):
    log("Step 7.2: 将传感器嵌入图...")

    edge_geom = {}
    for (u, v, key), row in edges_gdf.iterrows():
        ek = f"{u}_{v}_{key}"
        if row.geometry and not row.geometry.is_empty:
            edge_geom[ek] = row.geometry
        else:
            try:
                u_node = nodes_gdf.loc[u]
                v_node = nodes_gdf.loc[v]
                edge_geom[ek] = LineString([
                    (u_node["x"], u_node["y"]),
                    (v_node["x"], v_node["y"]),
                ])
            except KeyError:
                pass

    edge_sensors = defaultdict(list)
    for s in sensors:
        edge_sensors[s["link_id"]].append(s)

    sensor_node_map = {}
    embed_count = 0
    skip_count = 0

    for link_id, s_list in edge_sensors.items():
        geom = edge_geom.get(link_id)
        if geom is None:
            skip_count += len(s_list)
            continue

        try:
            u, v, key = parse_edge_key(link_id)
        except (ValueError, IndexError):
            skip_count += len(s_list)
            continue

        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            skip_count += len(s_list)
            continue

        if key not in edge_data:
            key = list(edge_data.keys())[0]

        original_data = dict(edge_data[key])
        edge_length_m = original_data.get("length", 0)
        total_geom_len = geom.length
        if total_geom_len == 0 or edge_length_m == 0:
            skip_count += len(s_list)
            continue

        for s in s_list:
            pt = Point(s["lon"], s["lat"])
            proj = geom.project(pt, normalized=True)
            proj = max(0.001, min(0.999, proj))
            s["_proj_ratio"] = proj

        s_list.sort(key=lambda x: x["_proj_ratio"])

        G.remove_edge(u, v, key)
        chain = [u]
        ratios = [0.0]

        for s in s_list:
            vnode = f"S_{s['sensor_id']}"
            proj_pt = geom.interpolate(s["_proj_ratio"], normalized=True)
            G.add_node(vnode, x=proj_pt.x, y=proj_pt.y, sensor_id=s["sensor_id"])
            sensor_node_map[s["sensor_id"]] = vnode
            chain.append(vnode)
            ratios.append(s["_proj_ratio"])

        chain.append(v)
        ratios.append(1.0)

        base_attrs = {k: val for k, val in original_data.items() if k not in ("length", "geometry")}
        for i in range(len(chain) - 1):
            seg_u = chain[i]
            seg_v = chain[i + 1]
            seg_ratio = ratios[i + 1] - ratios[i]
            attrs = dict(base_attrs)
            attrs["length"] = edge_length_m * seg_ratio
            G.add_edge(seg_u, seg_v, **attrs)

        embed_count += len(s_list)

    log(f"  嵌入成功: {embed_count} 个 sensor")
    if skip_count > 0:
        log(f"  跳过: {skip_count} 个 (无几何或 edge 不存在)", "WARN")
    log(f"  图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    return G, sensor_node_map


def path_nodes_to_coords(G, path_nodes):
    if not path_nodes or len(path_nodes) < 2:
        return []

    coords = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue

        attrs = edge_data[list(edge_data.keys())[0]]
        geom = attrs.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            seg = [(lat, lon) for lon, lat in geom.coords]
        else:
            node_u = G.nodes[u]
            node_v = G.nodes[v]
            seg = [(node_u["y"], node_u["x"]), (node_v["y"], node_v["x"])]

        if coords and seg and coords[-1] == seg[0]:
            seg = seg[1:]
        coords.extend(seg)

    return coords


def find_direct_sensor_neighbors(G, sensor_node_map, max_distance_m=None):
    """
    从每个 sensor 出发，在有向图上做 Dijkstra。
    一旦遇到另一个 sensor，就记为“直接邻居”，并停止从该节点继续扩展。

    这样保留的是:
    - 每个方向分支上遇到的第一个 sensor
    - 不会穿过中间 sensor 再去连接更远 sensor
    """
    log("Step 7.3: 计算物理直接连接图...")

    sensor_ids = sorted(sensor_node_map.keys())
    node_to_sid = {node: sid for sid, node in sensor_node_map.items()}
    direct_edges = []
    path_lookup = {}

    t0 = time.time()
    for idx, source_sid in enumerate(sensor_ids, start=1):
        source_node = sensor_node_map[source_sid]
        heap = [(0.0, source_node, [source_node])]
        best_dist = {source_node: 0.0}
        found_targets = {}

        while heap:
            dist_u, node_u, path_u = heapq.heappop(heap)
            if dist_u > best_dist.get(node_u, np.inf):
                continue

            if node_u != source_node and node_u in node_to_sid:
                target_sid = node_to_sid[node_u]
                if target_sid not in found_targets:
                    found_targets[target_sid] = (dist_u, path_u)
                continue

            for _, node_v, edge_attrs in G.out_edges(node_u, data=True):
                edge_length = float(edge_attrs.get("length", 0.0))
                if edge_length < 0:
                    continue

                new_dist = dist_u + edge_length
                if max_distance_m is not None and new_dist > max_distance_m:
                    continue
                if new_dist >= best_dist.get(node_v, np.inf):
                    continue

                best_dist[node_v] = new_dist
                heapq.heappush(heap, (new_dist, node_v, path_u + [node_v]))

        for target_sid, (dist_m, path_nodes) in found_targets.items():
            direct_edges.append({
                "source_id": source_sid,
                "target_id": target_sid,
                "weight_m": float(dist_m),
            })
            path_lookup[(source_sid, target_sid)] = path_nodes

        if idx % 200 == 0:
            log(f"    进度: {idx}/{len(sensor_ids)} ({time.time() - t0:.1f}s)")

    if direct_edges:
        edge_df = pd.DataFrame(direct_edges).drop_duplicates(
            subset=["source_id", "target_id"]
        ).sort_values(["source_id", "target_id"]).reset_index(drop=True)
    else:
        edge_df = pd.DataFrame(columns=["source_id", "target_id", "weight_m"])

    log(f"  直接有向边数: {len(edge_df)}")
    return edge_df, path_lookup


def build_matrices(sensor_ids, edge_df):
    sid_to_idx = {sid: idx for idx, sid in enumerate(sensor_ids)}
    n = len(sensor_ids)

    dist_mat = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0.0)
    adj_mat = np.zeros((n, n), dtype=np.int8)

    for _, row in edge_df.iterrows():
        i = sid_to_idx[int(row["source_id"])]
        j = sid_to_idx[int(row["target_id"])]
        dist_mat[i, j] = float(row["weight_m"])
        adj_mat[i, j] = 1

    dist_df = pd.DataFrame(dist_mat, index=sensor_ids, columns=sensor_ids)
    adj_df = pd.DataFrame(adj_mat, index=sensor_ids, columns=sensor_ids)
    return dist_df, adj_df


def save_results(dist_df, adj_df, edge_df, sensor_node_map, sensors, output_dir):
    log("Step 7.4: 保存结果...")

    save_intermediate(dist_df, "phase7_dist_matrix", output_dir)
    save_intermediate(adj_df, "phase7_adj_matrix", output_dir)
    save_intermediate(edge_df, "phase7_edges_direct", output_dir)

    sensor_info = []
    for s in sensors:
        if s["sensor_id"] in sensor_node_map:
            sensor_info.append({
                "sensor_id": s["sensor_id"],
                "type": s["type"],
                "fwy": s["fwy"],
                "dir": s["dir"],
                "lat": s["lat"],
                "lon": s["lon"],
                "link_id": s["link_id"],
                "graph_node": sensor_node_map[s["sensor_id"]],
            })
    sensor_info_df = pd.DataFrame(sensor_info)
    save_intermediate(sensor_info_df, "phase7_sensor_info", output_dir)

    np_dir = os.path.join(output_dir, "sensor_graph")
    os.makedirs(np_dir, exist_ok=True)

    sensor_ids = sorted(sensor_node_map.keys())
    np.save(os.path.join(np_dir, "distance_matrix.npy"), dist_df.values)
    np.save(os.path.join(np_dir, "adjacency_matrix.npy"), adj_df.values)
    np.save(os.path.join(np_dir, "sensor_ids.npy"), np.array(sensor_ids))
    log(f"  numpy 文件保存到: {np_dir}/")

    log(f"\n  === 最终输出 ===")
    log(f"  sensor 数量: {len(sensor_ids)}")
    log(f"  邻接矩阵: {adj_df.shape}")
    log(f"  有向边数: {adj_df.values.sum()}")


def draw_sensor_graph_map(edge_df, path_lookup, G, sensor_node_map, sensors, edges_gdf, output_dir):
    log("Step 7.5: 绘制传感器图地图...")

    sensor_dict = {}
    for s in sensors:
        if s["sensor_id"] in sensor_node_map:
            sensor_dict[s["sensor_id"]] = s

    sensor_ids = sorted(sensor_node_map.keys())
    lats = [sensor_dict[sid]["lat"] for sid in sensor_ids if sid in sensor_dict]
    lons = [sensor_dict[sid]["lon"] for sid in sensor_ids if sid in sensor_dict]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    fg_road = folium.FeatureGroup(name="路网底图", show=True)
    for _, edge in edges_gdf.iterrows():
        if edge.geometry and edge.geometry.geom_type == "LineString":
            coords = [(c[1], c[0]) for c in edge.geometry.coords]
            folium.PolyLine(coords, weight=1.5, color="gray", opacity=0.25).add_to(fg_road)
    fg_road.add_to(m)

    fg_edges = folium.FeatureGroup(name=f"直接连接边 ({len(edge_df)})", show=True)
    for _, row in edge_df.iterrows():
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        path_nodes = path_lookup.get((source_id, target_id), [])
        coords = path_nodes_to_coords(G, path_nodes)
        if len(coords) < 2 and source_id in sensor_dict and target_id in sensor_dict:
            coords = [
                (sensor_dict[source_id]["lat"], sensor_dict[source_id]["lon"]),
                (sensor_dict[target_id]["lat"], sensor_dict[target_id]["lon"]),
            ]

        folium.PolyLine(
            coords,
            weight=2.5,
            color="#cc5500",
            opacity=0.8,
            tooltip=f"{source_id}->{target_id} ({float(row['weight_m']):.0f}m)",
        ).add_to(fg_edges)
    fg_edges.add_to(m)

    type_colors = {"ML": "blue", "OR": "green", "FR": "red"}
    for stype in ["ML", "OR", "FR"]:
        type_sensors = [s for s in sensors if s["type"] == stype and s["sensor_id"] in sensor_node_map]
        fg = folium.FeatureGroup(name=f"{stype} sensors ({len(type_sensors)})", show=True)
        for s in type_sensors:
            color = type_colors.get(stype, "gray")
            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=5,
                color="black",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"{s['sensor_id']} ({stype}) Fwy{s['fwy']}{s['dir']}",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_path = os.path.join(output_dir, "sensor_graph_map.html")
    m.save(map_path)
    log(f"  地图已保存: {map_path}")


def build_sensor_graph(output_dir="output", max_distance_m=None):
    t0 = time.time()
    log("=" * 70)
    log("Phase 7-direct: 构建物理直接连接传感器图")
    log("=" * 70)

    log("加载中间结果...")
    G = load_intermediate("phase1_graph", output_dir)
    edges_gdf = load_intermediate("phase1_edges_gdf", output_dir)
    nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)

    sensors = collect_matched_sensors(output_dir)
    if not sensors:
        log("无已匹配 sensor, 退出", "ERROR")
        return

    G_copy = G.copy()
    G_copy, sensor_node_map = embed_sensors_into_graph(G_copy, edges_gdf, nodes_gdf, sensors)

    edge_df, path_lookup = find_direct_sensor_neighbors(
        G_copy,
        sensor_node_map,
        max_distance_m=max_distance_m,
    )

    sensor_ids = sorted(sensor_node_map.keys())
    dist_df, adj_df = build_matrices(sensor_ids, edge_df)
    save_results(dist_df, adj_df, edge_df, sensor_node_map, sensors, output_dir)
    draw_sensor_graph_map(edge_df, path_lookup, G_copy, sensor_node_map, sensors, edges_gdf, output_dir)

    elapsed = time.time() - t0
    log(f"\nPhase 7-direct 总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7-direct: 构建传感器物理直接连接图"
    )
    parser.add_argument(
        "output_dir",
        help="Pipeline 输出目录 (含 Phase 1-4 中间结果)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="搜索直接邻居的最大距离上限（米），默认不限",
    )

    args = parser.parse_args()
    build_sensor_graph(args.output_dir, args.max_distance)

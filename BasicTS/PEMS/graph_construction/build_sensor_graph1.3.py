"""
Phase 7: 构建传感器图 (Sensor Graph)
======================================
基于 Phase 1-4 的匹配结果, 将已匹配的 ML/OR/FR 传感器嵌入 OSM 路网图,
计算传感器之间的有向最短路距离, 构建带距离阈值截断的邻接矩阵.

输入: Phase 1 的 graph + edges_gdf, Phase 3/4 的 mapping 结果
输出: 距离矩阵 (N×N), 邻接矩阵 (0/1), 传感器 ID 列表

用法:
    python build_sensor_graph.py output/ --cutoff 5000
"""

import os
import pickle
import time
import math
import numpy as np
import pandas as pd
import networkx as nx
import folium
from shapely.geometry import Point, LineString
from collections import defaultdict
from datetime import datetime


# =============================================================================
# 工具函数
# =============================================================================
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
    """解析 edge_key='u_v_key' → (u, v, key), 支持负 ID."""
    parts = edge_key.split("_")
    # 负 ID 如 -1 会被 split 为 ['', '1', ...]
    # 策略: 从右边取 key (最后一个), v (倒数第二个), 剩余拼接为 u
    key = int(parts[-1])
    v = int(parts[-2])
    u = int("_".join(parts[:-2]))
    return u, v, key


# =============================================================================
# Step 7.1: 收集已匹配传感器
# =============================================================================
def collect_matched_sensors(output_dir):
    """合并 ML/OR/FR 匹配结果."""
    log("Step 7.1: 收集已匹配传感器...")

    sensors = []

    # ML
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

    # OR
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

    # FR
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


# =============================================================================
# Step 7.2: 将传感器嵌入图
# =============================================================================
def embed_sensors_into_graph(G, edges_gdf, nodes_gdf, sensors):
    """
    在图 G 中为每个 sensor 插入虚拟节点, 拆分所在 edge.

    对于同一 edge 上的多个 sensor, 按投影位置排序后链式拆分.
    虚拟节点 ID 格式: "S_{sensor_id}" (字符串, 避免与 OSM 节点冲突).

    返回:
        G: 修改后的图 (原图被修改)
        sensor_node_map: {sensor_id: virtual_node_id}
    """
    log("Step 7.2: 将传感器嵌入图...")

    # 构建 edge_key -> geometry 映射
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

    # 按 link_id 分组 (处理同一 edge 上多个 sensor)
    edge_sensors = defaultdict(list)
    for s in sensors:
        edge_sensors[s["link_id"]].append(s)

    sensor_node_map = {}  # sensor_id -> virtual_node_id
    embed_count = 0
    skip_count = 0

    for link_id, s_list in edge_sensors.items():
        geom = edge_geom.get(link_id)
        if geom is None:
            skip_count += len(s_list)
            continue

        # 解析 edge_key → (u, v, key)
        try:
            u, v, key = parse_edge_key(link_id)
        except (ValueError, IndexError):
            skip_count += len(s_list)
            continue

        # 检查 edge 是否存在
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            skip_count += len(s_list)
            continue

        # 找到正确的 key
        if key not in edge_data:
            # 可能 key 不匹配, 取第一个
            key = list(edge_data.keys())[0]

        original_data = dict(edge_data[key])
        edge_length_m = original_data.get("length", 0)
        total_geom_len = geom.length  # 度单位

        if total_geom_len == 0 or edge_length_m == 0:
            skip_count += len(s_list)
            continue

        # 计算每个 sensor 在 edge 上的投影比例
        for s in s_list:
            pt = Point(s["lon"], s["lat"])
            proj = geom.project(pt, normalized=True)  # 0~1
            proj = max(0.001, min(0.999, proj))  # 避免完全重合端点
            s["_proj_ratio"] = proj

        # 按投影位置排序
        s_list.sort(key=lambda x: x["_proj_ratio"])

        # 链式拆分: u → S1 → S2 → ... → v
        # 删除原 edge
        G.remove_edge(u, v, key)

        # 构建节点链: [u, S1, S2, ..., v]
        chain = [u]
        ratios = [0.0]

        for s in s_list:
            vnode = f"S_{s['sensor_id']}"
            # 在图中添加虚拟节点 (用 sensor 投影点的坐标)
            proj_pt = geom.interpolate(s["_proj_ratio"], normalized=True)
            G.add_node(vnode, x=proj_pt.x, y=proj_pt.y, sensor_id=s["sensor_id"])
            sensor_node_map[s["sensor_id"]] = vnode
            chain.append(vnode)
            ratios.append(s["_proj_ratio"])

        chain.append(v)
        ratios.append(1.0)

        # 添加子段 edge
        base_attrs = {k: val for k, val in original_data.items()
                      if k not in ("length", "geometry")}

        for i in range(len(chain) - 1):
            seg_u = chain[i]
            seg_v = chain[i + 1]
            seg_ratio = ratios[i + 1] - ratios[i]
            seg_length = edge_length_m * seg_ratio

            attrs = dict(base_attrs)
            attrs["length"] = seg_length
            G.add_edge(seg_u, seg_v, **attrs)

        embed_count += len(s_list)

    log(f"  嵌入成功: {embed_count} 个 sensor")
    if skip_count > 0:
        log(f"  跳过: {skip_count} 个 (无几何或 edge 不存在)", "WARN")
    log(f"  图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    return G, sensor_node_map


# =============================================================================
# Step 7.3: 计算最短路距离矩阵
# =============================================================================
def compute_distance_matrix(G, sensor_node_map, cutoff_m=5000):
    """
    对每个 sensor 虚拟节点, 运行带截断的 Dijkstra,
    构建 sensor×sensor 的有向距离矩阵.

    参数:
        G: 嵌入 sensor 后的有向图
        sensor_node_map: {sensor_id: virtual_node_id}
        cutoff_m: 距离截断阈值 (米)

    返回:
        dist_df: N×N 距离矩阵 (DataFrame, index/columns = sensor_id)
        adj_df: N×N 邻接矩阵 (0/1)
    """
    log(f"Step 7.3: 计算最短路距离矩阵 (cutoff={cutoff_m}m)...")

    sensor_ids = sorted(sensor_node_map.keys())
    n = len(sensor_ids)
    sid_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}

    # 反向映射: virtual_node_id -> sensor_id
    node_to_sid = {v: k for k, v in sensor_node_map.items()}

    # sensor 虚拟节点集合 (用于快速查找)
    sensor_nodes_set = set(sensor_node_map.values())

    log(f"  sensor 数量: {n}")
    log(f"  图节点: {G.number_of_nodes()}, 图边: {G.number_of_edges()}")

    # 初始化距离矩阵 (inf)
    dist_mat = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0.0)

    t0 = time.time()
    edge_count = 0

    for idx, sid in enumerate(sensor_ids):
        vnode = sensor_node_map[sid]

        if vnode not in G:
            continue

        # 带截断的 Dijkstra (只探索 cutoff 范围)
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, vnode, weight="length", cutoff=cutoff_m
            )
        except nx.NetworkXError:
            continue

        # 填充距离矩阵 (只关心到其他 sensor 的距离)
        for target_node, dist in lengths.items():
            if target_node in node_to_sid:
                target_sid = node_to_sid[target_node]
                if target_sid != sid:
                    j = sid_to_idx[target_sid]
                    dist_mat[idx, j] = dist
                    edge_count += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            log(f"    进度: {idx+1}/{n} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    log(f"  Dijkstra 完成 ({elapsed:.1f}s)")

    # 统计
    reachable = np.isfinite(dist_mat) & (dist_mat > 0)
    log(f"  可达 sensor 对数: {reachable.sum()} / {n*(n-1)}")
    if reachable.sum() > 0:
        valid_dists = dist_mat[reachable]
        log(f"  距离统计: min={valid_dists.min():.0f}m, "
            f"mean={valid_dists.mean():.0f}m, "
            f"median={np.median(valid_dists):.0f}m, "
            f"max={valid_dists.max():.0f}m")

    # 构建 DataFrame
    dist_df = pd.DataFrame(dist_mat, index=sensor_ids, columns=sensor_ids)

    # 邻接矩阵: 可达且在 cutoff 内 → 1
    adj_mat = (np.isfinite(dist_mat) & (dist_mat > 0)).astype(np.int8)
    adj_df = pd.DataFrame(adj_mat, index=sensor_ids, columns=sensor_ids)

    log(f"  邻接矩阵非零: {adj_mat.sum()} 条有向边")
    log(f"  平均每个 sensor 的出度: {adj_mat.sum(axis=1).mean():.1f}")
    log(f"  平均每个 sensor 的入度: {adj_mat.sum(axis=0).mean():.1f}")

    # 检查孤立节点 (无出边也无入边)
    isolated = ((adj_mat.sum(axis=0) == 0) & (adj_mat.sum(axis=1) == 0))
    if isolated.sum() > 0:
        log(f"  WARNING: {isolated.sum()} 个 sensor 完全孤立 (无邻居)", "WARN")

    return dist_df, adj_df


# =============================================================================
# Step 7.4: 输出
# =============================================================================
def save_results(dist_df, adj_df, sensor_node_map, sensors, G_embed, output_dir):
    """保存距离矩阵、邻接矩阵和 sensor 信息."""
    log("Step 7.4: 保存结果...")

    save_intermediate(dist_df, "phase7_dist_matrix", output_dir)
    save_intermediate(adj_df, "phase7_adj_matrix", output_dir)

    # sensor 信息表 (含投影坐标)
    sensor_info = []
    for s in sensors:
        sid = s["sensor_id"]
        if sid not in sensor_node_map:
            continue
        vnode = sensor_node_map[sid]
        if vnode in G_embed:
            proj_lon = G_embed.nodes[vnode].get("x", s["lon"])
            proj_lat = G_embed.nodes[vnode].get("y", s["lat"])
        else:
            proj_lon, proj_lat = s["lon"], s["lat"]
        sensor_info.append({
            "sensor_id": sid,
            "type": s["type"],
            "fwy": s["fwy"],
            "dir": s["dir"],
            "lat": s["lat"],
            "lon": s["lon"],
            "proj_lat": proj_lat,
            "proj_lon": proj_lon,
            "link_id": s["link_id"],
            "graph_node": vnode,
        })
    sensor_info_df = pd.DataFrame(sensor_info)
    save_intermediate(sensor_info_df, "phase7_sensor_info", output_dir)

    # 保存 numpy 格式 (方便 BasicTS 等框架直接读取)
    np_dir = os.path.join(output_dir, "sensor_graph")
    os.makedirs(np_dir, exist_ok=True)

    sensor_ids = sorted(sensor_node_map.keys())
    np.save(os.path.join(np_dir, "distance_matrix.npy"), dist_df.values)
    np.save(os.path.join(np_dir, "adjacency_matrix.npy"), adj_df.values)
    np.save(os.path.join(np_dir, "sensor_ids.npy"), np.array(sensor_ids))
    log(f"  numpy 文件保存到: {np_dir}/")

    log(f"\n  === 最终输出 ===")
    log(f"  sensor 数量: {len(sensor_ids)}")
    log(f"  距离矩阵: {dist_df.shape}")
    log(f"  邻接矩阵: {adj_df.shape}")
    log(f"  有向边数: {adj_df.values.sum()}")


# =============================================================================
# Step 7.5: 可视化传感器图
# =============================================================================
def draw_sensor_graph_map(dist_df, adj_df, sensor_node_map, sensors,
                          G_embed, edges_gdf, output_dir):
    """
    在 Folium 地图上绘制传感器图.

    使用投影后的 sensor 坐标 (虚拟节点在路网上的位置).

    图层:
        - 路网底图 (灰色)
        - sensor 间有向连接 (箭头线, 颜色按距离渐变)
        - sensor 节点 (ML=蓝, OR=绿, FR=红)
    """

    log("Step 7.5: 绘制传感器图地图...")

    # sensor_id -> 信息 (含投影坐标)
    sensor_dict = {}
    for s in sensors:
        sid = s["sensor_id"]
        if sid not in sensor_node_map:
            continue
        vnode = sensor_node_map[sid]
        # 从嵌入图的虚拟节点读取投影坐标
        if vnode in G_embed:
            proj_x = G_embed.nodes[vnode].get("x", s["lon"])  # x=lon
            proj_y = G_embed.nodes[vnode].get("y", s["lat"])  # y=lat
        else:
            proj_x, proj_y = s["lon"], s["lat"]
        sensor_dict[sid] = {
            **s,
            "proj_lat": proj_y,
            "proj_lon": proj_x,
        }

    sensor_ids = sorted(sensor_node_map.keys())

    # 地图中心
    lats = [sensor_dict[sid]["proj_lat"] for sid in sensor_ids if sid in sensor_dict]
    lons = [sensor_dict[sid]["proj_lon"] for sid in sensor_ids if sid in sensor_dict]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # --- 层 1: 路网底图 ---
    fg_road = folium.FeatureGroup(name="路网底图", show=True)
    for _, edge in edges_gdf.iterrows():
        if edge.geometry and edge.geometry.geom_type == "LineString":
            coords = [(c[1], c[0]) for c in edge.geometry.coords]
            folium.PolyLine(
                coords, weight=1.5, color="gray", opacity=0.25
            ).add_to(fg_road)
    fg_road.add_to(m)

    # --- 层 2: 有向连接线 ---
    fg_edges = folium.FeatureGroup(name=f"连接边 ({int(adj_df.values.sum())})", show=True)

    # 距离→颜色: 近=深绿, 远=深红
    valid_dists = dist_df.values[np.isfinite(dist_df.values) & (dist_df.values > 0)]
    if len(valid_dists) > 0:
        d_min = valid_dists.min()
        d_max = valid_dists.max()
    else:
        d_min, d_max = 0, 5000

    def dist_to_color(d):
        """距离映射到颜色: 绿(近) → 黄(中) → 红(远)."""
        if d_max <= d_min:
            ratio = 0.5
        else:
            ratio = (d - d_min) / (d_max - d_min)
        ratio = max(0, min(1, ratio))
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f"#{r:02x}{g:02x}00"

    def arrow_head(lat1, lon1, lat2, lon2, size=0.001):
        """在终点处生成简易箭头三角形坐标."""
        dx = lon2 - lon1
        dy = lat2 - lat1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-8:
            return None
        # 单位向量
        ux, uy = dx / length, dy / length
        # 垂直向量
        px, py = -uy, ux
        # 箭头两翼
        tip = (lat2, lon2)
        left = (lat2 - size * uy - size * 0.4 * px,
                lon2 - size * ux - size * 0.4 * py)
        right = (lat2 - size * uy + size * 0.4 * px,
                 lon2 - size * ux + size * 0.4 * py)
        return [left, tip, right]

    edge_count = 0
    for i, sid_from in enumerate(sensor_ids):
        if sid_from not in sensor_dict:
            continue
        s_from = sensor_dict[sid_from]

        for j, sid_to in enumerate(sensor_ids):
            if i == j or adj_df.values[i, j] == 0:
                continue
            if sid_to not in sensor_dict:
                continue
            s_to = sensor_dict[sid_to]

            dist_m = dist_df.values[i, j]
            color = dist_to_color(dist_m)

            lat1, lon1 = s_from["proj_lat"], s_from["proj_lon"]
            lat2, lon2 = s_to["proj_lat"], s_to["proj_lon"]

            # 连接线
            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                weight=1.5, color=color, opacity=0.6,
                tooltip=(f"{sid_from}→{sid_to} "
                         f"({dist_m:.0f}m)"),
            ).add_to(fg_edges)

            # 箭头
            arrow = arrow_head(lat1, lon1, lat2, lon2)
            if arrow:
                folium.Polygon(
                    locations=arrow,
                    color=color, fill=True, fill_color=color,
                    fill_opacity=0.8, weight=1,
                ).add_to(fg_edges)

            edge_count += 1

    fg_edges.add_to(m)
    log(f"  绘制 {edge_count} 条有向边")

    # --- 层 3: sensor 节点 ---
    type_colors = {"ML": "blue", "OR": "green", "FR": "red"}
    for stype in ["ML", "OR", "FR"]:
        type_sids = [s["sensor_id"] for s in sensors
                     if s["type"] == stype and s["sensor_id"] in sensor_dict]
        fg = folium.FeatureGroup(
            name=f"{stype} sensors ({len(type_sids)})", show=True
        )
        for sid in type_sids:
            s = sensor_dict[sid]
            idx = sensor_ids.index(sid) if sid in sensor_ids else -1
            out_degree = int(adj_df.values[idx].sum()) if idx >= 0 else 0
            in_degree = int(adj_df.values[:, idx].sum()) if idx >= 0 else 0

            popup = (
                f"<b>Sensor {sid}</b> ({stype})<br>"
                f"Fwy {s['fwy']} {s['dir']}<br>"
                f"Link: {s['link_id']}<br>"
                f"出度: {out_degree}, 入度: {in_degree}<br>"
                f"投影坐标: ({s['proj_lat']:.6f}, {s['proj_lon']:.6f})<br>"
                f"原始坐标: ({s['lat']:.6f}, {s['lon']:.6f})"
            )

            color = type_colors.get(stype, "gray")
            folium.CircleMarker(
                location=[s["proj_lat"], s["proj_lon"]],
                radius=5, color="black", weight=1.5,
                fill=True, fill_color=color, fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=300),
                tooltip=f"{sid} ({stype}) Fwy{s['fwy']}{s['dir']} "
                        f"out={out_degree} in={in_degree}",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    map_path = os.path.join(output_dir, "sensor_graph_map.html")
    m.save(map_path)
    log(f"  地图已保存: {map_path}")


# =============================================================================
# 主流程
# =============================================================================
def build_sensor_graph(output_dir="output", cutoff_m=5000):
    """
    构建传感器图.

    参数:
        output_dir: pipeline 输出目录 (含 Phase 1-4 中间结果)
        cutoff_m: 最短路截断距离 (米, 默认 5000)
    """
    t0 = time.time()
    log("=" * 70)
    log("Phase 7: 构建传感器图 (Sensor Graph)")
    log("=" * 70)

    # --- 加载中间结果 ---
    log("加载中间结果...")
    G = load_intermediate("phase1_graph", output_dir)
    edges_gdf = load_intermediate("phase1_edges_gdf", output_dir)
    nodes_gdf = load_intermediate("phase1_nodes_gdf", output_dir)

    log(f"  图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    log(f"  edges_gdf: {len(edges_gdf)} 行")

    # --- Step 7.1 ---
    sensors = collect_matched_sensors(output_dir)
    if not sensors:
        log("无已匹配 sensor, 退出", "ERROR")
        return

    # --- Step 7.2 ---
    G_copy = G.copy()  # 不修改原图
    G_copy, sensor_node_map = embed_sensors_into_graph(
        G_copy, edges_gdf, nodes_gdf, sensors
    )

    # --- Step 7.3 ---
    dist_df, adj_df = compute_distance_matrix(G_copy, sensor_node_map, cutoff_m)

    # --- Step 7.4 ---
    save_results(dist_df, adj_df, sensor_node_map, sensors, G_copy, output_dir)

    # --- Step 7.5 ---
    draw_sensor_graph_map(dist_df, adj_df, sensor_node_map, sensors,
                          G_copy, edges_gdf, output_dir)

    elapsed = time.time() - t0
    log(f"\nPhase 7 总耗时: {elapsed:.1f}s")


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7: 构建传感器图 (最短路距离)"
    )
    parser.add_argument(
        "output_dir",
        help="Pipeline 输出目录 (含 Phase 1-4 中间结果)",
    )
    parser.add_argument(
        "--cutoff", type=float, default=5000,
        help="最短路截断距离 (米, 默认: 5000)",
    )

    args = parser.parse_args()
    build_sensor_graph(args.output_dir, args.cutoff)

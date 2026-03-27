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
def phase0_load_metadata(metadata_file, output_dir, buffer_deg=0.02):
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

    # 每条高速公路的传感器分布保存为诊断 csv
    fwy_summary = (
        sensors_df.groupby(["Fwy", "Dir", "Type"])
        .agg(count=("ID", "count"),
             lat_min=("Latitude", "min"),
             lat_max=("Latitude", "max"),
             lon_min=("Longitude", "min"),
             lon_max=("Longitude", "max"))
        .reset_index()
    )
    save_intermediate(fwy_summary, "phase0_fwy_summary", output_dir)

    log("Phase 0 完成\n")
    return sensors_df, bbox_dict


# =============================================================================
# Phase 1: 提取 OSM 路网, 构建关联矩阵
# =============================================================================
def phase1_extract_network(bbox_dict, output_dir, max_edge_length_m=2000):
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

    # PBF 文件路径 (修改为你的实际路径)
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

    # 过滤: 只保留 PeMS 中存在的高速编号
    extra_fwy = set(fwy_groups.keys()) - pems_fwy_numbers
    if extra_fwy:
        log(f"  OSM 中有但 PeMS 中无的高速编号 (将过滤): {sorted(extra_fwy)}")
        for f in extra_fwy:
            del fwy_groups[f]

    missing_fwy = pems_fwy_numbers - set(fwy_groups.keys())
    if missing_fwy:
        log(f"  WARNING: PeMS 中有但 OSM 中未找到的高速编号: {sorted(missing_fwy)}", "WARN")

    log(f"  过滤后保留 {len(fwy_groups)} 条高速公路")
    for fwy_num in sorted(fwy_groups.keys()):
        log(f"    Fwy {fwy_num}: {len(fwy_groups[fwy_num])} links")

    # --- Step 2.2: 方向判定 (仅使用 Caltrans SHN Lines) ---
    log("\nStep 2.2: 方向判定 (Caltrans SHN Lines)...")
    log(f"  SHN shapefile: {shn_shapefile_path}")

    # 加载 SHN Lines
    shn_gdf = gpd.read_file(shn_shapefile_path)
    shn_gdf = shn_gdf.to_crs(epsg=4326)  # 转 WGS84 与 OSM 一致
    log(f"  SHN 全州记录数: {len(shn_gdf)}")

    # 按 Route 建索引 (只保留 PeMS 中存在的 Route)
    shn_by_route = {}
    for route_num in fwy_groups.keys():
        route_lines = shn_gdf[shn_gdf["Route"] == route_num].copy()
        if len(route_lines) > 0:
            shn_by_route[route_num] = route_lines
    log(f"  匹配到 SHN Route 的高速: {sorted(shn_by_route.keys())}")
    missing_shn = set(fwy_groups.keys()) - set(shn_by_route.keys())
    if missing_shn:
        log(f"  WARNING: 以下高速无 SHN Route 数据: {sorted(missing_shn)}", "WARN")

    # 构建 edge_key -> geometry 映射 (从 edges_gdf)
    edge_geom = {}
    for (u, v, key), row in edges_gdf.iterrows():
        ek = f"{u}_{v}_{key}"
        edge_geom[ek] = row.geometry

    # 对每条 edge 匹配 SHN 方向
    dir_stats = {"from_shn": 0, "no_shn_route": 0, "no_geometry": 0, "shn_match_failed": 0}

    for fwy_num, links in fwy_groups.items():
        if fwy_num not in shn_by_route:
            for link_info in links:
                link_info["dir"] = None
                dir_stats["no_shn_route"] += 1
            continue

        route_shn = shn_by_route[fwy_num]

        for link_info in links:
            ek = link_info["id"]
            geom = edge_geom.get(ek)

            if geom is None or geom.is_empty:
                link_info["dir"] = None
                dir_stats["no_geometry"] += 1
                continue

            # 计算 edge 几何中点
            centroid = geom.interpolate(0.5, normalized=True)

            # 计算到每条 SHN 线段的距离, 取最近
            dists = route_shn.geometry.distance(centroid)
            nearest_idx = dists.idxmin()
            nearest_dist = dists[nearest_idx]
            shn_dir = route_shn.loc[nearest_idx, "Direction"]

            # 距离合理性检查 (>0.01 度 约 1.1km, 认为匹配失败)
            if nearest_dist > 0.01:
                link_info["dir"] = None
                dir_stats["shn_match_failed"] += 1
                continue

            pems_dir = SHN_DIR_TO_PEMS.get(shn_dir)
            if pems_dir:
                link_info["dir"] = pems_dir
                dir_stats["from_shn"] += 1
            else:
                link_info["dir"] = None
                dir_stats["shn_match_failed"] += 1

    log(f"  方向判定结果: SHN={dir_stats['from_shn']}, "
        f"无SHN Route={dir_stats['no_shn_route']}, "
        f"无几何={dir_stats['no_geometry']}, "
        f"SHN匹配失败={dir_stats['shn_match_failed']}")

    # 各高速各方向的 link 数
    for fwy_num in sorted(fwy_groups.keys()):
        dir_counts = defaultdict(int)
        for l in fwy_groups[fwy_num]:
            d = l["dir"] if l["dir"] else "unknown"
            dir_counts[d] += 1
        log(f"    Fwy {fwy_num}: {dict(dir_counts)}")

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

    # edge_key -> edge geometry (用于高亮匹配路段, edge_key 保证唯一)
    link_geom = {}
    for (u, v, key), edge in edges_gdf.iterrows():
        if edge.geometry and edge.geometry.geom_type == "LineString":
            edge_key = f"{u}_{v}_{key}"
            link_geom[edge_key] = edge.geometry

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

    # --- 绘制底图路网 (灰色) ---
    fg_road = folium.FeatureGroup(name="路网底图", show=True)
    for _, edge in edges_gdf.iterrows():
        if edge.geometry and edge.geometry.geom_type == "LineString":
            coords = [(c[1], c[0]) for c in edge.geometry.coords]
            folium.PolyLine(coords, weight=2, color="gray", opacity=0.3).add_to(fg_road)
    fg_road.add_to(m)

    # --- 绘制匹配路段高亮线 ---
    fg_highlight = folium.FeatureGroup(name="匹配路段 (高亮)", show=True)
    highlight_colors = {"ML": "blue", "OR": "green", "FR": "red"}
    drawn_links = set()

    for sid, info in matched_info.items():
        lid = str(info["link_id"])
        if lid in drawn_links:
            continue
        drawn_links.add(lid)
        geom = link_geom.get(lid)
        if geom:
            coords = [(c[1], c[0]) for c in geom.coords]
            color = highlight_colors.get(info["type"], "cyan")
            length_m = link_length_m(geom)
            folium.PolyLine(
                coords, weight=5, color=color, opacity=0.8,
                tooltip=f"Link {lid} ({info['type']}, {length_m:.0f}m)",
            ).add_to(fg_highlight)
    fg_highlight.add_to(m)

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
                 shn_shapefile=None, max_edge_length_m=2000):
    """
    执行完整 pipeline.

    参数:
        metadata_file: PeMS 元数据文件路径
        output_dir: 输出目录
        buffer_deg: bbox 外扩缓冲 (度)
        ml_threshold: ML 三角不等式阈值
        ramp_threshold: OR/FR 匝道匹配坐标阈值 (度)
        shn_shapefile: Caltrans SHN Lines shapefile 路径
        max_edge_length_m: 边最大长度 (米), 超过则拆分
    """
    if shn_shapefile is None:
        raise ValueError("必须提供 --shn-shapefile 参数")

    t0 = time.time()

    # Phase 0
    sensors_df, bbox_dict = phase0_load_metadata(
        metadata_file, output_dir, buffer_deg
    )

    # Phase 1
    (G, nodes_gdf, edges_gdf, incidence_df, edge_info_df,
     edge_hw_map, links_dict, way_to_rels) = \
        phase1_extract_network(bbox_dict, output_dir, max_edge_length_m)

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
        return phase0_load_metadata(kwargs["metadata_file"], output_dir)

    elif phase_num == 1:
        bbox_dict = load_intermediate("phase0_bbox", output_dir)
        max_edge_length_m = kwargs.get("max_edge_length_m", 2000)
        return phase1_extract_network(bbox_dict, output_dir, max_edge_length_m)

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
        "--max-edge-length", type=float, default=2000,
        help="边最大长度 (米), 超过则拆分 (默认: 2000)",
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
            ml_threshold=args.ml_threshold,
            ramp_threshold=args.ramp_threshold,
            shn_shapefile=args.shn_shapefile,
            max_edge_length_m=args.max_edge_length,
        )
    else:
        run_pipeline(
            args.metadata_file,
            output_dir=args.output,
            buffer_deg=args.buffer,
            ml_threshold=args.ml_threshold,
            ramp_threshold=args.ramp_threshold,
            shn_shapefile=args.shn_shapefile,
            max_edge_length_m=args.max_edge_length,
        )

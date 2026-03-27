"""
Directed Network Graph Formation
=================================
Original R code by: Saurabh Maheshwari (UC Davis, May 2018)
Python conversion for PeMS sensor-to-OSM road network mapping.

Objective:
  Automate the process of directed network graph formation:
  - Creation of incidence matrix
  - Node adjacency matrix
  - Map PeMS sensors to appropriate OSM links

Dependencies:
  pip install osmnx geopandas pandas numpy folium shapely
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from shapely.geometry import Point, LineString
from collections import defaultdict

warnings.filterwarnings("ignore")


# =============================================================================
# 配置参数
# =============================================================================
class Config:
    """Configuration for the network graph construction pipeline."""

    # 区域边界框 (west, south, east, north)
    # 原始 R 代码使用 Orange County 子区域:
    # bbox = corner_bbox(-118.0042, 33.6363, -117.7226, 33.9194)
    BBOX_WEST = -118.0042
    BBOX_SOUTH = 33.6363
    BBOX_EAST = -117.7226
    BBOX_NORTH = 33.9194

    # PeMS 元数据文件路径
    METADATA_FILE = "../d03_meta/d03_text_meta_2024_11_27.txt"

    # 传感器到 link 匹配的距离阈值
    SENSOR_LINK_THRESHOLD = 1e-4

    # OR/FR 传感器坐标匹配阈值 (度)
    RAMP_COORD_THRESHOLD = 0.01

    # 输出目录
    OUTPUT_DIR = "output"

    # 高速公路名称到编号的映射 (需要根据区域手动配置)
    # key: OSM 中的 name 标签值, value: 统一编号
    FWY_NAME_MAP = {
        "Garden Grove Freeway": "CA 22",
        "Costa Mesa Freeway": "CA 55",
        "Laguna Canyon Road": "CA 133 Toll",
        "Santa Ana Freeway": "I 5",
        "San Diego Freeway": "I 405",
        "Laguna Freeway": "CA 133",
        "Foothill/Eastern Transportation Corridor": "CA 241 Toll",
        "Riverside Freeway": "CA 91",
        "Artesia Freeway": "CA 91",
        "Corona del Mar Freeway": "CA 73",
        "Orange Freeway": "CA 57",
        "Express": "CA 91 Toll",
        "Marina Freeway": "CA 90",
        "San Joaquin Hills Transportation Corridor": "CA 73 Toll",
    }


# =============================================================================
# 第一步: 从 OSM 提取高速公路网络
# =============================================================================
class OSMNetworkExtractor:
    """
    从 OpenStreetMap 提取高速公路网络 (motorway + motorway_link).

    对应 R 代码 L12-23: 使用 bounding box 提取 highway=motorway 的 ways 和 nodes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.G = None              # osmnx 图对象
        self.nodes_gdf = None      # 所有节点 GeoDataFrame
        self.edges_gdf = None      # 所有边 GeoDataFrame
        self.ways_data = {}        # way_id -> {nodes: [...], tags: {...}}

    def extract(self):
        """从 OSM 提取 motorway 和 motorway_link 数据."""
        print("=" * 60)
        print("Step 1: Extracting highway network from OSM...")
        bbox = (
            self.config.BBOX_NORTH,
            self.config.BBOX_SOUTH,
            self.config.BBOX_EAST,
            self.config.BBOX_WEST,
        )

        # 使用 osmnx 获取 motorway 类型的道路网络
        # custom_filter 对应 R 代码中 v %grep% "motorway" 的提取
        cf = '["highway"~"motorway|motorway_link"]'
        self.G = ox.graph_from_bbox(
            bbox=bbox,
            network_type="drive",
            custom_filter=cf,
            retain_all=True,
        )

        self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G)
        print(f"  Extracted {len(self.nodes_gdf)} nodes, {len(self.edges_gdf)} edges")

        # 提取 ways 的详细信息 (tags, node refs)
        self._extract_way_details()
        return self

    def _extract_way_details(self):
        """提取每条 way 的标签和节点引用列表."""
        for u, v, key, data in self.G.edges(keys=True, data=True):
            osmid = data.get("osmid", f"{u}-{v}")
            if isinstance(osmid, list):
                for oid in osmid:
                    self._store_way(oid, u, v, data)
            else:
                self._store_way(osmid, u, v, data)

    def _store_way(self, osmid, u, v, data):
        """存储单条 way 的信息."""
        if osmid not in self.ways_data:
            self.ways_data[osmid] = {
                "nodes": [],
                "tags": {
                    "highway": data.get("highway", ""),
                    "name": data.get("name", ""),
                    "ref": data.get("ref", ""),
                    "oneway": data.get("oneway", False),
                    "lanes": data.get("lanes", ""),
                    "maxspeed": data.get("maxspeed", ""),
                },
                "u": u,
                "v": v,
            }
        way = self.ways_data[osmid]
        if u not in way["nodes"]:
            way["nodes"].append(u)
        if v not in way["nodes"]:
            way["nodes"].append(v)


# =============================================================================
# 第二步: 节点精简与关联矩阵构建
# =============================================================================
class GraphBuilder:
    """
    构建有向网络图: 节点精简、关联矩阵、邻接矩阵.

    对应 R 代码:
    - L26-41:  提取连接两条 link 的节点 (节点精简 12351 -> 2973)
    - L49-72:  创建关联矩阵 (incidence matrix)
    - L203-232: 拆分包含 >2 个节点的 link
    """

    def __init__(self, G, nodes_gdf, edges_gdf):
        self.G = G
        self.nodes_gdf = nodes_gdf
        self.edges_gdf = edges_gdf
        self.topo_nodes = set()        # 拓扑节点集合 (精简后)
        self.links = {}                # link_id -> (head_node, tail_node)
        self.incidence_matrix = None   # 关联矩阵 DataFrame
        self.adjacency_matrix = None   # 邻接矩阵 DataFrame

    def simplify_nodes(self):
        """
        节点精简: 只保留连接两条 link 的拓扑节点.

        对应 R 代码 L26-41:
        遍历每条 way, 只保留第一个和最后一个节点.
        """
        print("\nStep 2: Simplifying nodes...")
        print(f"  Original nodes: {len(self.nodes_gdf)}")

        # 对应 R 代码: 对每条 way 只取首尾节点
        for u, v, key, data in self.G.edges(keys=True, data=True):
            self.topo_nodes.add(u)
            self.topo_nodes.add(v)

        # 进一步精简: 只保留度 >= 1 的节点 (即参与连接的节点)
        print(f"  Topological nodes: {len(self.topo_nodes)}")
        return self

    def build_incidence_matrix(self):
        """
        构建关联矩阵.

        对应 R 代码 L49-72:
        行 = 节点, 列 = link
        尾部节点 (起点) = -1, 头部节点 (终点) = +1

        以及 L203-232:
        如果一条 link 包含 >2 个节点, 将其拆分为多条子 link.
        """
        print("\nStep 3: Building incidence matrix...")

        topo_node_list = sorted(list(self.topo_nodes))
        node_to_idx = {n: i for i, n in enumerate(topo_node_list)}

        link_records = []  # (link_id, tail_node, head_node)

        for u, v, key, data in self.G.edges(keys=True, data=True):
            osmid = data.get("osmid", f"{u}_{v}_{key}")
            if isinstance(osmid, list):
                # 多个 osmid 共享同一条边
                for oid in osmid:
                    link_id = str(oid)
                    link_records.append((link_id, u, v))
            else:
                link_id = str(osmid)
                link_records.append((link_id, u, v))

        # 去重
        seen = set()
        unique_records = []
        for lid, u, v in link_records:
            if lid not in seen:
                seen.add(lid)
                unique_records.append((lid, u, v))
                self.links[lid] = (u, v)  # tail -> head

        link_ids = [r[0] for r in unique_records]
        n_nodes = len(topo_node_list)
        n_links = len(unique_records)

        # 构建关联矩阵
        mat = np.zeros((n_nodes, n_links), dtype=int)
        for j, (lid, u, v) in enumerate(unique_records):
            if u in node_to_idx:
                mat[node_to_idx[u], j] = -1  # 尾部 (起点)
            if v in node_to_idx:
                mat[node_to_idx[v], j] = +1  # 头部 (终点)

        self.incidence_matrix = pd.DataFrame(
            mat,
            index=[str(n) for n in topo_node_list],
            columns=link_ids,
        )

        print(f"  Incidence matrix shape: {self.incidence_matrix.shape}")
        print(f"  (rows=nodes, cols=links)")
        return self

    def build_adjacency_matrix(self):
        """
        从关联矩阵构建节点邻接矩阵.

        对应 R 代码末尾 "Creating the adjacency matrix" 部分:
        对每条 link, 找到值为 +1 和 -1 的节点, 在邻接矩阵中设置对应单元格为 1.
        """
        print("\nBuilding adjacency matrix from incidence matrix...")

        node_ids = list(self.incidence_matrix.index)
        n = len(node_ids)
        adj = np.zeros((n, n), dtype=int)
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        for col in self.incidence_matrix.columns:
            col_data = self.incidence_matrix[col]
            tail_nodes = col_data[col_data == -1].index.tolist()
            head_nodes = col_data[col_data == +1].index.tolist()
            for t in tail_nodes:
                for h in head_nodes:
                    ti, hi = node_to_idx[t], node_to_idx[h]
                    adj[ti][hi] = 1
                    adj[hi][ti] = 1  # 无向邻接 (与 R 代码一致)

        self.adjacency_matrix = pd.DataFrame(adj, index=node_ids, columns=node_ids)
        print(f"  Adjacency matrix shape: {self.adjacency_matrix.shape}")
        return self


# =============================================================================
# 第三步: 高速公路归属与方向判定
# =============================================================================
class FreewayAssigner:
    """
    将 OSM link 分配到对应的高速公路, 并判定方向.

    对应 R 代码:
    - L74-116:  使用 name/ref 标签归属高速公路
    - L119-153: 比较首尾节点坐标判定方向 (N/S/E/W)
    - L155-201: 添加匝道 (motorway_link) 到高速公路数据框
    """

    def __init__(self, graph_builder, osm_extractor, config):
        self.gb = graph_builder
        self.osm = osm_extractor
        self.config = config
        self.fwy_links = {}      # fwy_number -> [{id, dir}, ...]
        self.fwy_links_mtr = {}  # 含匝道的版本

    def assign_freeways(self):
        """
        将每条 link 分配到对应的高速公路编号.

        对应 R 代码 L90-116:
        使用 "ref" 和 "name" 标签组合, 通过 FWY_NAME_MAP 映射到高速公路编号.
        """
        print("\nStep 4: Assigning links to freeways...")

        fwy_map = self.config.FWY_NAME_MAP
        link_to_fwy = {}

        for way_id, way_info in self.osm.ways_data.items():
            tags = way_info["tags"]
            if tags["highway"] not in ("motorway", "motorway_link"):
                continue

            fwy_id = None

            # 优先使用 ref 标签 (如 "I 5", "CA 55")
            ref = tags.get("ref", "")
            if ref:
                fwy_id = self._extract_fwy_number(ref)

            # 如果 ref 为空, 使用 name 标签通过映射表
            if fwy_id is None:
                name = tags.get("name", "")
                if name in fwy_map:
                    mapped_ref = fwy_map[name]
                    fwy_id = self._extract_fwy_number(mapped_ref)

            if fwy_id is not None:
                link_to_fwy[str(way_id)] = fwy_id

        # 按高速公路编号分组
        fwy_groups = defaultdict(list)
        for link_id, fwy_num in link_to_fwy.items():
            fwy_groups[fwy_num].append({"id": link_id, "dir": None})

        self.fwy_links = dict(fwy_groups)
        print(f"  Assigned links to {len(self.fwy_links)} freeways: "
              f"{sorted(self.fwy_links.keys())}")
        return self

    def assign_directions(self, sensor_data):
        """
        为每条 link 判定方向.

        对应 R 代码 L119-153:
        利用关联矩阵找到首尾节点坐标, 结合 PeMS 传感器的 Dir 字段:
        - 南北向: 比较纬度, 首节点纬度 > 尾节点纬度 → "S"
        - 东西向: 比较经度, 首节点经度 > 尾节点经度 → "W"
        """
        print("\nStep 5: Assigning directions to links...")

        inc = self.gb.incidence_matrix
        nodes_gdf = self.osm.nodes_gdf

        for fwy_num, links in self.fwy_links.items():
            # 从传感器数据获取该高速公路的主方向
            fwy_sensors = sensor_data[sensor_data["Fwy"] == fwy_num]
            if len(fwy_sensors) == 0:
                continue
            primary_dir = fwy_sensors["Dir"].iloc[0]

            for link_info in links:
                link_id = link_info["id"]
                if link_id not in inc.columns:
                    continue

                col = inc[link_id]
                tail_nodes = col[col == -1].index.tolist()
                head_nodes = col[col == +1].index.tolist()

                if not tail_nodes or not head_nodes:
                    continue

                tail_id = int(tail_nodes[0])
                head_id = int(head_nodes[0])

                # 获取节点坐标
                try:
                    tail_coord = nodes_gdf.loc[tail_id]
                    head_coord = nodes_gdf.loc[head_id]
                except KeyError:
                    continue

                tail_lat = tail_coord["y"]
                tail_lon = tail_coord["x"]
                head_lat = head_coord["y"]
                head_lon = head_coord["x"]

                if primary_dir in ("N", "S"):
                    # 尾部纬度 > 头部纬度 → 方向为 S (从北向南)
                    link_info["dir"] = "S" if tail_lat > head_lat else "N"
                elif primary_dir in ("E", "W"):
                    # 尾部经度 > 头部经度 → 方向为 W (从东向西)
                    link_info["dir"] = "W" if tail_lon > head_lon else "E"

        assigned = sum(
            1 for links in self.fwy_links.values()
            for l in links if l["dir"] is not None
        )
        print(f"  Directions assigned to {assigned} links")
        return self

    def add_ramps(self):
        """
        将匝道 (motorway_link) 添加到高速公路数据框.

        对应 R 代码 L155-201:
        对高速主线上的每个节点, 检查关联矩阵中与之相连的 link
        是否是 motorway_link, 如果是则添加到 fwy_links_mtr.
        """
        print("\nStep 6: Adding ramps to freeway data...")

        import copy
        self.fwy_links_mtr = copy.deepcopy(self.fwy_links)

        inc = self.gb.incidence_matrix
        motorway_link_ways = set()

        for way_id, way_info in self.osm.ways_data.items():
            if way_info["tags"]["highway"] == "motorway_link":
                motorway_link_ways.add(str(way_id))

        for fwy_num, links in self.fwy_links.items():
            existing_ids = {l["id"] for l in links}
            ramp_links = []

            for link_info in links:
                link_id = link_info["id"]
                if link_id not in inc.columns:
                    continue

                col = inc[link_id]
                # 获取该 link 的首尾节点
                connected_nodes = col[col != 0].index.tolist()

                for node_id in connected_nodes:
                    # 找到该节点关联的所有 link
                    row = inc.loc[node_id]
                    connected_links = row[row != 0].index.tolist()

                    for cl in connected_links:
                        if cl not in existing_ids and cl in motorway_link_ways:
                            ramp_links.append({
                                "id": cl,
                                "dir": link_info["dir"],
                            })
                            existing_ids.add(cl)

            self.fwy_links_mtr[fwy_num] = links + ramp_links

        total_ramps = sum(
            len(v) - len(self.fwy_links.get(k, []))
            for k, v in self.fwy_links_mtr.items()
        )
        print(f"  Added {total_ramps} ramp links")
        return self

    @staticmethod
    def _extract_fwy_number(ref_str):
        """从 ref 字符串中提取高速公路编号 (如 'I 5' -> 5)."""
        import re
        numbers = re.findall(r"\d+", ref_str)
        if numbers:
            return int(numbers[0])
        return None


# =============================================================================
# 第四步: 传感器到 Link 的映射
# =============================================================================
class SensorMapper:
    """
    将 PeMS 传感器映射到 OSM link.

    对应 R 代码:
    - L236-283: ML 传感器的三角不等式匹配算法
    - L285-341: 处理被拆分的 link
    - L422-611: OR/FR 匝道传感器匹配
    - L1028-1061: FF 传感器匹配
    """

    def __init__(self, graph_builder, osm_extractor, freeway_assigner, config):
        self.gb = graph_builder
        self.osm = osm_extractor
        self.fa = freeway_assigner
        self.config = config
        self.ml_mapping = {}    # sensor_id -> link_id
        self.or_mapping = {}    # sensor_id -> link_id
        self.fr_mapping = {}    # sensor_id -> link_id
        self.ff_mapping = {}    # sensor_id -> link_id
        self.unmatched = []     # 需要手动匹配的传感器

    def map_ml_sensors(self, sensor_data):
        """
        将主线 (ML) 传感器映射到 link.

        对应 R 代码 L236-283, 基于三角不等式的匹配算法:

        对每个传感器位置 S, 对该方向上每条 link 的首尾节点 (N1, N2):
          d1 = dist(N1, N2)       # 节点间距离
          d2 = dist(N1, S)        # 首节点到传感器距离
          d3 = dist(N2, S)        # 尾节点到传感器距离
          d4 = (d2 + d3) - d1     # 偏差值

        d4 最小且 < threshold 的 link 为匹配结果.

        几何直觉: 如果传感器恰好在 link 线段上, 则 d2+d3=d1, 即 d4=0.
        """
        print("\nStep 7: Mapping ML sensors to links...")

        ml_sensors = sensor_data[sensor_data["Type"] == "ML"].copy()
        nodes_gdf = self.osm.nodes_gdf
        inc = self.gb.incidence_matrix
        threshold = self.config.SENSOR_LINK_THRESHOLD

        matched = 0
        total = len(ml_sensors)

        for _, sensor in ml_sensors.iterrows():
            sensor_id = sensor["ID"]
            sensor_lat = sensor["Latitude"]
            sensor_lon = sensor["Longitude"]
            sensor_fwy = sensor["Fwy"]
            sensor_dir = sensor["Dir"]

            # 获取该高速公路该方向上的所有 link
            fwy_links = self.fa.fwy_links_mtr.get(sensor_fwy, [])
            candidate_links = [
                l for l in fwy_links
                if l["dir"] == sensor_dir and l["id"] in inc.columns
            ]

            best_d4 = float("inf")
            best_link = None

            for link_info in candidate_links:
                link_id = link_info["id"]
                col = inc[link_id]
                tail_ids = col[col == -1].index.tolist()
                head_ids = col[col == +1].index.tolist()

                if not tail_ids or not head_ids:
                    continue

                for tail_str in tail_ids:
                    for head_str in head_ids:
                        try:
                            tail_node = nodes_gdf.loc[int(tail_str)]
                            head_node = nodes_gdf.loc[int(head_str)]
                        except KeyError:
                            continue

                        # 三角不等式计算
                        d1 = np.sqrt(
                            (tail_node["y"] - head_node["y"]) ** 2
                            + (tail_node["x"] - head_node["x"]) ** 2
                        )
                        d2 = np.sqrt(
                            (sensor_lat - tail_node["y"]) ** 2
                            + (sensor_lon - tail_node["x"]) ** 2
                        )
                        d3 = np.sqrt(
                            (sensor_lat - head_node["y"]) ** 2
                            + (sensor_lon - head_node["x"]) ** 2
                        )
                        d4 = (d2 + d3) - d1

                        if abs(d4) < best_d4:
                            best_d4 = abs(d4)
                            best_link = link_id

            if best_link is not None and best_d4 < threshold:
                self.ml_mapping[sensor_id] = best_link
                matched += 1
            else:
                self.unmatched.append({
                    "sensor_id": sensor_id,
                    "type": "ML",
                    "lat": sensor_lat,
                    "lon": sensor_lon,
                    "best_d4": best_d4,
                })

            if matched % 50 == 0 and matched > 0:
                print(f"    Matched {matched}/{total} ML sensors...")

        print(f"  ML sensors matched: {matched}/{total}")
        print(f"  Unmatched: {total - matched}")
        return self

    def map_ramp_sensors(self, sensor_data):
        """
        将 OR (On-Ramp) 和 FR (Off-Ramp) 传感器映射到匝道 link.

        对应 R 代码 L500-982:
        1. 识别高速主线节点上的匝道 link
        2. 根据关联矩阵的方向区分 OR (+1方向) 和 FR (-1方向)
        3. 按坐标顺序自动匹配传感器和匝道
        4. 无法匹配的标记为 NA, 需要手动处理

        关键区分逻辑 (R 代码 L517-529):
        - 在某个节点处, +1 方向 (流出主线) 的 motorway_link → OR
        - -1 方向 (流入主线) 的 motorway_link → FR
        """
        print("\nStep 8: Mapping OR/FR sensors to ramp links...")

        or_sensors = sensor_data[sensor_data["Type"] == "OR"]
        fr_sensors = sensor_data[sensor_data["Type"] == "FR"]

        inc = self.gb.incidence_matrix
        nodes_gdf = self.osm.nodes_gdf

        # 识别所有 motorway_link 的 way ID
        motorway_link_ids = set()
        for way_id, way_info in self.osm.ways_data.items():
            if way_info["tags"]["highway"] == "motorway_link":
                motorway_link_ids.add(str(way_id))

        # 对每条高速公路, 找到主线节点上的匝道
        or_ramp_links = {}  # fwy_num -> [(node_id, link_id, lat, lon, dir), ...]
        fr_ramp_links = {}

        for fwy_num, links in self.fa.fwy_links.items():
            or_ramps = []
            fr_ramps = []

            for link_info in links:
                link_id = link_info["id"]
                if link_id not in inc.columns:
                    continue

                col = inc[link_id]
                connected_nodes = col[col != 0].index.tolist()

                for node_str in connected_nodes:
                    row = inc.loc[node_str]
                    connected_links = row[row != 0].index.tolist()

                    for cl in connected_links:
                        if cl in motorway_link_ids:
                            try:
                                node_coord = nodes_gdf.loc[int(node_str)]
                            except KeyError:
                                continue

                            # 判断 OR 还是 FR:
                            # 从主线节点看, +1 = 流出 = OR, -1 = 流入 = FR
                            val = inc.loc[node_str, cl]
                            ramp_info = {
                                "node_id": node_str,
                                "link_id": cl,
                                "lat": node_coord["y"],
                                "lon": node_coord["x"],
                                "dir": link_info["dir"],
                            }

                            if val == +1:  # 从该节点流出 -> on-ramp
                                or_ramps.append(ramp_info)
                            elif val == -1:  # 从该节点流入 -> off-ramp
                                fr_ramps.append(ramp_info)

            or_ramp_links[fwy_num] = or_ramps
            fr_ramp_links[fwy_num] = fr_ramps

        # 按方向和坐标排序后, 逐一匹配传感器
        or_matched, or_total = self._match_ramp_sensors(
            or_sensors, or_ramp_links, self.or_mapping, "OR"
        )
        fr_matched, fr_total = self._match_ramp_sensors(
            fr_sensors, fr_ramp_links, self.fr_mapping, "FR"
        )

        print(f"  OR sensors matched: {or_matched}/{or_total}")
        print(f"  FR sensors matched: {fr_matched}/{fr_total}")
        return self

    def _match_ramp_sensors(self, sensors_df, ramp_links_dict,
                            mapping_dict, sensor_type):
        """
        基于坐标最近距离匹配匝道传感器.

        对应 R 代码 L624-982 中的坐标顺序匹配逻辑,
        简化为最近距离匹配以提高鲁棒性.
        """
        matched = 0
        total = len(sensors_df)
        threshold = self.config.RAMP_COORD_THRESHOLD

        for _, sensor in sensors_df.iterrows():
            sensor_id = sensor["ID"]
            sensor_lat = sensor["Latitude"]
            sensor_lon = sensor["Longitude"]
            sensor_fwy = sensor["Fwy"]
            sensor_dir = sensor["Dir"]

            ramps = ramp_links_dict.get(sensor_fwy, [])
            # 过滤同方向的匝道
            dir_ramps = [r for r in ramps if r["dir"] == sensor_dir]

            if not dir_ramps:
                self.unmatched.append({
                    "sensor_id": sensor_id,
                    "type": sensor_type,
                    "lat": sensor_lat,
                    "lon": sensor_lon,
                    "reason": "no_ramps_in_direction",
                })
                continue

            # 找距离最近的匝道节点
            best_dist = float("inf")
            best_link = None

            for ramp in dir_ramps:
                dist = np.sqrt(
                    (sensor_lat - ramp["lat"]) ** 2
                    + (sensor_lon - ramp["lon"]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_link = ramp["link_id"]

            if best_link is not None and best_dist < threshold:
                mapping_dict[sensor_id] = best_link
                matched += 1
            else:
                self.unmatched.append({
                    "sensor_id": sensor_id,
                    "type": sensor_type,
                    "lat": sensor_lat,
                    "lon": sensor_lon,
                    "best_dist": best_dist,
                    "reason": "distance_exceeds_threshold",
                })

        return matched, total

    def map_ff_sensors(self, sensor_data):
        """
        标记 FF (Freeway-Freeway) 传感器为需要手动匹配.

        对应 R 代码 L1028-1061:
        FF 传感器在原始 R 代码中完全通过 readline() 手动输入.
        这里标记所有 FF 传感器并输出到待手动匹配列表.
        """
        print("\nStep 9: Identifying FF sensors for manual mapping...")

        ff_sensors = sensor_data[sensor_data["Type"] == "FF"]
        for _, sensor in ff_sensors.iterrows():
            self.unmatched.append({
                "sensor_id": sensor["ID"],
                "type": "FF",
                "lat": sensor["Latitude"],
                "lon": sensor["Longitude"],
                "reason": "ff_requires_manual",
            })

        print(f"  FF sensors requiring manual mapping: {len(ff_sensors)}")
        return self


# =============================================================================
# 第五步: 可视化
# =============================================================================
class NetworkVisualizer:
    """
    使用 Folium 创建交互式地图.

    对应 R 代码中使用 leaflet/mapview 的可视化部分.
    """

    # 传感器类型的颜色映射
    SENSOR_COLORS = {
        "ML": "blue",
        "OR": "green",
        "FR": "red",
        "FF": "orange",
        "HV": "purple",
        "CD": "cadetblue",
    }

    @staticmethod
    def create_map(osm_extractor, sensor_data, sensor_mapper=None):
        """创建包含路网和传感器的交互式地图."""
        print("\nCreating interactive map...")

        center_lat = (Config.BBOX_NORTH + Config.BBOX_SOUTH) / 2
        center_lon = (Config.BBOX_EAST + Config.BBOX_WEST) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # 添加路网边
        edges_gdf = osm_extractor.edges_gdf
        for _, edge in edges_gdf.iterrows():
            if edge.geometry and edge.geometry.geom_type == "LineString":
                coords = [(c[1], c[0]) for c in edge.geometry.coords]
                folium.PolyLine(
                    coords,
                    weight=2,
                    color="gray",
                    opacity=0.7,
                ).add_to(m)

        # 按类型分组添加传感器
        for sensor_type, color in NetworkVisualizer.SENSOR_COLORS.items():
            type_sensors = sensor_data[sensor_data["Type"] == sensor_type]
            fg = folium.FeatureGroup(
                name=f"{sensor_type} ({len(type_sensors)})",
                show=(sensor_type == "ML"),
            )

            for _, s in type_sensors.iterrows():
                popup_text = (
                    f"ID: {s['ID']}<br>"
                    f"Type: {s['Type']}<br>"
                    f"Fwy: {s.get('Fwy', 'N/A')}<br>"
                    f"Dir: {s.get('Dir', 'N/A')}<br>"
                    f"Name: {s.get('Name', 'N/A')}"
                )

                # 如果有匹配结果, 添加 link 信息
                if sensor_mapper:
                    link = (
                        sensor_mapper.ml_mapping.get(s["ID"])
                        or sensor_mapper.or_mapping.get(s["ID"])
                        or sensor_mapper.fr_mapping.get(s["ID"])
                        or sensor_mapper.ff_mapping.get(s["ID"])
                    )
                    if link:
                        popup_text += f"<br>Link: {link}"
                    else:
                        popup_text += "<br>Link: <b>UNMATCHED</b>"

                folium.CircleMarker(
                    location=[s["Latitude"], s["Longitude"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=str(s["ID"]),
                ).add_to(fg)

            fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m


# =============================================================================
# 第六步: 输出结果
# =============================================================================
class ResultExporter:
    """导出所有构建结果."""

    @staticmethod
    def export(output_dir, graph_builder, sensor_mapper, fwy_assigner):
        """导出关联矩阵、邻接矩阵、传感器映射结果."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 关联矩阵
        inc_path = os.path.join(output_dir, "incidence_matrix.csv")
        graph_builder.incidence_matrix.to_csv(inc_path)
        print(f"  Saved incidence matrix to {inc_path}")

        # 2. 邻接矩阵
        adj_path = os.path.join(output_dir, "adjacency_matrix.csv")
        graph_builder.adjacency_matrix.to_csv(adj_path)
        print(f"  Saved adjacency matrix to {adj_path}")

        # 3. 传感器映射结果
        all_mappings = []
        for sid, lid in sensor_mapper.ml_mapping.items():
            all_mappings.append({"sensor_id": sid, "link_id": lid, "type": "ML"})
        for sid, lid in sensor_mapper.or_mapping.items():
            all_mappings.append({"sensor_id": sid, "link_id": lid, "type": "OR"})
        for sid, lid in sensor_mapper.fr_mapping.items():
            all_mappings.append({"sensor_id": sid, "link_id": lid, "type": "FR"})
        for sid, lid in sensor_mapper.ff_mapping.items():
            all_mappings.append({"sensor_id": sid, "link_id": lid, "type": "FF"})

        mapping_df = pd.DataFrame(all_mappings)
        mapping_path = os.path.join(output_dir, "sensor_link_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False)
        print(f"  Saved sensor-link mapping ({len(mapping_df)} records) to {mapping_path}")

        # 4. 未匹配传感器 (需要手动处理)
        unmatched_df = pd.DataFrame(sensor_mapper.unmatched)
        unmatched_path = os.path.join(output_dir, "unmatched_sensors.csv")
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"  Saved unmatched sensors ({len(unmatched_df)} records) to {unmatched_path}")

        # 5. 高速公路-link 关系
        fwy_records = []
        for fwy_num, links in fwy_assigner.fwy_links_mtr.items():
            for l in links:
                fwy_records.append({
                    "fwy_number": fwy_num,
                    "link_id": l["id"],
                    "direction": l["dir"],
                })
        fwy_df = pd.DataFrame(fwy_records)
        fwy_path = os.path.join(output_dir, "freeway_link_relation.csv")
        fwy_df.to_csv(fwy_path, index=False)
        print(f"  Saved freeway-link relation ({len(fwy_df)} records) to {fwy_path}")

        # 6. 汇总统计
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total ML sensors matched:  {len(sensor_mapper.ml_mapping)}")
        print(f"  Total OR sensors matched:  {len(sensor_mapper.or_mapping)}")
        print(f"  Total FR sensors matched:  {len(sensor_mapper.fr_mapping)}")
        print(f"  Total FF sensors matched:  {len(sensor_mapper.ff_mapping)}")
        print(f"  Total unmatched sensors:   {len(sensor_mapper.unmatched)}")
        print(f"  Incidence matrix shape:    {graph_builder.incidence_matrix.shape}")
        print(f"  Adjacency matrix shape:    {graph_builder.adjacency_matrix.shape}")


# =============================================================================
# 主流程
# =============================================================================
def load_pems_metadata(filepath):
    """
    加载 PeMS 元数据文件.

    对应 R 代码 L75-79:
    MetaData = read.table("d012_text_meta.txt", header=TRUE, sep="\\t", fill=TRUE)
    """
    print(f"\nLoading PeMS metadata from {filepath}...")

    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        print("  Creating sample metadata for demonstration...")
        # 创建示例数据
        sample_data = {
            "ID": [1204701, 1204759, 1204823, 1205035, 1205060],
            "Fwy": [5, 5, 5, 5, 5],
            "Dir": ["S", "S", "S", "S", "S"],
            "Type": ["ML", "ML", "ML", "ML", "ML"],
            "Latitude": [33.79, 33.78, 33.77, 33.75, 33.74],
            "Longitude": [-117.88, -117.88, -117.87, -117.87, -117.87],
            "Name": ["I-5 S ML"] * 5,
            "User_ID_1": [""] * 5,
        }
        return pd.DataFrame(sample_data)

    df = pd.read_csv(filepath, sep="\t", encoding="latin-1")

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

    df = df.rename(columns=col_map)
    print(f"  Loaded {len(df)} sensor records")
    print(f"  Types: {df['Type'].value_counts().to_dict()}")
    return df


def main():
    """
    主流程: 完整的有向网络图构建 pipeline.

    执行顺序:
    1. 从 OSM 提取高速公路网络
    2. 节点精简
    3. 构建关联矩阵
    4. 加载 PeMS 元数据
    5. 高速公路归属与方向判定
    6. 添加匝道
    7. ML 传感器匹配
    8. OR/FR 传感器匹配
    9. FF 传感器标记
    10. 构建邻接矩阵
    11. 可视化与导出
    """
    config = Config()

    # ---- Step 1: 提取 OSM 路网 ----
    extractor = OSMNetworkExtractor(config)
    extractor.extract()

    # ---- Step 2-3: 节点精简与关联矩阵 ----
    builder = GraphBuilder(
        extractor.G, extractor.nodes_gdf, extractor.edges_gdf
    )
    builder.simplify_nodes()
    builder.build_incidence_matrix()

    # ---- Step 4: 加载 PeMS 元数据 ----
    sensor_data = load_pems_metadata(config.METADATA_FILE)

    # ---- Step 5-6: 高速公路归属、方向判定、匝道添加 ----
    assigner = FreewayAssigner(builder, extractor, config)
    assigner.assign_freeways()
    assigner.assign_directions(sensor_data)
    assigner.add_ramps()

    # ---- Step 7-9: 传感器匹配 ----
    mapper = SensorMapper(builder, extractor, assigner, config)
    mapper.map_ml_sensors(sensor_data)
    mapper.map_ramp_sensors(sensor_data)
    mapper.map_ff_sensors(sensor_data)

    # ---- Step 10: 构建邻接矩阵 ----
    builder.build_adjacency_matrix()

    # ---- Step 11: 可视化与导出 ----
    viz_map = NetworkVisualizer.create_map(extractor, sensor_data, mapper)
    map_path = os.path.join(config.OUTPUT_DIR, "network_map.html")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    viz_map.save(map_path)
    print(f"  Saved interactive map to {map_path}")

    ResultExporter.export(config.OUTPUT_DIR, builder, mapper, assigner)

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

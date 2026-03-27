"""
传感器匹配审查脚本 2.7
======================
交叉检查 PeMS 传感器名称与 OSM 路段标签 (destination, ref, name),
并把 2.7 Phase 0 的 SHN postmile 坐标修正信息一并带入审查结果.

用法:
    python review_matching2.7.py output_d03_2.7_full/ d3_motorway.osm

输出:
    output/review_results.csv      - 所有已匹配 sensor 的审查结果
    output/review_suspicious.csv   - 仅可疑项
    output/review_unmatched.csv    - 未匹配项汇总
    output/review_override.csv     - 人工修正模板
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# =============================================================================
# LLM 地名匹配
# =============================================================================
def llm_batch_name_check(pairs, api_key=None):
    """
    调用通义千问 API 批量判断 PeMS 地名与 OSM destination 是否指向同一地点.

    依赖: pip install openai
    环境变量: DASHSCOPE_API_KEY
    """
    try:
        from openai import OpenAI
    except ImportError:
        log("  openai 库未安装, 跳过 LLM 审查 (pip install openai)", "WARN")
        return {}

    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-fc29696324714eb799c8fd6a25eba20c")
    if not api_key:
        log("  DASHSCOPE_API_KEY 未设置, 跳过 LLM 审查", "WARN")
        return {}

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def _build_prompt(batch):
        items_lines = []
        for i, p in enumerate(batch):
            line = f'{i+1}. sensor_id={p["id"]}'
            line += f', type={p.get("sensor_type","")}'
            line += f', fwy={p.get("fwy","")}{p.get("dir","")}'
            line += f', PeMS_name="{p["pems"]}"'
            line += f', PeMS_full="{p.get("pems_full_name","")}"'
            line += f', OSM_destination="{p["osm"]}"'
            if p.get("osm_dest_ref"):
                line += f', OSM_dest_ref="{p["osm_dest_ref"]}"'
            if p.get("origin"):
                line += f', origin_road="{p["origin"]}"'
            items_lines.append(line)
        items_text = "\n".join(items_lines)

        return f"""你是加州高速公路交通传感器匹配审查专家。你需要判断 PeMS 传感器的匹配路段是否正确。

## 背景
PeMS 传感器安装在加州高速公路的匝道上 (OR=入口匝道, FR=出口匝道)。每个传感器有一个名称 (PeMS_name), 描述该匝道连接的地方道路或目标高速。每个传感器已被自动匹配到一条 OSM 路段, 该路段有 destination (路牌目的地) 和 origin_road (匝道连接的地方道路) 等属性。

你的任务: 判断 PeMS_name 描述的匝道与 OSM 路段的信息是否一致 (即匹配是否正确)。

## 判断规则

### OR (入口匝道) 审查重点: origin_road
OR 传感器在入口匝道上, 车辆从地方道路驶入高速。PeMS_name 描述的是"从哪条路来"。
- 优先检查 origin_road 是否与 PeMS_name 一致
- OSM_destination 通常是高速方向的远端城市 (如 "Sacramento", "Los Angeles"), 与 PeMS_name 不同是正常的
- 如果 origin_road 与 PeMS_name 匹配 → match=true, 即使 destination 是远端城市
- 如果无 origin_road, 则检查 PeMS_name 中 "to XXX" 的目标高速是否与 OSM_dest_ref 一致

### FR (出口匝道) 审查重点: destination
FR 传感器在出口匝道上, 车辆从高速驶向地方道路。PeMS_name 描述的是"去哪条路"。
- 优先检查 OSM_destination 是否与 PeMS_name 一致
- origin_road 对 FR 通常不适用 (FR 从主线出发, 不从地方道路出发)
- 如果 destination 与 PeMS_name 匹配 → match=true

### 通用地名匹配规则 (OR 和 FR 均适用):
1. 缩写等价: "T Street"="T St", "Boulevard"="Blvd", "Avenue"="Ave", "Road"="Rd", "Drive"="Dr"
2. 别名等价: "Business 80"="Capital City Freeway"="I 80 Bus", "Westside Freeway"="I 5"
3. 方向/编号后缀忽略: "Florin Rd" 匹配 "Florin Road East", "CA 99" 匹配 "99"
4. 部分匹配: "Richards Blvd/I St" 匹配 "Richards Boulevard"
5. 多目的地: OSM 字段含分号分隔的多个地名, PeMS 匹配其中任一个即可
6. HOV/Slip/Loop/Express 等匝道类型后缀不影响地名判断
7. PeMS "to 5SB" 与 OSM_dest_ref="I 5 South" 是等价的

### 地名不匹配 (match=false):
8. PeMS 说的地方道路名与 OSM 的所有字段 (destination, origin_road, dest_ref) 完全不同, 且无法用缩写/别名解释。
9. 例如: FR, PeMS="Broadway" 但 destination="South Lake Tahoe" 且无匹配的 origin_road → 不匹配。

## 数据 ({len(batch)} 条):
{items_text}

## 输出要求
必须对每一条都给出判断, 返回数组长度必须等于 {len(batch)}.
仅返回 JSON 数组, 不要任何其他内容:
[{{"id": sensor_id, "match": true/false, "reason": "简短原因"}}]"""


    def _call_llm(batch):
        """调用一次 LLM, 返回 {sensor_id: {match, reason}}."""
        prompt = _build_prompt(batch)
        partial = {}

        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
            )

            text = completion.choices[0].message.content.strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"^```json\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)

            for item in parsed:
                partial[item["id"]] = {
                    "match": item.get("match", True),
                    "reason": item.get("reason", ""),
                }

        except Exception as e:
            log(f"    API 调用失败: {e}", "WARN")
            for p in batch:
                partial[p["id"]] = {"match": None, "reason": f"API error: {e}"}

        return partial

    results = {}
    BATCH_SIZE = 10
    total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch = pairs[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_idx + 1

        # 第一次调用
        partial = _call_llm(batch)
        results.update(partial)

        # 检查遗漏, 重试一次
        missing = [p for p in batch if p["id"] not in partial]
        if missing:
            log(f"    批次 {batch_num}: 遗漏 {len(missing)} 个, 重试...", "WARN")
            retry_partial = _call_llm(missing)
            results.update(retry_partial)

            still_missing = [p for p in missing if p["id"] not in retry_partial]
            for p in still_missing:
                results[p["id"]] = {
                    "match": None,
                    "reason": "LLM 两次调用均未返回此 sensor",
                }
            if still_missing:
                log(f"    批次 {batch_num}: 重试后仍遗漏 {len(still_missing)} 个",
                    "WARN")

        if batch_num % 5 == 0 or batch_num == total_batches:
            log(f"    LLM 进度: {batch_num}/{total_batches} 批次")

    # 统计
    n_ok = sum(1 for v in results.values() if v.get("match") is True)
    n_bad = sum(1 for v in results.values() if v.get("match") is False)
    n_none = sum(1 for v in results.values() if v.get("match") is None)
    log(f"  LLM 最终结果: 匹配={n_ok}, 不匹配={n_bad}, 未知={n_none}")

    return results




def load_intermediate(name, output_dir):
    pkl_path = os.path.join(output_dir, f"{name}.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_optional_intermediate(name, output_dir, default=None):
    pkl_path = os.path.join(output_dir, f"{name}.pkl")
    if not os.path.exists(pkl_path):
        return default
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def resolve_osm_path(output_dir, osm_path):
    """兼容从 output_dir 内部引用 d3_motorway.osm 的常见用法."""
    if osm_path and os.path.exists(osm_path):
        return osm_path

    candidates = []
    if osm_path:
        candidates.append(os.path.join(output_dir, osm_path))
        candidates.append(os.path.join(output_dir, os.path.basename(osm_path)))
    candidates.append(os.path.join(output_dir, "d3_motorway.osm"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            log(f"  OSM 路径解析为: {candidate}")
            return candidate

    raise FileNotFoundError(f"找不到 OSM 文件: {osm_path}")


# =============================================================================
# Step 1: 加载数据
# =============================================================================
def load_all_data(output_dir, osm_path):
    """加载所有需要的中间结果和 OSM 数据."""
    log("Step 1: 加载数据...")

    # PeMS 元数据
    sensors_df = load_intermediate("phase0_sensors", output_dir)
    log(f"  传感器元数据: {len(sensors_df)} 行")

    # edge 信息 (含 osmid)
    edge_info_df = load_intermediate("phase1_edge_info", output_dir)
    log(f"  edge 信息: {len(edge_info_df)} 行")

    phase0_details_df = load_optional_intermediate(
        "phase0_shn_corrections", output_dir, default=pd.DataFrame()
    )
    if phase0_details_df is not None and len(phase0_details_df) > 0:
        log(f"  Phase 0 修正明细: {len(phase0_details_df)} 行")
    else:
        phase0_details_df = pd.DataFrame()
        log("  Phase 0 修正明细: 未找到")

    ff_manual_df = load_optional_intermediate(
        "phase5_ff_to_manual", output_dir, default=pd.DataFrame()
    )
    if ff_manual_df is not None and len(ff_manual_df) > 0:
        log(f"  FF 待手工分配: {len(ff_manual_df)} 行")
    else:
        ff_manual_df = pd.DataFrame()
        log("  FF 待手工分配: 未找到")

    # 匹配结果
    mappings = {}
    for name, label in [("phase3_ml_mapping", "ML"),
                         ("phase4_or_mapping", "OR"),
                         ("phase4_fr_mapping", "FR")]:
        try:
            df = load_intermediate(name, output_dir)
            mappings[label] = df
            log(f"  {label} 匹配: {len(df)} 行")
        except FileNotFoundError:
            mappings[label] = pd.DataFrame()
            log(f"  {label} 匹配: 未找到")

    # 未匹配结果
    unmatched = {}
    for name, label in [("phase3_ml_unmatched", "ML"),
                         ("phase4_or_unmatched", "OR"),
                         ("phase4_fr_unmatched", "FR")]:
        try:
            df = load_intermediate(name, output_dir)
            unmatched[label] = df
        except FileNotFoundError:
            unmatched[label] = pd.DataFrame()

    # 解析 OSM 文件
    log(f"  解析 OSM: {osm_path}")
    way_tags = parse_osm_ways(osm_path)
    log(f"  OSM way 数量: {len(way_tags)}")

    return sensors_df, edge_info_df, mappings, unmatched, way_tags, phase0_details_df, ff_manual_df


def parse_osm_ways(osm_path):
    """解析 OSM 文件, 提取每条 way 的关键标签."""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    way_tags = {}
    for way in root.findall("way"):
        wid = int(way.get("id"))
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        nd_refs = [nd.get("ref") for nd in way.findall("nd")]
        way_tags[wid] = {
            "highway": tags.get("highway", ""),
            "ref": tags.get("ref", ""),
            "unsigned_ref": tags.get("unsigned_ref", ""),
            "name": tags.get("name", ""),
            "destination": tags.get("destination", ""),
            "destination:ref": tags.get("destination:ref", ""),
            "destination:ref:to": tags.get("destination:ref:to", ""),
            "destination:street": tags.get("destination:street", ""),
            "oneway": tags.get("oneway", ""),
            "nodes": nd_refs,
        }
    return way_tags


# =============================================================================
# Step 2: 构建 edge_key → OSM tag 映射
# =============================================================================
def build_edge_to_osm(edge_info_df, way_tags):
    """构建 edge_key → OSM 标签 的映射."""
    log("Step 2: 构建 edge → OSM 标签映射...")

    edge_osm = {}  # edge_key -> {merged OSM tags}

    for _, row in edge_info_df.iterrows():
        ek = row["edge_key"]
        osmid_str = str(row["osmid"])

        # 一条 edge 可能对应多个 osmid (简化合并)
        merged = {
            "highway": set(),
            "ref": set(),
            "name": set(),
            "destination": set(),
            "destination_ref": set(),
            "destination_ref_to": set(),
            "destination_street": set(),
            "unsigned_ref": set(),
        }

        for oid_str in osmid_str.split(";"):
            oid_str = oid_str.strip()
            try:
                oid = int(oid_str)
            except ValueError:
                continue

            wt = way_tags.get(oid)
            if wt is None:
                continue

            if wt["highway"]:
                merged["highway"].add(wt["highway"])
            if wt["ref"]:
                merged["ref"].add(wt["ref"])
            if wt["name"]:
                merged["name"].add(wt["name"])
            if wt["destination"]:
                for d in wt["destination"].split(";"):
                    merged["destination"].add(d.strip())
            if wt["destination:ref"]:
                for d in wt["destination:ref"].split(";"):
                    merged["destination_ref"].add(d.strip())
            if wt["destination:ref:to"]:
                for d in wt["destination:ref:to"].split(";"):
                    merged["destination_ref_to"].add(d.strip())
            if wt.get("destination:street", ""):
                for d in wt["destination:street"].split(";"):
                    merged["destination_street"].add(d.strip())
            if wt["unsigned_ref"]:
                merged["unsigned_ref"].add(wt["unsigned_ref"])

        # 转为字符串
        edge_osm[ek] = {k: "; ".join(sorted(v)) if v else ""
                        for k, v in merged.items()}

    log(f"  映射 edge 数量: {len(edge_osm)}")
    return edge_osm


# =============================================================================
# Step 2.5: 从 PBF 提取匝道出发地道路名
# =============================================================================
def extract_ramp_origins(way_tags, edge_info_df, pbf_path):
    """
    找到匝道链的非高速端 (dead-end) 节点, 扫描完整 PBF 查找连接的地方道路名.

    返回:
        node_to_road_names: {node_id_str: ["Elk Grove Boulevard", ...]}
    """
    import sys
    try:
        import osmium
    except ImportError:
        log("  osmium 未安装, 跳过匝道出发地提取 (pip install osmium)", "WARN")
        return {}

    log("Step 2.5: 从 PBF 提取匝道出发地道路名...")

    # --- 找匝道的 dead-end 节点 ---
    # 构建 node → 连接的 highway 类型
    node_highway_types = defaultdict(set)  # node_id -> {highway_type, ...}
    for wid, wt in way_tags.items():
        hw = wt["highway"]
        if not hw:
            continue
        nodes = wt["nodes"]
        if nodes:
            # 只看端点
            node_highway_types[nodes[0]].add(hw)
            node_highway_types[nodes[-1]].add(hw)

    # dead-end: 只连接 motorway_link/trunk_link, 不连接 motorway/trunk
    mainline_types = {"motorway", "trunk"}
    ramp_types = {"motorway_link", "trunk_link"}
    dead_end_nodes = set()
    for nid, hw_set in node_highway_types.items():
        if hw_set <= ramp_types and not (hw_set & mainline_types):
            dead_end_nodes.add(int(nid))

    log(f"  匝道 dead-end 节点: {len(dead_end_nodes)} 个")
    if not dead_end_nodes:
        return {}

    # --- 扫描 PBF: 找包含 dead-end 节点的非高速 way ---
    class RoadFinder(osmium.SimpleHandler):
        def __init__(self, target_nodes):
            super().__init__()
            self.target_nodes = target_nodes
            self.node_roads = defaultdict(set)  # node_id -> {road_name, ...}
            self._count = 0
            self._hit = 0

        def way(self, w):
            self._count += 1
            if self._count % 1_000_000 == 0:
                sys.stdout.write(
                    f"\r  [PBF] 已扫描 {self._count:,} ways, "
                    f"命中 {self._hit}..."
                )
                sys.stdout.flush()

            hw = w.tags.get("highway", "")
            # 跳过高速本身, 只要地方道路
            if hw in ("motorway", "motorway_link", "trunk", "trunk_link", ""):
                return

            name = w.tags.get("name", "")
            if not name:
                return

            node_refs = [n.ref for n in w.nodes]
            for nref in node_refs:
                if nref in self.target_nodes:
                    self.node_roads[nref].add(name)
                    self._hit += 1

    log(f"  扫描 PBF: {pbf_path}")
    finder = RoadFinder(dead_end_nodes)
    finder.apply_file(pbf_path)
    sys.stdout.write("\n")

    matched_nodes = len(finder.node_roads)
    log(f"  找到 {matched_nodes}/{len(dead_end_nodes)} 个 dead-end 节点的连接道路")

    # 转为 str key
    node_to_road_names = {
        str(nid): sorted(names)
        for nid, names in finder.node_roads.items()
    }

    # 示例输出
    for nid, names in list(node_to_road_names.items())[:5]:
        log(f"    node {nid}: {names}")

    return node_to_road_names


def enrich_edge_osm_with_origins(edge_osm, edge_info_df, way_tags,
                                  node_to_road_names):
    """将匝道出发地道路名写入 edge_osm 的 origin_road 字段."""
    if not node_to_road_names:
        return

    log("  将出发地道路名写入 edge_osm...")
    enriched = 0

    for _, row in edge_info_df.iterrows():
        ek = row["edge_key"]
        osmid_str = str(row["osmid"])

        # 检查是否是匝道
        osm_info = edge_osm.get(ek, {})
        hw = osm_info.get("highway", "")
        if "link" not in hw:
            continue

        # 收集该 edge 所有 way 的端点
        origin_names = set()
        for oid_str in osmid_str.split(";"):
            oid_str = oid_str.strip()
            try:
                oid = int(oid_str)
            except ValueError:
                continue
            wt = way_tags.get(oid)
            if wt is None:
                continue
            for nid in [wt["nodes"][0], wt["nodes"][-1]]:
                names = node_to_road_names.get(nid, [])
                origin_names.update(names)

        if origin_names:
            edge_osm[ek]["origin_road"] = "; ".join(sorted(origin_names))
            enriched += 1

    log(f"  {enriched} 条匝道 edge 直接找到 origin_road")

    # --- 沿匝道链传播 origin_road ---
    # 构建匝道 edge 的节点邻接: node → [edge_key, ...]
    ramp_node_edges = defaultdict(list)
    ramp_edges_uv = {}  # edge_key -> (u, v)
    for _, row in edge_info_df.iterrows():
        ek = row["edge_key"]
        hw = edge_osm.get(ek, {}).get("highway", "")
        if "link" not in hw:
            continue
        u, v = str(row["u"]), str(row["v"])
        ramp_edges_uv[ek] = (u, v)
        ramp_node_edges[u].append(ek)
        ramp_node_edges[v].append(ek)

    # BFS 传播: 从有 origin_road 的 edge 向连接的 ramp edge 扩散
    propagated = 0
    changed = True
    while changed:
        changed = False
        for ek, (u, v) in ramp_edges_uv.items():
            if edge_osm.get(ek, {}).get("origin_road"):
                continue  # 已有, 跳过

            # 查看相邻 ramp edge 是否有 origin_road
            neighbor_origins = set()
            for node in (u, v):
                for neighbor_ek in ramp_node_edges[node]:
                    if neighbor_ek == ek:
                        continue
                    nr = edge_osm.get(neighbor_ek, {}).get("origin_road", "")
                    if nr:
                        neighbor_origins.add(nr)

            if neighbor_origins:
                edge_osm[ek]["origin_road"] = "; ".join(sorted(neighbor_origins))
                propagated += 1
                changed = True

    log(f"  {propagated} 条匝道 edge 通过链传播获得 origin_road")
    log(f"  合计: {enriched + propagated} 条匝道有 origin_road")


# =============================================================================
# Step 3: 逐 sensor 审查
# =============================================================================
def review_all_sensors(sensors_df, mappings, unmatched, edge_osm,
                       phase0_details_df=None, use_llm=True):
    """对每个已匹配 sensor 进行审查."""
    log("Step 3: 逐 sensor 审查...")

    results = []

    # 合并所有匹配
    all_matched = []
    for stype, df in mappings.items():
        if len(df) == 0:
            continue
        for _, r in df.iterrows():
            all_matched.append({
                "sensor_id": int(r["sensor_id"]),
                "link_id": r["link_id"],
                "type": stype,
                "fwy": r["fwy"],
                "dir": r["dir"],
                "match_dist": r.get("dist", r.get("d4", None)),
            })

    log(f"  已匹配 sensor 总数: {len(all_matched)}")

    phase0_lookup = {}
    if phase0_details_df is not None and len(phase0_details_df) > 0:
        phase0_lookup = {
            int(row["sensor_id"]): row.to_dict()
            for _, row in phase0_details_df.iterrows()
        }

    # sensor_id → PeMS 名称等信息
    sensor_info = {}
    for _, s in sensors_df.iterrows():
        sensor_id = int(s["ID"])
        phase0_info = phase0_lookup.get(sensor_id, {})
        sensor_info[int(s["ID"])] = {
            "pems_name": s.get("Name", ""),
            "fwy": s.get("Fwy", ""),
            "dir": s.get("Dir", ""),
            "type": s.get("Type", ""),
            "lat": s.get("Latitude", 0),
            "lon": s.get("Longitude", 0),
            "abs_pm": s.get("Abs_PM", ""),
            "raw_lat": s.get("raw_lat", s.get("Latitude", 0)),
            "raw_lon": s.get("raw_lon", s.get("Longitude", 0)),
            "corrected_lat": s.get("corrected_lat", s.get("Latitude", 0)),
            "corrected_lon": s.get("corrected_lon", s.get("Longitude", 0)),
            "correction_method": s.get("correction_method", ""),
            "correction_dist_m": s.get("correction_dist_m", np.nan),
            "shn_correction_applied": s.get("shn_correction_applied", False),
            "shn_pm_match": s.get("shn_pm_match", False),
            "shn_route": s.get("shn_route", ""),
            "shn_direction": s.get("shn_direction", ""),
            "shn_align_code": s.get("shn_align_code", ""),
            "shn_pmrouteid": s.get("shn_pmrouteid", ""),
            "phase0_status": phase0_info.get("status", ""),
            "phase0_target_lat": phase0_info.get("shn_target_lat", np.nan),
            "phase0_target_lon": phase0_info.get("shn_target_lon", np.nan),
        }

    # --- 预收集需要 LLM 地名审查的 sensor 对 ---
    name_pairs = []  # [{id, pems, osm, origin, ...}, ...]
    llm_skipped = {}  # sensor_id -> reason (信息缺失, 无法发给 LLM)

    for m in all_matched:
        if m["type"] not in ("OR", "FR"):
            continue
        sid = m["sensor_id"]
        sinfo = sensor_info.get(sid, {})
        pems_name = sinfo.get("pems_name", "")

        # 从 PeMS 名称提取地名
        name_clean = re.sub(r"^\d+[-–]\w\s+", "", pems_name) if pems_name else ""
        name_clean = re.sub(r"\s*(On|Off)[-\s]*(Ramp|ramp).*$", "", name_clean)
        name_clean = name_clean.strip()

        ek = m["link_id"]
        osm = edge_osm.get(ek, {})
        osm_dest = osm.get("destination", "")
        osm_dest_street = osm.get("destination_street", "")
        osm_name = osm.get("name", "")
        osm_dest_ref = osm.get("destination_ref", "")
        origin_road = osm.get("origin_road", "")
        combined_osm = "; ".join(filter(None, [osm_dest, osm_dest_street, osm_name]))

        # 信息缺失检查
        if not name_clean or len(name_clean) <= 2:
            llm_skipped[sid] = "PeMS名称缺失或过短, 无法进行地名审查"
            continue
        if not combined_osm.strip() and not origin_road.strip():
            llm_skipped[sid] = "OSM destination 和 origin_road 均为空, 无法进行地名审查"
            continue

        name_pairs.append({
            "id": sid,
            "sensor_type": m["type"],           # OR or FR
            "fwy": int(m["fwy"]),               # 所属高速编号
            "dir": m["dir"],                     # 方向
            "pems_full_name": pems_name,         # 完整 PeMS 名称
            "pems": name_clean,                  # 清理后的地名
            "osm": combined_osm,                 # OSM destination
            "osm_dest_ref": osm_dest_ref,        # OSM destination:ref
            "origin": origin_road,               # 匝道出发地道路
        })

    log(f"  需要 LLM 地名审查的 sensor: {len(name_pairs)}")
    log(f"  信息缺失跳过 LLM: {len(llm_skipped)} (将标记 severity=2)")

    # 调用 LLM 批量审查
    llm_results = {}
    if name_pairs and use_llm:
        llm_results = llm_batch_name_check(name_pairs)
        llm_ok = sum(1 for v in llm_results.values()
                     if v.get("match") is True)
        llm_bad = sum(1 for v in llm_results.values()
                      if v.get("match") is False)
        llm_unknown = sum(1 for v in llm_results.values()
                          if v.get("match") is None)
        log(f"  LLM 结果: 匹配={llm_ok}, 不匹配={llm_bad}, 未知={llm_unknown}")

    # --- 逐个审查 ---
    for m in all_matched:
        sid = m["sensor_id"]
        sinfo = sensor_info.get(sid, {})
        pems_name = sinfo.get("pems_name", "")
        ek = m["link_id"]
        osm = edge_osm.get(ek, {})

        flags = []
        severity = 0  # 0=正常, 1=轻微, 2=可疑, 3=高度可疑
        correction_dist_m = pd.to_numeric(
            sinfo.get("correction_dist_m", np.nan), errors="coerce"
        )

        # --- 审查规则 ---

        # 规则 1: FR/OR 匹配到有 destination:ref 的 link (可能是 FF connector)
        # 如果 PeMS name 中也含有高速编号, 说明该 sensor 实际是 FF, 匹配合理
        if m["type"] in ("OR", "FR"):
            dest_ref = osm.get("destination_ref", "")
            if dest_ref:
                # destination:ref 指向另一条高速 = FF connector
                own_fwy_strs = [str(int(m["fwy"])), f"CA {int(m['fwy'])}",
                                f"I {int(m['fwy'])}", f"US {int(m['fwy'])}"]
                points_to_other = not any(f in dest_ref for f in own_fwy_strs)
                if points_to_other:
                    # 检查 PeMS name 是否也指向另一条高速 (实际是 FF)
                    pems_has_hwy = bool(re.search(
                        r"\b(I[-\s]?\d|US[-\s]?\d|CA[-\s]?\d|SR[-\s]?\d"
                        r"|Hwy|Fwy|Jct|Junction|Connector)\b",
                        pems_name, re.I
                    )) if pems_name else False

                    if pems_has_hwy:
                        flags.append(
                            (1, f"OR/FR匹配到FF connector, PeMS名称含高速编号"
                            f" → 可能实际为FF (dest_ref={dest_ref})")
                        )
                        severity = max(severity, 1)
                    else:
                        flags.append(
                            (3, f"OR/FR匹配到FF connector, 但PeMS名称为地方道路"
                            f" (dest_ref={dest_ref})")
                        )
                        severity = max(severity, 3)

        # 规则 2: PeMS 地名与 OSM destination 不一致 (LLM 审查)
        if m["type"] in ("OR", "FR"):
            if sid in llm_results:
                llm_r2 = llm_results[sid]
                if llm_r2.get("match") is False:
                    flags.append(
                        (3, f"LLM判定地名不匹配: {llm_r2.get('reason', '')}")
                    )
                    severity = max(severity, 3)
                elif llm_r2.get("match") is None:
                    flags.append(
                        (2, f"LLM审查未覆盖: {llm_r2.get('reason', '')}")
                    )
                    severity = max(severity, 2)
            elif sid in llm_skipped:
                flags.append(
                    (2, f"信息缺失跳过LLM: {llm_skipped[sid]}")
                )
                severity = max(severity, 2)

        # 规则 3: 匹配距离偏大
        dist = m.get("match_dist")
        if dist is not None:
            if m["type"] == "ML" and dist > 5e-5:  # > ~5m
                flags.append((1, f"ML匹配距离偏大: {dist:.6f}° ≈ {dist*111000:.0f}m"))
                severity = max(severity, 1)
            elif m["type"] in ("OR", "FR") and dist > 0.005:  # > ~550m
                flags.append((2, f"OR/FR匹配距离偏大: {dist:.6f}° ≈ {dist*111000:.0f}m"))
                severity = max(severity, 2)

        # 规则 4: Phase 0 坐标修正位移过大, 需要人工关注
        if pd.notna(correction_dist_m):
            if correction_dist_m >= 250:
                flags.append((2, f"Phase0坐标修正位移较大: {correction_dist_m:.1f}m"))
                severity = max(severity, 2)
            elif correction_dist_m >= 100:
                flags.append((1, f"Phase0坐标修正位移偏大: {correction_dist_m:.1f}m"))
                severity = max(severity, 1)

        # 规则 4.5: 该点最终仍使用原始坐标参与匹配
        phase0_status = sinfo.get("phase0_status", "")
        correction_method = str(sinfo.get("correction_method", ""))
        shn_applied = bool(sinfo.get("shn_correction_applied", False))
        if not shn_applied and phase0_status in {"distance_too_large", "no_postmile_bracket",
                                                 "no_align_candidates", "no_shn_candidates"}:
            flags.append((1, f"Phase0未应用修正, 仍用原始坐标匹配: status={phase0_status}, method={correction_method}"))
            severity = max(severity, 1)

        # 规则 5: ML 匹配到 motorway_link (应该在 motorway 上)
        if m["type"] == "ML":
            hw = osm.get("highway", "")
            if "link" in hw:
                flags.append((3, f"ML匹配到ramp类型: highway={hw}"))
                severity = max(severity, 3)

        # 规则 6: OR/FR 匹配到 motorway (应该在 motorway_link 上)
        if m["type"] in ("OR", "FR"):
            hw = osm.get("highway", "")
            if hw in ("motorway", "trunk") and "link" not in hw:
                flags.append((3, f"OR/FR匹配到主线: highway={hw}"))
                severity = max(severity, 3)

        # 规则 7: PeMS name 含 "Jct" 或 "Fwy" 但匹配到非 FF link
        if pems_name and m["type"] in ("OR", "FR"):
            if re.search(r"\b(Jct|Junction|Connector)\b", pems_name, re.I):
                dest_ref = osm.get("destination_ref", "")
                if not dest_ref:
                    flags.append(
                        (1, f"PeMS名称含Junction/Connector但link无destination:ref")
                    )
                    severity = max(severity, 1)

        # LLM 审查结果
        llm = llm_results.get(sid, {})
        llm_name_match = llm.get("match")  # True/False/None
        llm_reason = llm.get("reason", "")

        # flags 只保留最高 severity 的 reason
        if flags:
            max_sev = max(s for s, _ in flags)
            top_reasons = [reason for s, reason in flags if s == max_sev]
            flags_str = " | ".join(top_reasons)
        else:
            flags_str = ""

        results.append({
            "sensor_id": sid,
            "type": m["type"],
            "fwy": m["fwy"],
            "dir": m["dir"],
            "pems_name": pems_name,
            "link_id": ek,
            "match_dist": dist,
            "raw_lat": sinfo.get("raw_lat", np.nan),
            "raw_lon": sinfo.get("raw_lon", np.nan),
            "matched_lat": sinfo.get("lat", np.nan),
            "matched_lon": sinfo.get("lon", np.nan),
            "corrected_lat": sinfo.get("corrected_lat", np.nan),
            "corrected_lon": sinfo.get("corrected_lon", np.nan),
            "phase0_status": phase0_status,
            "phase0_correction_method": correction_method,
            "phase0_correction_dist_m": correction_dist_m,
            "shn_correction_applied": shn_applied,
            "shn_pm_match": sinfo.get("shn_pm_match", False),
            "shn_route": sinfo.get("shn_route", ""),
            "shn_direction": sinfo.get("shn_direction", ""),
            "shn_align_code": sinfo.get("shn_align_code", ""),
            "shn_pmrouteid": sinfo.get("shn_pmrouteid", ""),
            "osm_highway": osm.get("highway", ""),
            "osm_ref": osm.get("ref", ""),
            "osm_name": osm.get("name", ""),
            "osm_destination": osm.get("destination", ""),
            "osm_destination_ref": osm.get("destination_ref", ""),
            "osm_origin_road": osm.get("origin_road", ""),
            "llm_name_match": llm_name_match,
            "llm_reason": llm_reason,
            "severity": severity,
            "flags": flags_str,
        })

    results_df = pd.DataFrame(results)
    log(f"  审查完成: {len(results_df)} 条记录")

    # 统计
    sev_counts = results_df["severity"].value_counts().sort_index()
    for sev, cnt in sev_counts.items():
        label = {0: "正常", 1: "轻微", 2: "可疑", 3: "高度可疑"}[sev]
        log(f"    severity={sev} ({label}): {cnt}")

    return results_df


# =============================================================================
# Step 4: 审查未匹配 sensor
# =============================================================================
def review_unmatched(sensors_df, unmatched, phase0_details_df=None):
    """汇总未匹配 sensor 的信息."""
    log("Step 4: 未匹配 sensor 汇总...")

    phase0_lookup = {}
    if phase0_details_df is not None and len(phase0_details_df) > 0:
        phase0_lookup = {
            int(row["sensor_id"]): row.to_dict()
            for _, row in phase0_details_df.iterrows()
        }

    records = []
    for stype, df in unmatched.items():
        if len(df) == 0:
            continue
        for _, r in df.iterrows():
            sid = int(r["sensor_id"])
            srow = sensors_df[sensors_df["ID"] == sid]
            pems_name = srow["Name"].iloc[0] if len(srow) > 0 else ""
            phase0_info = phase0_lookup.get(sid, {})
            lat = srow["Latitude"].iloc[0] if len(srow) > 0 else np.nan
            lon = srow["Longitude"].iloc[0] if len(srow) > 0 else np.nan
            raw_lat = srow["raw_lat"].iloc[0] if len(srow) > 0 and "raw_lat" in srow else lat
            raw_lon = srow["raw_lon"].iloc[0] if len(srow) > 0 and "raw_lon" in srow else lon

            records.append({
                "sensor_id": sid,
                "type": stype,
                "fwy": r.get("fwy", ""),
                "dir": r.get("dir", ""),
                "pems_name": pems_name,
                "raw_lat": raw_lat,
                "raw_lon": raw_lon,
                "matched_lat": lat,
                "matched_lon": lon,
                "phase0_status": phase0_info.get("status", ""),
                "phase0_correction_method": srow["correction_method"].iloc[0]
                if len(srow) > 0 and "correction_method" in srow else "",
                "phase0_correction_dist_m": srow["correction_dist_m"].iloc[0]
                if len(srow) > 0 and "correction_dist_m" in srow else np.nan,
                "shn_correction_applied": srow["shn_correction_applied"].iloc[0]
                if len(srow) > 0 and "shn_correction_applied" in srow else False,
                "reason": r.get("reason", ""),
                "best_dist": r.get("best_dist", r.get("best_d4", None)),
                "best_link": r.get("best_link", ""),
                "n_candidates": r.get("n_candidates",
                                       r.get("n_ramps_total", "")),
            })

    unmatched_df = pd.DataFrame(records)
    log(f"  未匹配 sensor: {len(unmatched_df)} 个")
    if len(unmatched_df) > 0:
        reasons = unmatched_df["reason"].value_counts()
        for reason, cnt in reasons.items():
            log(f"    {reason}: {cnt}")

    return unmatched_df


# =============================================================================
# 主流程
# =============================================================================
def run_review(output_dir, osm_path, use_llm=True, pbf_path=None):
    """运行审查."""
    log("=" * 70)
    log("传感器匹配审查")
    log("=" * 70)
    if not use_llm:
        log("  (已禁用 LLM 地名审查)")

    osm_path = resolve_osm_path(output_dir, osm_path)

    # Step 1
    sensors_df, edge_info_df, mappings, unmatched, way_tags, phase0_details_df, ff_manual_df = \
        load_all_data(output_dir, osm_path)

    # Step 2
    edge_osm = build_edge_to_osm(edge_info_df, way_tags)

    # Step 2.5: 提取匝道出发地 (可选, 需要 PBF)
    if pbf_path and os.path.exists(pbf_path):
        node_to_road_names = extract_ramp_origins(way_tags, edge_info_df, pbf_path)
        enrich_edge_osm_with_origins(edge_osm, edge_info_df, way_tags,
                                     node_to_road_names)
    else:
        if pbf_path:
            log(f"  PBF 文件不存在: {pbf_path}", "WARN")
        else:
            log("  未提供 --pbf, 跳过匝道出发地提取")

    # Step 3
    results_df = review_all_sensors(sensors_df, mappings, unmatched, edge_osm,
                                    phase0_details_df=phase0_details_df,
                                    use_llm=use_llm)

    # Step 4
    unmatched_df = review_unmatched(sensors_df, unmatched,
                                    phase0_details_df=phase0_details_df)

    # 保存
    results_path = os.path.join(output_dir, "review_results.csv")
    results_df.to_csv(results_path, index=False)
    log(f"\n全部结果: {results_path}")

    suspicious = results_df[results_df["severity"] >= 2].copy()
    suspicious_path = os.path.join(output_dir, "review_suspicious.csv")
    suspicious.to_csv(suspicious_path, index=False)
    log(f"可疑项 (severity>=2): {suspicious_path} ({len(suspicious)} 条)")

    # --- 生成人工修正文件 (可疑匹配 + 未匹配 + FF) ---

    # 从原始 mapping 获取完整匹配信息 (含 node_id 等)
    all_mapping_rows = {}
    for stype, df in mappings.items():
        if len(df) == 0:
            continue
        for _, r in df.iterrows():
            all_mapping_rows[int(r["sensor_id"])] = {
                "link_id": r.get("link_id", ""),
                "node_id": r.get("node_id", ""),
                "dist": r.get("dist", r.get("d4", "")),
            }

    # sensor_id → PeMS 信息
    sensor_lookup = {}
    for _, s in sensors_df.iterrows():
        sensor_lookup[int(s["ID"])] = s

    override_rows = []

    # --- Part 1: 可疑匹配 (severity >= 2) ---
    for _, r in suspicious.iterrows():
        sid = int(r["sensor_id"])
        orig = all_mapping_rows.get(sid, {})
        srow = sensor_lookup.get(sid, {})

        override_rows.append({
            "category": "可疑匹配",
            "sensor_id": sid,
            "link_id": orig.get("link_id", r.get("link_id", "")),
            "node_id": orig.get("node_id", ""),
            "dist": orig.get("dist", r.get("match_dist", "")),
            "fwy": int(r["fwy"]),
            "dir": r["dir"],
            "raw_lat": srow.get("raw_lat", srow.get("Latitude", "")),
            "raw_lon": srow.get("raw_lon", srow.get("Longitude", "")),
            "lat": srow.get("Latitude", ""),
            "lon": srow.get("Longitude", ""),
            "phase0_status": r.get("phase0_status", ""),
            "phase0_correction_method": r.get("phase0_correction_method", ""),
            "phase0_correction_dist_m": r.get("phase0_correction_dist_m", ""),
            "shn_correction_applied": r.get("shn_correction_applied", ""),
            "action": "",
            "new_link_id": "",
            "new_node_id": "",
            "type": r["type"],
            "pems_name": r["pems_name"],
            "severity": r["severity"],
            "flags": r["flags"],
            "osm_highway": r.get("osm_highway", ""),
            "osm_destination": r.get("osm_destination", ""),
            "osm_destination_ref": r.get("osm_destination_ref", ""),
            "osm_origin_road": r.get("osm_origin_road", ""),
            "llm_name_match": r.get("llm_name_match", ""),
            "llm_reason": r.get("llm_reason", ""),
        })

    # --- Part 2: 未匹配 ML/OR/FR ---
    for _, r in unmatched_df.iterrows():
        sid = int(r["sensor_id"])
        srow = sensor_lookup.get(sid, {})

        override_rows.append({
            "category": "未匹配",
            "sensor_id": sid,
            "link_id": r.get("best_link", ""),
            "node_id": "",
            "dist": r.get("best_dist", ""),
            "fwy": r.get("fwy", srow.get("Fwy", "")),
            "dir": r.get("dir", srow.get("Dir", "")),
            "raw_lat": r.get("raw_lat", ""),
            "raw_lon": r.get("raw_lon", ""),
            "lat": r.get("matched_lat", srow.get("Latitude", "")),
            "lon": r.get("matched_lon", srow.get("Longitude", "")),
            "phase0_status": r.get("phase0_status", ""),
            "phase0_correction_method": r.get("phase0_correction_method", ""),
            "phase0_correction_dist_m": r.get("phase0_correction_dist_m", ""),
            "shn_correction_applied": r.get("shn_correction_applied", ""),
            "action": "",
            "new_link_id": "",
            "new_node_id": "",
            "type": r["type"],
            "pems_name": r.get("pems_name", ""),
            "severity": "",
            "flags": f"未匹配: {r.get('reason', '')} (候选数={r.get('n_candidates', '')})",
            "osm_highway": "",
            "osm_destination": "",
            "osm_destination_ref": "",
            "osm_origin_road": "",
            "llm_name_match": "",
            "llm_reason": "",
        })

    # --- Part 3: FF 传感器 (全部待手动分配) ---
    ff_source_df = ff_manual_df if len(ff_manual_df) > 0 else sensors_df[sensors_df["Type"] == "FF"]
    for _, s in ff_source_df.iterrows():
        sid = int(s.get("sensor_id", s.get("ID")))
        override_rows.append({
            "category": "FF待分配",
            "sensor_id": sid,
            "link_id": s.get("assigned_link", s.get("link_id", "")),
            "node_id": s.get("node_id", ""),
            "dist": s.get("dist", ""),
            "fwy": s.get("fwy", s.get("Fwy", "")),
            "dir": s.get("dir", s.get("Dir", "")),
            "raw_lat": s.get("raw_lat", s.get("Latitude", "")),
            "raw_lon": s.get("raw_lon", s.get("Longitude", "")),
            "lat": s.get("lat", s.get("Latitude", "")),
            "lon": s.get("lon", s.get("Longitude", "")),
            "phase0_status": s.get("status", s.get("phase0_status", "")),
            "phase0_correction_method": s.get("correction_method", s.get("phase0_correction_method", "")),
            "phase0_correction_dist_m": s.get("correction_dist_m", s.get("phase0_correction_dist_m", "")),
            "shn_correction_applied": s.get("shn_correction_applied", ""),
            "action": "",
            "new_link_id": "",
            "new_node_id": "",
            "type": "FF",
            "pems_name": s.get("Name", s.get("name", "")),
            "severity": "",
            "flags": "FF传感器, 需手动分配link",
            "osm_highway": "",
            "osm_destination": "",
            "osm_destination_ref": "",
            "osm_origin_road": "",
            "llm_name_match": "",
            "llm_reason": "",
        })

    override_df = pd.DataFrame(override_rows)
    override_path = os.path.join(output_dir, "review_override.csv")
    override_df.to_csv(override_path, index=False)

    n_suspicious = len(suspicious)
    n_unmatched = len(unmatched_df)
    n_ff = len(ff_source_df)
    log(f"人工修正文件: {override_path}")
    log(f"  可疑匹配: {n_suspicious}, 未匹配: {n_unmatched}, FF: {n_ff}, "
        f"合计: {len(override_df)}")
    log(f"  使用方法: 填写 action 列 (keep/change/remove/assign)")
    log(f"           action=change/assign 时填写 new_link_id / new_node_id")

    unmatched_path = os.path.join(output_dir, "review_unmatched.csv")
    unmatched_df.to_csv(unmatched_path, index=False)
    log(f"未匹配: {unmatched_path} ({len(unmatched_df)} 条)")

    # 打印可疑 TOP 项
    if len(suspicious) > 0:
        log(f"\n{'='*70}")
        log(f"可疑匹配 TOP 20:")
        log(f"{'='*70}")
        top = suspicious.sort_values("severity", ascending=False).head(20)
        for _, r in top.iterrows():
            log(f"\n  sensor {int(r['sensor_id'])} ({r['type']}) "
                f"Fwy{int(r['fwy'])}{r['dir']}")
            log(f"    PeMS: {r['pems_name']}")
            log(f"    Link: {r['link_id']}")
            log(f"    OSM:  hw={r['osm_highway']}, dest={r['osm_destination']}, "
                f"dest_ref={r['osm_destination_ref']}")
            log(f"    问题: {r['flags']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="传感器匹配审查")
    parser.add_argument("output_dir", help="Pipeline 输出目录")
    parser.add_argument("osm_file", help="OSM 文件路径 (.osm)")
    parser.add_argument("--pbf", type=str, default=None,
                        help="完整 PBF 文件路径 (用于提取匝道出发地道路名)")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 地名审查 (仅用规则 1,3,5-7)")

    args = parser.parse_args()
    run_review(args.output_dir, args.osm_file,
               use_llm=not args.no_llm, pbf_path=args.pbf)

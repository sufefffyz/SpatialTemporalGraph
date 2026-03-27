import osmnx as ox

# 方法 1: 如果 osmnx 支持直接读 pbf（需要 osmnx >= 2.0）
G = ox.graph_from_xml("california-latest.osm.pbf")

# 如果报错，先用 osmium 裁剪 + 转 osm 格式
# pip install osmium
# 或 sudo apt install osmium-tool
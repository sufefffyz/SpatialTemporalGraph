import osmnx as ox
print(f"osmnx 版本: {ox.__version__}")

ox.settings.timeout = 600
ox.settings.use_cache = True
cf = '["highway"~"motorway|motorway_link"]'

# 用两种格式都试一下
south, north = 38.13, 38.55
west, east = -122.22, -121.45

# osmnx 2.x 格式: (west, south, east, north)
try:
    g = ox.graph_from_bbox(bbox=(west, south, east, north),
                           network_type="drive", custom_filter=cf, retain_all=True)
    print(f"格式 (W,S,E,N) 成功: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边")
except Exception as e:
    print(f"格式 (W,S,E,N) 失败: {e}")
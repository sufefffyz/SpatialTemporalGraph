# 可接入数据集扫描

目标：为自适应阈值 / K64 候选池动态开关实验寻找能获得节点坐标或两两距离的数据集。优先级按“马上能在 BasicTS 接入”和“能支撑论文泛化实验”排序。

## 结论

| 优先级 | 数据集 | 规模和时间粒度 | 距离/坐标可得性 | 当前仓库状态 | 建议 |
|---|---|---:|---|---|---|
| P0 | SD / GLA / GBA / CA, LargeST | SD 716；GLA 3834；GBA 2352；CA 8600；5min 原始，可切 15min | 官方提供 sensor metadata 和基于路网距离的 adjacency；我们也已有 SD OSRM/GSP 距离与 K64 图 | SD 已跑；180 有 GLA/GBA 5min full；SD 图变体已存在 | 主实验仍以 SD 15min 对齐 LargeST；泛化可选 GBA 或 GLA 子集 |
| P0 | METR-LA | 207 nodes, 5min | DCRNN 官方 sensor_graph 有 `distances_la_2012.csv` 和 `graph_sensor_locations.csv` | BasicTS 有多个模型配置；180 有数据集 | 最快补充 bench，适合 sanity/generalization |
| P0 | PEMS-BAY | 325 nodes, 5min | DCRNN 官方 sensor_graph 有 `distances_bay_2017.csv` 和 `graph_sensor_locations_bay.csv` | BasicTS 有多个模型配置；180 有数据集 | 与 METR-LA 配对，计算成本低 |
| P1 | PEMS04 / PEMS08 | PEMS04 307；PEMS08 170；5min | 标准 ASTGNN 风格文件给 directed `from,to,cost`；PEMS04F 另有地理距离版本 | BasicTS 有 PEMS04/08 配置；180 有 PEMS04/08；本地只有 PEMS04F desc | 可作为 flow 数据补充；要先确认 cost 与坐标/距离定义 |
| P1 | KnowAir-V2 | 2016-2023，空气质量，多站点 | 数据集面向 AQ forecasting，通常可由站点经纬度构造 Haversine 距离 | BasicTS 暂无配置 | 适合跨领域泛化，但要新写预处理 |
| P2 | Weather2K | 2130 weather stations，hourly，40896 steps | 论文说明含位置常量，可直接算地理距离 | BasicTS 暂无配置 | 节点多、跨领域；先不作为第一轮 |
| P2 | Traffic4cast 2022 | London/Madrid/Melbourne，15min loop counters + road graph edges | 提供 OSM 简化 road graph、loop counter 映射、edge labels | BasicTS 暂无配置 | 数据结构是边预测/图边状态，接入成本高，不适合作为当前 MVP |
| P3 | CHRONOGRAPH | microservice graph time series | 有机器可读依赖图，但不是地理距离 | BasicTS 暂无配置 | 不适合“距离阈值”主张，只能做结构图泛化 |

## 近期工作线索

- FlowNet, NeurIPS 2025：使用 PEMS04F、SINPA 等动态系统数据，方法上和“自适应 spatial mask / 动态传播半径”相关，但它更偏 flow redistribution；可作为方法对照，不是最便宜的数据扩展。
- DPGNet, ICLR 2026 submission：强调 plug-and-play dynamic graph learner 和 weak-connection suppression，可作为动态图学习相关工作，但其 benchmark 需要看代码释放后的数据清单。
- PI-STGNN, ICML 2025：使用 LibCity Traffic Dataset，说明近期顶会仍接受 LibCity/PeMS 类交通 benchmark。
- Traffic4cast 2022/NeurIPS：路网、计数器、edge label 很完整，但任务形态和我们当前 sensor-level forecasting 不同。

## 当前建议

1. 先做 METR-LA + PEMS-BAY：节点少、距离和坐标最可靠，BasicTS 配置已有，能快速测试动态图模块是否只在 SD 有效。
2. 再做 PEMS04F 或标准 PEMS04：FlowNet 相关性强，但要先统一 scaler、split、cost/geo-distance 的解释。
3. 跨领域只保留 KnowAir-V2 作为后续可选项：如果交通数据上有稳定增益，再证明“坐标距离图”不是只对路网有效。

## 主要来源

- LargeST official repo: https://github.com/liuxu77/LargeST
- DCRNN sensor graph files: https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph
- LibCity dataset converters: https://github.com/LibCity/Bigscity-LibCity-Datasets
- FlowNet OpenReview: https://openreview.net/forum?id=3jjeQNeCsl
- KnowAir / KnowAir-V2 repo: https://github.com/shuowang-ai/PM2.5-GNN
- Weather2K: https://arxiv.org/abs/2302.10493
- Traffic4cast 2022: https://github.com/iarai/NeurIPS2022-traffic4cast

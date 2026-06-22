# 自建 PEMS Benchmark Inventory

更新时间：2026-06-08  
目的：追踪本地和服务器上的自建 PEMS 数据集、地图可视化、以及 ML / OR / FR / HV 传感器分布，判断是否适合作为动态图阈值研究 benchmark。

## 结论

当前材料分成两层，不能混用：

1. **可直接训练的 BasicTS 数据集**：180 服务器上目前只确认了 D3/D4。

- `BasicTS/datasets/PEMSD3_2025_full_phys`
- `BasicTS/datasets/PEMSD3_2025_full_distthres`
- `BasicTS/datasets/PEMSD4_2025_full_phys`
- `BasicTS/datasets/PEMSD4_2025_full_distthres`

它们有完整的 `sensor_catalog.csv`、训练数据、图文件和类型标签。

2. **已整理的自建 PEMS 图构建归档**：180 服务器上确认存在 D3-D8、D10-D12 的完整构图输出，路径为：

`/home/yuzhang_fei/stg_artifacts_archive/pems_graph_construction/outputs_20260511/`

这些归档包含 `exports/basicts/{adj_mx,distance_mx,sensor_ids}`、`sensor_graph_map.html`、`network_map.html`、`phase7_sensor_info.csv`、`graph_summary.json` 等。当前没有扫到 D9。

目前没在本地、178 或 180 上找到 D4-D12 的 station raw / station_5min 全年时序文件；本地只看到 D3 的 2025-06 一个月 station raw。因此 D3-D8/D10-D12 现在是“图和元数据已归档”，但还需要补齐/重新下载 5min 时序并生成 BasicTS 格式后才能作为统一 benchmark 训练。

本地只有打包后的 `TRAFFIC_VOLUME_{5MIN,15MIN,30MIN,1H}` 和一个 metadata-only skeleton，缺少可直接对齐的 `sensor_catalog.csv` / ramp 类型地图，因此本地版本暂时更适合作为缓存或轻量引用，不适合直接做类型分层 benchmark。

## 数据集位置

### 本地

| 路径 | 状态 | 备注 |
|---|---|---|
| `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/datasets/TRAFFIC_VOLUME_5MIN` | 存在 | 5135 节点，5min，3 特征，无 catalog/map |
| `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/datasets/TRAFFIC_VOLUME_15MIN` | 存在 | 5135 节点，15min，3 特征，无 catalog/map |
| `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/causalrivers/pems_metadata_skeleton` | 存在 | ML-only metadata skeleton，880 节点，不是完整 road graph |
| `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/datasets/PEMS04F` | 存在 | FlowNet/PEMS04F 相关，不是自建 D3/D4 全量 benchmark |

本地 `TRAFFIC_VOLUME_15MIN`：

```json
{
  "shape": [11817, 5135, 3],
  "frequency (minutes)": 15,
  "feature_description": ["traffic volume", "time of day", "day of week"],
  "has_graph": true
}
```

本地 `TRAFFIC_VOLUME_5MIN`：

```json
{
  "shape": [35449, 5135, 3],
  "frequency (minutes)": 5,
  "feature_description": ["traffic volume", "time of day", "day of week"],
  "has_graph": true
}
```

### 服务器 180

| 路径 | 状态 | 备注 |
|---|---|---|
| `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/datasets/PEMSD3_2025_full_phys` | 完整 | 物理图，自建 D3 全量稳定节点 |
| `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/datasets/PEMSD3_2025_full_distthres` | 完整 | 同节点，距离阈值图 |
| `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/datasets/PEMSD4_2025_full_phys` | 完整 | 物理图，自建 D4 全量稳定节点 |
| `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/datasets/PEMSD4_2025_full_distthres` | 完整 | 同节点，距离阈值图 |
| `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/analysis_outputs/sensor_diagnostics` | 完整 | D3/D4 按类型诊断图和 CSV |
| `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full` | 完整 | D3 构图阶段产物和地图 |
| `/home/yuzhang_fei/stg_artifacts_archive/pems_graph_construction/outputs_20260511/PEMSD{3,4,5,6,7,8,10,11,12}` | 图构建归档完整 | 有 basicts 图导出、地图、phase0/phase7 sensor info |
| `/home/yuzhang_fei/stg_artifacts_archive/pems_output/output_20260511` | 分析归档 | 有 topology / direction inference / station tracking / OSRM validation 图表 |

### 服务器 178

178 当前主要是 `PEMS-BAY`、`PEMS04F`、空气质量和 adaptive threshold 小数据集结果，没有找到 `PEMSD3_2025_full_*` / `PEMSD4_2025_full_*` 自建全量 benchmark。

## 基本规模

| Dataset | Shape | Frequency | Nodes | Raw missing ratio | Graph variants |
|---|---:|---:|---:|---:|---|
| `PEMSD3_2025_full_phys` | `[104544, 1434, 1]` | 5min | 1434 | 0.0877 | physical |
| `PEMSD3_2025_full_distthres` | `[104544, 1434, 1]` | 5min | 1434 | 0.0877 | distance threshold |
| `PEMSD4_2025_full_phys` | `[104544, 3490, 1]` | 5min | 3490 | 0.2203 | physical |
| `PEMSD4_2025_full_distthres` | `[104544, 3490, 1]` | 5min | 3490 | 0.2203 | distance threshold |

`desc.json` 中 `num_features=1`，但 `mvp_metadata` 记录实际 assembled 特征为：

```text
flow, time of day, day of week
```

训练配置是否使用 1 个特征还是 3 个特征，需要在具体 config 中继续核对。

## D3-D12 图构建归档规模

路径：`/home/yuzhang_fei/stg_artifacts_archive/pems_graph_construction/outputs_20260511/`

| Dataset | Phase7 nodes | Directed edges | ML | OR / on-ramp | FR / off-ramp | HV / HOV | BasicTS graph export | Maps |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `PEMSD3` | 1434 | 1611 | 765 | 406 | 263 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD4` | 3490 | 4545 | 2007 | 888 | 595 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD5` | 513 | 657 | 380 | 62 | 71 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD6` | 726 | 786 | 636 | 88 | 2 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD7` | 3712 | 5320 | 1875 | 1071 | 766 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD8` | 1591 | 2043 | 1054 | 343 | 194 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD10` | 1085 | 1305 | 595 | 251 | 239 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD11` | 1258 | 1851 | 724 | 304 | 230 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |
| `PEMSD12` | 1645 | 2048 | 969 | 367 | 309 | 0 | yes | `network_map.html`, `sensor_graph_map.html` |

归档中每个 district 的关键文件：

- `exports/basicts/adj_mx.npy`
- `exports/basicts/adj_mx.pkl`
- `exports/basicts/distance_mx.npy`
- `exports/basicts/distance_mx.pkl`
- `exports/basicts/sensor_ids.npy`
- `exports/analysis/sensor_index.csv`
- `exports/analysis/directed_edges.csv`
- `exports/analysis/graph_summary.json`
- `phase0_sensors.csv`
- `phase7_sensor_info.csv`
- `network_map.html`
- `sensor_graph_map.html`

注意：`graph_summary.json` 中的导出路径仍指向原始工作区 `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/outputs/...`，但当前实际文件在 `stg_artifacts_archive` 归档路径下。

## 传感器类型分布

类型解释按当前数据样例和代码使用方式：

- `ML`: mainline。
- `OR`: on-ramp。样例：`Bear Creek Rd on`, `Elk Grove Blvd to 5SB Loop`。
- `FR`: off-ramp / freeway-ramp 类出口连接。样例：`Bear Creek Rd off`, `5SB to Elk Grove Blvd`。
- `HV`: HOV。注意最终训练集没有保留 `HV`。
- `FF`: freeway-to-freeway。当前最终训练集也没有保留。

### 最终训练数据集

| Dataset | Total | ML / mainline | OR / on-ramp | FR / off-ramp | HV / HOV |
|---|---:|---:|---:|---:|---:|
| `PEMSD3_2025_full_phys` | 1434 | 765 | 406 | 263 | 0 |
| `PEMSD3_2025_full_distthres` | 1434 | 765 | 406 | 263 | 0 |
| `PEMSD4_2025_full_phys` | 3490 | 2007 | 888 | 595 | 0 |
| `PEMSD4_2025_full_distthres` | 3490 | 2007 | 888 | 595 | 0 |

### D3 原始 phase0 构图输入

`/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/phase0_sensors.csv`

| Total | ML | OR | FR | HV | FF |
|---:|---:|---:|---:|---:|---:|
| 1857 | 870 | 416 | 271 | 272 | 28 |

这说明 D3 原始 metadata 中是有 HOV 传感器的，但最终训练数据和 sensor graph 当前只保留了 `ML/OR/FR`。

### D3 phase7 sensor graph

`/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_sensor_info.csv`

| Total | ML | OR | FR | HV |
|---:|---:|---:|---:|---:|
| 1439 | 771 | 403 | 265 | 0 |

phase7 sensor graph 和最终 `PEMSD3_2025_full_*` 的 1434 个节点有 5 个节点差异，应该来自后续稳定节点筛选或数据可用性过滤。

## 分 freeway / direction 分布

### PEMSD3 top freeway

| Fwy | Count |
|---|---:|
| 80 | 372 |
| 50 | 335 |
| 99 | 257 |
| 5 | 228 |
| 51 | 83 |
| 65 | 63 |
| 20 | 41 |

方向分布：`E=395, W=378, N=329, S=332`。

### PEMSD4 top freeway

| Fwy | Count |
|---|---:|
| 101 | 1069 |
| 80 | 493 |
| 680 | 431 |
| 880 | 401 |
| 4 | 221 |
| 280 | 161 |
| 85 | 139 |

方向分布：`N=1207, S=1186, E=540, W=557`。

## 已有地图和可视化

### D3 构图地图

最重要的 D3 地图在：

- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/network_map.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/phase0_correction_map.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_graph_map.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_osrm_graph_map.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_local_graph_map.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_local_graph_map_msf.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_local_graph_map_compare_original_vs_msf.html`

还有 postmile / coordinate correction / ramp matching 相关地图：

- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/scripts/output/postmile_comparison_v3/postmile_pems_multi_basemap.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/scripts/output/d3_interpolation_correction/correction_d3_overview.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/scripts/output/layered_graph_v2/final_graph.html`
- `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/scripts/output/ff_connection_analysis/ff_connections_all.html`

### D3/D4 类型诊断图

`/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/analysis_outputs/sensor_diagnostics/PEMSD3/figures/`

- `predictability_by_type.png`
- `weekday_summary_by_type.png`
- `weekday_profiles_by_type.png`

`/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/PEMS/analysis_outputs/sensor_diagnostics/PEMSD4/figures/`

- `predictability_by_type.png`
- `weekday_summary_by_type.png`
- `weekday_profiles_by_type.png`

这些不是地图，但对 benchmark 设计很有用，因为它们已经按 `ML/OR/FR` 分析了可预测性和工作日模式。

## 类型可预测性摘要

`analysis_outputs/sensor_diagnostics/combined/type_predictability_summary.csv`

| Dataset | Type | Num sensors in diagnostic | Mean flow | Lag-1 autocorr | Prev-day profile corr | Prev-week profile corr |
|---|---|---:|---:|---:|---:|---:|
| PEMSD3 | ML | 765 | 154.17 | 0.9477 | 0.8876 | 0.9250 |
| PEMSD3 | OR | 387 | 14.85 | 0.6975 | 0.6499 | 0.6737 |
| PEMSD3 | FR | 242 | 19.33 | 0.7468 | 0.6947 | 0.7241 |
| PEMSD4 | ML | 2007 | 234.41 | 0.9669 | 0.8806 | 0.9270 |
| PEMSD4 | OR | 681 | 24.48 | 0.8163 | 0.7348 | 0.7578 |
| PEMSD4 | FR | 328 | 26.16 | 0.8341 | 0.7453 | 0.7678 |

诊断里的 OR/FR 数量少于 catalog，是因为该分析流程做了额外覆盖率/可预测性过滤；正式 benchmark 节点数应以 `sensor_catalog.csv` 为准。

## 作为当前研究 benchmark 的价值

优点：

1. 比标准 PEMS04/08 更大，而且有 `ML/OR/FR` 类型标签，适合做“不同道路功能节点需要不同邻居预算”的分析。
2. 同一节点集合有 `phys` 和 `distthres` 两套图，可以天然比较固定物理图、距离阈值图、动态阈值图。
3. D3 有完整构图可视化，方便做 case study。
4. ML 与 ramp 的时序可预测性明显不同，适合验证动态邻居机制是否对 ramp 节点更有帮助。

风险：

1. 最终数据集没有 HOV/HV，因此不能直接声称覆盖 HOV 传感器。
2. D3 phase7 local graph 的最大 WCC 只有 589 个节点，图连通性碎片化明显；这会影响图卷积和图重连结论。
3. D4 缺少像 D3 `output_d03_2.7_full` 那样完整的构图地图目录，至少当前扫描未找到。
4. `desc.json` 的 `num_features=1` 和 `mvp_metadata.assembled_num_features=3` 有口径差异，跑 baseline 前必须确认 config 的 `FORWARD_FEATURES`。

## 建议下一步

1. 先把 180 上的 `PEMSD3_2025_full_*` 和 `PEMSD4_2025_full_*` 作为 canonical benchmark，不从本地 `TRAFFIC_VOLUME_*` 直接起实验。
2. 从服务器同步 `sensor_catalog.csv`、`desc.json`、关键地图 HTML 到本地归档，避免后面讨论找不到证据。
3. 补 D4 地图生成，至少需要一张按 `ML/OR/FR` 着色的 sensor map。
4. 在 benchmark 表中明确：当前最终 benchmark 包含 `mainline / on-ramp / off-ramp`，不包含 HOV。
5. 跑动态图时按类型分层报告 `ML / OR / FR / Ramp(OR+FR)`，这比只报 overall 更能体现方法价值。

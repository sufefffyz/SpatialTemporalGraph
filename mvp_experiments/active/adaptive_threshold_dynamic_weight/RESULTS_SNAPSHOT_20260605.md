# 自适应阈值图实验结果快照

采样时间：2026-06-05 11:00-11:20 CST。结果主要来自 `183.174.228.180:/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/checkpoints/*/test_metrics.json`；`183.174.228.178` 当前没有训练日志和 checkpoint 落盘。除非特别说明，表中为 best-val checkpoint 的 test overall 指标。

## 当前运行状态

| 项目 | 状态 |
|---|---|
| 178 / 180 GPU | 当前均空闲，无 screen |
| MTGNN + OSRM K64 + degree-quantile | 已完成，见 SD 表 |
| MTGNN + fullPair / GSP K64 + degree-quantile | 178 上有启动脚本 `/tmp/run_mtgnn_dynamic_178_g0.sh`，但未找到 log/checkpoint，不能算有效结果 |
| KnowAir / CCAQ MAGE-dim | 178 的 KNN-K32 fixed/adaptive/full-dynamic 已完成；180 的 PM25GNN graph 首次启动失败，后续 178 pm_g0/pm_g1 已补齐结果 |
| AGCRN / D2STGNN / STGODE on SD | 当前未找到 SD 正式 checkpoint；D2STGNN/AGCRN 只有其他数据集结果 |

## 结果解读约定

- `fixed`：只替换/指定静态图，不加动态阈值。
- `dynamic exp_tanh`：旧阈值函数，学习半径倍数。
- `dynamic degree-quantile`：新阈值函数，学习目标度/分位数。
- `addaptadj`：保留 Graph WaveNet 原 adaptive adjacency。
- `K64`：候选池限制到 64 个邻居；`fullPair` 表示全节点候选。
- `flowonly / largest_aligned` 是不同阶段的配置标签，不应和同一行外的 protocol 混作严格公平比较。

## SD 主线结果

| Model | Graph / module | Threshold | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---|---:|---:|---:|---:|---:|
| STID | original | / | / | 17.8422 | 31.1878 | 0.1203 | / |
| GWNet | original | / | yes | 18.7983 | 31.1842 | 0.1416 | 0.1217 |
| GWNet | original, sd-aligned fresh | / | yes | 18.9315 | 31.4135 | 0.1264 | 0.1126 |
| GWNet | original fixed | / | no | 20.4228 | 33.1292 | 0.1377 | 0.1244 |
| GWNet | physical + adaptive | / | yes | 18.1488 | 30.1777 | 0.1190 | 0.1057 |
| DCRNN | original | / | / | 19.0016 | 31.0799 | 0.1251 | / |
| STGCN | original | / | / | 21.2788 | 36.2263 | 0.1492 | 0.1346 |
| GWNet | OSRM beta=1 fixed | / | no | 19.8386 | 32.2385 | 0.1348 | 0.1203 |
| GWNet | OSRM beta=1 sparse | / | no | 20.2352 | 32.8860 | 0.1366 | 0.1221 |
| GWNet | OSRM beta=1 sparse adaptive | / | yes | 18.3930 | 30.9195 | 0.1259 | 0.1110 |
| DCRNN | OSRM beta=1 fixed | / | / | 18.6836 | 30.6549 | 0.1243 | 0.1105 |
| DCRNN | OSRM beta=1 sparse | / | / | 18.6593 | 30.4750 | 0.1237 | 0.1097 |
| GWNet | fullPair dynamic | exp_tanh | no | 18.7159 | 31.5546 | 0.1262 | 0.1125 |
| GWNet | OSRM K64 fixed | / | no | 20.1006 | 32.7183 | 0.1327 | 0.1201 |
| GWNet | OSRM K64 dynamic | exp_tanh | no | 19.4922 | 31.5299 | 0.1287 | 0.1166 |
| GWNet | OSRM K64 dynamic, largest_aligned | exp_tanh | no | 19.3561 | 31.7904 | 0.1316 | 0.1163 |
| GWNet | GSP K64 fixed | / | no | 20.5546 | 33.3987 | 0.1359 | 0.1218 |
| GWNet | GSP K64 dynamic | exp_tanh | no | 18.7582 | 30.6328 | 0.1294 | 0.1137 |
| GWNet | GSP K64 dynamic | exp_tanh | yes | 17.9169 | 30.5399 | 0.1235 | 0.1083 |
| GWNet | EFFRES K64 fixed | / | no | 20.7981 | 33.8019 | 0.1386 | 0.1244 |
| GWNet | EFFRES K64 dynamic | exp_tanh | no | 21.9839 | 35.6300 | 0.1389 | 0.1266 |
| GWNet | fullPair dynamic | exp_tanh | yes | 19.3685 | 32.4364 | 0.1247 | 0.1122 |
| DCRNN | fullPair dynamic | old untagged | / | 18.2713 | 30.2007 | 0.1232 | 0.1089 |
| DCRNN | OSRM K64 dynamic | exp_tanh | / | 18.2100 | 29.5414 | 0.1223 | 0.1084 |
| DCRNN | GSP K64 dynamic | exp_tanh | / | 18.0222 | 29.7664 | 0.1300 | 0.1138 |
| STGCN | fullPair dynamic | exp_tanh | / | 19.5426 | 34.1285 | 0.1373 | 0.1238 |
| STGCN | OSRM K64 fixed | / | / | 19.8511 | 34.3655 | 0.1347 | 0.1204 |
| STGCN | GSP K64 fixed | / | / | 19.7806 | 34.0461 | 0.1426 | 0.1274 |
| STGCN | GSP K64 dynamic | exp_tanh | / | 19.7075 | 34.1798 | 0.1403 | 0.1263 |
| MTGNN | OSRM K64 dynamic | degree-quantile | no original adaptive | 19.0145 | 31.8734 | 0.1306 | 0.1171 |

## 补充核对

| Item | Type | MAE | RMSE | MAPE | WAPE | 备注 |
|---|---|---:|---:|---:|---:|---|
| GWNet original (SD) | checkpoint test | 18.7983 | 31.1842 | 0.1416 | 0.1217 | `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/checkpoints/GraphWaveNet/SD_original_100_12_12/25ff6d2813909f2521e3c655e08f3f76/test_metrics.json` |
| GWNet GSP K64 dynamic + addaptadj (SD) | nodewise analysis summary | 17.5840 | / | / | / | 只有 analysis summary；相对 original adaptive 的 mean-node delta 为 `-0.9549`，617 个节点更好、79 个更差 |
| GWNet GSP K64 dynamic + addaptadj (METR-LA) | checkpoint test | 3.0759 | 6.2076 | 0.0850 | 0.0711 | `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/checkpoints/DynamicThresholdGraphWaveNet/METR-LA_dynamic_threshold_hard_exp_tanh_gwnet_100_12_12_gspK64_addaptadj_smallbench_metrla_osrm_20260527/96e42117beabc15d4fc0ee16a236935e/test_metrics.json` |

## SD OSRM beta 图扫描

| Model | beta | Graph mode | MAE | RMSE | MAPE | WAPE |
|---|---:|---|---:|---:|---:|---:|
| GWNet | 0.25 | fixed | 20.3697 | 33.0733 | 0.1344 | 0.1207 |
| GWNet | 0.50 | fixed | 20.0499 | 32.5359 | 0.1381 | 0.1228 |
| GWNet | 1.00 | fixed | 19.8386 | 32.2385 | 0.1348 | 0.1203 |
| GWNet | 1.50 | fixed | 20.1302 | 32.8318 | 0.1334 | 0.1195 |
| GWNet | 2.00 | fixed | 19.9546 | 32.4269 | 0.1306 | 0.1178 |
| GWNet | 0.25 | fixed + adaptive adjacency | 18.4062 | 30.7466 | 0.1282 | 0.1142 |
| GWNet | 0.50 | fixed + adaptive adjacency | 18.4850 | 30.7935 | 0.1310 | 0.1189 |
| GWNet | 1.00 | fixed + adaptive adjacency | 18.1753 | 30.6229 | 0.1330 | 0.1157 |
| GWNet | 1.50 | fixed + adaptive adjacency | 18.2978 | 30.8032 | 0.1298 | 0.1138 |
| GWNet | 2.00 | fixed + adaptive adjacency | 18.5650 | 31.2280 | 0.1250 | 0.1135 |
| DCRNN | 0.25 | fixed | 19.0217 | 31.0545 | 0.1260 | 0.1121 |
| DCRNN | 0.50 | fixed | 18.6996 | 30.6197 | 0.1238 | 0.1097 |
| DCRNN | 1.00 | fixed | 18.6836 | 30.6549 | 0.1243 | 0.1105 |
| DCRNN | 1.50 | fixed | 18.5721 | 30.6035 | 0.1239 | 0.1102 |
| DCRNN | 2.00 | fixed | 18.5658 | 30.3048 | 0.1245 | 0.1106 |

## METR-LA 小数据集结果

| Model | Graph / module | Threshold | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---|---:|---:|---:|---:|---:|
| GWNet | OSRM full fixed | / | yes | 3.0718 | 6.2409 | 0.0837 | 0.0704 |
| GWNet | OSRM K64 fixed | / | yes | 3.0807 | 6.2159 | 0.0840 | 0.0707 |
| GWNet | GSP K64 fixed | / | yes | 3.0851 | 6.2335 | 0.0847 | 0.0708 |
| GWNet | fullPair dynamic | exp_tanh | yes | 3.0730 | 6.2099 | 0.0830 | 0.0700 |
| GWNet | OSRM K64 dynamic | exp_tanh | yes | 3.0729 | 6.1719 | 0.0829 | 0.0699 |
| GWNet | GSP K64 dynamic | exp_tanh | yes | 3.0759 | 6.2076 | 0.0850 | 0.0711 |
| GWNet | fullPair dynamic, identity tau=0.5 | degree-quantile | yes | 3.0654 | 6.2058 | 0.0842 | 0.0705 |
| GWNet | fullPair dynamic, probe | degree-quantile | yes | 3.0540 | 6.1530 | 0.0846 | 0.0705 |
| DCRNN | OSRM full fixed | / | / | 3.0266 | 6.2229 | 0.0822 | 0.0689 |
| DCRNN | OSRM K64 fixed | / | / | 3.1172 | 6.3181 | 0.0839 | 0.0707 |
| DCRNN | GSP K64 fixed | / | / | 3.0940 | 6.3059 | 0.0855 | 0.0712 |
| DCRNN | fullPair dynamic | exp_tanh | / | 3.0943 | 6.3623 | 0.0850 | 0.0712 |
| DCRNN | OSRM K64 dynamic | exp_tanh | / | 3.0919 | 6.3548 | 0.0851 | 0.0712 |
| DCRNN | GSP K64 dynamic | exp_tanh | / | 3.0732 | 6.3295 | 0.0845 | 0.0706 |
| STGCN | OSRM full fixed | / | / | 3.1993 | 6.4571 | 0.0881 | 0.0737 |
| STGCN | OSRM K64 fixed | / | / | 3.2050 | 6.4812 | 0.0880 | 0.0739 |
| STGCN | GSP K64 fixed | / | / | 3.1791 | 6.3908 | 0.0868 | 0.0731 |
| STGCN | fullPair dynamic | exp_tanh | / | 3.2701 | 6.6716 | 0.0902 | 0.0758 |
| STGCN | OSRM K64 dynamic | exp_tanh | / | 3.2350 | 6.5203 | 0.0882 | 0.0749 |
| STGCN | GSP K64 dynamic | exp_tanh | / | 3.2221 | 6.5336 | 0.0894 | 0.0751 |

## PEMS-BAY / LargeST 结果

| Dataset | Model | Graph / module | Threshold | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---:|---:|---:|---:|
| PEMS-BAY | GWNet | fullPair dynamic, identity tau=0.5 | degree-quantile | 1.5880 | 3.6522 | 0.0355 | 0.0330 |
| PEMS-BAY | GWNet | fullPair dynamic, probe | degree-quantile | 1.5916 | 3.6756 | 0.0351 | 0.0328 |
| GBA | STID | original | / | 20.1298 | 34.4777 | 0.1598 | / |
| GBA | GWNet | OSRM K64 dynamic no-adaptive | degree-quantile | 21.9341 | 35.1577 | 0.2058 | 0.1626 |
| GLA | STID | original | / | 19.7113 | 34.1685 | 0.1220 | / |
| CA | STID | original | / | 18.3057 | 31.7102 | 0.1379 | / |

## FlowNet / PEMS04F 参考结果

| Dataset | Model | Setting | MAE | RMSE | MAPE | WAPE |
|---|---|---|---:|---:|---:|---:|
| SD | FlowNet | OSRM, BasicTS migrated | 21.1180 | 54.9349 | 0.2386 | 0.2020 |
| PEMS04F | FlowNet | official scaler style | 18.6029 | 30.5343 | / | / |
| PEMS04F | FlowNet | BasicTS train scaler | 18.6488 | 30.6617 | / | / |

## 空气质量结果状态

| Dataset / run | 状态 | 说明 |
|---|---|---|
| KnowAir / CCAQ 旧三通道版本 | invalid | 只用了 `[target, tod, dow]`，tracker 已标为不用于 claim |
| KnowAir_FULLCOV / CCAQ_FULLCOV | stopped/replaced | 不是 MAGE-aligned，不用于正式比较 |
| KnowAir_MAGE13 / CCAQ_MAGE10 | completed | 178 的 MAGE-dim KNN-K32 fixed/adaptive/full-dynamic 已落盘并有 test |
| KnowAir_MAGE13_PM25GNN_GRAPH all-feature z-score | mixed | 180 的 PM_G1 启动失败缺 `desc.json`，但 178 的 PM_G0/PM_G1 已有 PM25GNN graph fixed/adaptive/dynamic test |

## 空气质量补充结果

| Dataset | Model | Graph / module | Threshold | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---:|---:|---:|---:|---:|
| KnowAir_MAGE13 | GWNet | KNN-K32 fixed | / | no | 15.4282 | 24.3760 | 0.5498 | 0.4087 |
| KnowAir_MAGE13 | GWNet | KNN-K32 adaptive | / | yes | 15.3720 | 23.6234 | 0.5809 | 0.4187 |
| KnowAir_MAGE13 | GWNet | full dynamic | degree-quantile | no | 15.4408 | 24.1921 | 0.5612 | 0.4166 |
| CCAQ_MAGE10 | GWNet | KNN-K32 fixed | / | no | 18.4675 | 31.9226 | 0.2821 | 0.2590 |
| CCAQ_MAGE10 | GWNet | KNN-K32 adaptive | / | yes | 18.4954 | 31.7794 | 0.2877 | 0.2629 |
| CCAQ_MAGE10 | GWNet | full dynamic | degree-quantile | no | 18.8322 | 32.0493 | 0.3054 | 0.2736 |
| KnowAir_MAGE13 | GWNet | PM25GNN graph fixed | / | no | 15.5086 | 24.5147 | 0.5458 | 0.4093 |
| KnowAir_MAGE13 | GWNet | PM25GNN graph adaptive | / | yes | 15.4156 | 23.7896 | 0.5780 | 0.4179 |
| KnowAir_MAGE13 | GWNet | PM25GNN graph dynamic | degree-quantile | no | 15.6407 | 24.1581 | 0.5945 | 0.4297 |
| KnowAir_MAGE13 | GWNet | altitude candidate dynamic | degree-quantile | no | 15.4596 | 24.0690 | 0.5635 | 0.4175 |

## 当前可支持的初步结论

1. SD 上，旧 `exp_tanh` 动态阈值对 DCRNN 最稳定，GSP K64 dynamic 的 MAE 为 18.0222，明显强于 DCRNN original 19.0016。
2. SD 上，GWNet 的 GSP K64 dynamic 比 GSP K64 fixed 明显好；带 `addaptadj` 的 GSP K64 dynamic 最好，MAE 17.9169，已经超过 `GWNet physical + adaptive` 和 STID。
3. METR-LA 上，GWNet 的新 `degree-quantile probe full` 是当前同组最优 MAE 3.0540；DCRNN/STGCN 上旧 `exp_tanh` 动态阈值没有形成同样稳定增益。
4. GBA 上当前只有 `GWNet + OSRM K64 + degree-quantile no-adaptive`，效果弱于 STID，暂时不能支撑 large-scale claim。
5. MTGNN 动态阈值 MVP 目前只有 OSRM K64 有效结果，MAE 19.0145；fullPair/GSP K64 需要重新跑或查启动原因。
6. 空气质量线已经有 MAGE-dim 正式结果，KnowAir_MAGE13 的 KNN-K32 fixed/adaptive/full-dynamic 和 PM25GNN graph 版本都已落盘；180 上的 `PM25GNN_GRAPH` 启动失败只代表那一次尝试，不代表整条线没有结果。

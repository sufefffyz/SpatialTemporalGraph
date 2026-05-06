# Delay-Selective Traffic Forecasting: Research Gap and Experiment Plan

## 核心结论

当前方向不能写成“提出一个 delay-aware traffic forecasting model”。这个表述会和 `PDFormer`、`STDDE`、`MillGNN` 高度撞车。

更稳的定位是：

```text
Delay-selective / observability-aware traffic forecasting:
不是所有边都应该学习 delay。
只有当 delay 相对采样间隔、空间距离、方向和交通状态可辨识时，
asynchronous message passing 才有意义。
近邻、短距离、低置信边应保持同步或弱延迟；
跨 group / region / corridor 的传播才学习非零 delay。
```

最关键的 research question 应从：

```text
How to model delay?
```

改成：

```text
When is delay worth modeling,
at what spatial scale,
and how can a model avoid learning spurious delays on near-neighbor edges?
```

## 与已有工作的碰撞风险

| 工作 | 已经覆盖什么 | 对本 idea 的风险 | 可避开的角度 |
|---|---|---|---|
| PDFormer | propagation delay-aware traffic forecasting；dynamic long-range dependency；delay-aware feature transformation / attention | 如果只说 delay-aware STGNN，会高度撞车 | 不主张“delay module 新”，而强调 delay observability、near-zero delay suppression、scale selection |
| STDDE | 显式 edge-level delay；spatial-temporal delay differential equation；可学习 delay estimator | edge-specific delay 基本被覆盖，碰撞最高 | 不做 all-edge delay，而做 delay-selective / delay-sparse / zero-delay-gated modeling |
| MillGNN | multi-scale grouping lead-lag dependencies；generic MTS forecasting；包含 Traffic / PeMS | group-level lead-lag 本身被覆盖 | 强调 traffic road topology、direction、distance regime、physical delay validation，而不是泛化 grouping |

## 最稳 Novelty Framing

### Framing A: Delay-Selective / Observability-Aware Traffic Forecasting

核心表述：

```text
Existing delay-aware traffic forecasting methods often assume delay is a generally useful edge-wise modeling object.
However, under fixed sampling intervals and short sensor spacing, many neighboring edges have unobservable or near-zero delay.
We study when delay is observable and useful, and only introduce asynchronous message passing on high-confidence delayed relations.
```

需要证据：

```text
1. 大量近邻边的 estimated lag 接近 0 或不稳定
2. near-neighbor edge-wise delay 在跨天 / bootstrap / event split 下不稳定
3. all-edge delay baseline 不如 delay-selective baseline
4. 模型收益集中在 long-delay / congested / incident / long-range propagation 子集
```

### Framing B: Distance-Regime-Aware / Scale-Aware Delay Modeling

核心表述：

```text
Traffic delay is scale-dependent rather than uniformly edge-wise.
Local neighboring sensors are often synchronous at 5-min aggregation,
while cross-corridor / cross-region / bottleneck propagation shows identifiable lead-lag.
```

候选命名：

```text
delay-sparse hierarchical STGNN
scale-aware asynchronous traffic propagation
distance-regime-aware delay effect
intra-group synchronous, inter-group asynchronous message passing
```

需要证据：

```text
1. best lag vs distance 曲线
2. 按 distance bin 的 delay 显著性
3. free-flow / congestion / incident regime 下 delay 差异
4. intra-group delay 更接近 0，inter-group delay 更稳定
5. group-level delay 比 node-pair all-edge delay 更鲁棒
```

### Framing C: Delay Validation Before Delay Modeling

核心表述：

```text
Before adding delay modules to STGNNs, we need a protocol for verifying whether propagation delay is real, learnable, and useful.
```

这可以作为辅助贡献，但最好不要作为唯一贡献，否则容易被认为只是 empirical analysis。

## 推荐主线

### 主 claim

```text
Delay-aware forecasting should be selective and scale-aware:
explicit delay helps only on identifiable long-range / inter-group / congested propagation,
while forcing delay on near-neighbor edges introduces noise.
```

### 不建议声称

```text
不要声称：
1. 第一次做 delay-aware traffic forecasting
2. 第一次学习 lead-lag dependencies
3. 第一次做 group-level lead-lag
4. 第一次把 delay 引入 STGNN
```

### 可以声称

```text
可以争取声称：
1. 第一次系统研究交通预测中 delay 的可观测尺度和适用边界
2. 第一次把 near-zero-delay suppression 引入 delay-aware STGNN
3. 第一次比较 synchronous、all-edge delay、group-level delay、delay-selective delay 的作用范围
4. 第一次用物理 delay audit + STGNN benchmark 联合验证 delay modeling 的必要性
```

## 实验路线

### Stage 1: Delay Phenomenon and Observability Audit

首选数据：

```text
FT-AED
I-24 MOTION
```

目标不是训练模型，而是证明：

```text
1. delay 在事件 / 拥堵 / long-range propagation 中真实存在
2. 近邻边 delay 常接近 0 或不可稳定估计
3. delay 的可辨识性受 distance / direction / traffic regime / sampling interval 影响
```

必做图：

```text
1. time-space speed heatmap
2. event onset time vs distance
3. lagged correlation curve
4. best lag distribution
5. best lag vs distance / congestion regime
6. near-zero delay ratio by distance bin
```

统计检验：

```text
bootstrap stability
day-wise stability
permutation / shuffled-lag sanity check
incident vs non-incident stratification
```

### Stage 2: Main STGNN Benchmark

首选数据：

```text
LargeST
PeMS04 / PeMS07 / PeMS08
```

必须正面对比：

```text
PDFormer
STDDE
MillGNN
synchronous STGNN backbone
all-edge delay variant
```

核心模型变体：

```text
1. synchronous STGNN
2. all-edge delay STGNN
3. group-level delay without zero-delay gate
4. delay-selective hierarchical STGNN
5. random group
6. distance-only group
7. direction-agnostic delay
8. delay-shuffled control
```

评价方式：

```text
overall MAE/RMSE/MAPE
long-horizon performance
high-delay edge/node subset
long-distance group-edge subset
congested / incident window subset
learned delay vs measured delay consistency
delay stability across days
runtime and parameter count
```

### Stage 3: Road-Link / Segment-Level Generalization

如果 claim 只到 detector graph：

```text
FT-AED + LargeST/PeMS + MeTS-10 or Chicago Traffic Tracker
```

如果 claim 要到 road-link graph：

```text
必须至少补一个：
MeTS-10
Traffic4cast 2022
NPMRDS
WebTRIS
NDW
```

推荐顺序：

```text
1. MeTS-10：OSM road graph + multi-city segment speed，处理成本相对可控
2. Traffic4cast 2022：road graph edge / super-segment 强，但任务复杂
3. NPMRDS：TMC road segment 长期数据最强，但权限和工程成本最高
4. WebTRIS / NDW：欧洲 road network 泛化增强
5. Chicago / NYC：公开 road-segment quick demo，只适合 supplement
```

## 数据集优先级

| 优先级 | 数据集 | 角色 | 说明 |
|---:|---|---|---|
| 1 | FT-AED | 快速物理验证 | 30s、lane-level、I-24、event/anomaly label，适合 onset delay |
| 2 | LargeST / PeMS | 主 STGNN benchmark | 与 PDFormer / STDDE / MillGNN 可比，LargeST 时间跨度更长 |
| 3 | MeTS-10 | road-link 泛化首选 | OSM road graph、10 cities、15min segment speed |
| 4 | Traffic4cast 2022 | road graph edge 强补强 | dynamic road graph / edge-level prediction |
| 5 | Chicago Traffic Tracker | public segment supplement | 10min segment speed，公开，但来源是公交 GPS |
| 6 | NPMRDS | 最强 TMC 长期 road-link 数据 | 长期、覆盖 National Highway System，但权限成本高 |
| 7 | WebTRIS / NDW | 欧洲 road network 泛化 | API/导出权限和服务稳定性需核实 |
| 8 | Madrid / MTD | 城市 sensor graph 补强 | 多测点、长时段，但多为 intensity / occupancy |
| 9 | I-24 MOTION | 高分辨率可视化 | time-space wavefront 很强，但轨迹转 virtual detector 有工程选择 |
| 10 | pNEUMA / exiD | microscope case study | 局部、短时段，适合现象图，不适合主实验 |

## 立即核实 Checklist

1. `PDFormer / STDDE / MillGNN` 官方代码和 PeMS 结果是否可复现。
2. `STDDE` 的 delay estimator 是否支持或隐含 all-edge delay，是否可以作为直接 baseline。
3. `MillGNN` 的 grouping 是否与 road topology 无关，从而给 traffic-specific grouping 留空间。
4. `FT-AED` 是否能直接构造 `49 milemarkers × 4 lanes` 的 directed lane-level graph。
5. `LargeST / PeMS` 的 5min 粒度下 delay 是否仍可观测；如果不可观测，必须把 zero-delay / delay-selective 写成主卖点。
6. group 构造不能只用 DTW similarity，必须比较 road-topology group、distance-regime group、corridor/community group、learned group、random group。
7. MeTS-10 是否能获取完整 OSM graph + segment speed。
8. Traffic4cast 2022 是否能转成普通 STGNN setting。
9. NPMRDS / WebTRIS / NDW 是否有权限和可批量导出路径。
10. 最终 claim 边界要和数据匹配：没有 road-link 数据就不要写 road-link graph level。

## 给导师汇报的一段话

单纯做 delay-aware traffic forecasting 会和已有工作高度重叠：PDFormer 已经以 propagation delay-aware 为核心，STDDE 已经显式学习 traffic graph 上的 edge-specific delay，MillGNN 已经做 multi-scale group-level lead-lag。因此我准备把 novelty 从“又一个 delay 模块”改成“delay 何时值得建模、在哪个空间尺度上建模、如何避免近邻零延迟带来的噪声”。具体做法是，先用 FT-AED / I-24 验证真实交通扰动的 propagation delay，并证明很多近邻边在采样粒度下 delay 接近零或不可稳定估计；再在 LargeST / PeMS 上比较 synchronous、all-edge delay、group-level delay 和 delay-selective hierarchical message passing；最后用 MeTS-10 / Traffic4cast / NPMRDS / WebTRIS 等 road-link 或 segment-level 数据补强，避免论文只停留在 freeway detector graph。这样路线既承认 I-24/exiD/pNEUMA 时间短或拓扑简单的局限，也保留它们在物理验证上的价值，并把 road-link 数据作为 claim 升级的必要补强。

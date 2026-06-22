# 自适应阈值图模块变体讨论版

更新时间：2026-06-08  
用途：讨论当前已经尝试过的图构造 / 动态阈值 / 候选池 / backbone 变体，明确哪些结果能支持方法，哪些只能作为分析参考。  
结果来源：`RESULTS_SNAPSHOT_20260605.md` 与当前落盘的 best-val checkpoint test metrics。除特别说明外，下表使用 BasicTS 训练协议下的 overall test MAE/RMSE/MAPE/WAPE。

## 一句话结论

当前最有希望的方向不是“完全替代 STGNN 内部 adaptive graph”，而是：

> 用物理距离或信号平滑性先验构造候选池，再在候选池内学习节点级动态邻居预算。

最强证据来自 `SD` 和 `METR-LA`：

- `SD`: `GWNet + GSP K64 dynamic + addaptadj` 得到 MAE `17.9169`，强于 `GWNet original` 和 `GWNet physical + adaptive`，但还没有超过 `STID 17.8422`。
- `SD`: `DCRNN + GSP K64 dynamic` 得到 MAE `18.0222`，强于 `DCRNN original 19.0016`。
- `METR-LA`: `GWNet + fullPair degree-quantile probe` 得到 MAE `3.0540`，优于多个 fixed graph 和旧 `exp_tanh` 动态头。
- `KnowAir / CCAQ`: 当前不支持动态阈值优势，经典 KNN / adaptive adjacency 更稳。

## 已尝试的模块变体

| 模块类别 | 具体变体 | 核心设计 | 当前判断 |
|---|---|---|---|
| 固定物理图 | OSRM Gaussian global threshold, beta sweep | 对 OSRM 距离做高斯核，再按目标平均度选全局阈值 | 可作为固定图 Pareto / 度分布分析基线 |
| 固定候选池 | OSRM K64 | 每个节点按 OSRM 距离取最近 64 个候选邻居 | 稳定、解释清楚，但单独 fixed 效果一般 |
| 固定候选池 | GSP K64 | 按信号平滑性/相似性候选取 top-64 | 在 SD 上配合动态阈值最有价值 |
| 固定候选池 | Effective resistance K64 | 按有效电阻选候选邻居 | 当前 SD 上效果较差，不建议作为主线 |
| 动态阈值旧头 | `exp_tanh` | 学习半径倍数 `r_i(t)=r_i^0 exp(alpha tanh(s_i(t)))` | SD 上有效，尤其 GSP K64 + DCRNN/GWNet |
| 动态阈值新头 | `degree-quantile` | 学习目标度数，再映射到距离分位数 | METR-LA / PEMS-BAY 上更稳，更符合“控制稀疏度”的故事 |
| 动态头激活 | `probe` | `state_act=tanh`, `quantile_temperature=1.0` | METR-LA 最好，学习更温和 |
| 动态头激活 | `identity tau=0.5` | 不压缩状态，sigmoid 更陡 | PEMS-BAY 略好，但差距很小 |
| 图权重 | binary mask | 0-1 消息传递 | 当前主线，最容易解释模块贡献 |
| 图权重 | learned edge weight | 在动态边上继续学权重 | 曾启动但暂停，因与动态阈值贡献耦合，暂不作为主结论 |
| 与 GWNet adaptive 组合 | `addaptadj=False/True` | 是否保留 GWNet 原 learned adaptive adjacency | SD 上 `GSP K64 dynamic + addaptadj` 最强；但这意味着动态阈值更像增强而非替代 |
| 稀疏算子 | SparseGraphWaveNet / SparseDCRNN | 固定图 CSR/sparse support | 证明固定稀疏图可加速，但动态 hard mask 仍主要是 dense 路径 |
| Backbone 扩展 | DCRNN / STGCN / MTGNN | 插入同一 DynamicThresholdSupport | DCRNN 在 SD 有明显增益；STGCN/METR-LA 不稳；MTGNN 只有 OSRM K64 有有效结果 |
| 空气质量图 | KNN-K32 / PM25GNN graph / altitude candidate | 用坐标或空气质量先验构图 | 当前 KnowAir/CCAQ 不支持动态阈值优势 |

## 阈值函数对比

### 旧版 `exp_tanh`

每个节点学习一个动态半径倍数：

```math
r_i(t)=r_i^0 \exp(\alpha \tanh(s_i(t))).
```

优点是实现简单，能从初始物理半径附近做动态伸缩。缺点是可调范围强依赖初始半径，不同初始化会带来不同的可学习空间。

### 新版 `degree-quantile`

每个节点学习目标邻居数：

```math
d_i(t)=d_{\min}+(d_{\max}-d_{\min})\sigma(s_i(t)/\tau),
```

再把目标度数映射成该节点候选距离序列中的分位阈值：

```math
r_i(t)=D_i^{(\lfloor d_i(t) \rfloor)}.
```

优点是直接控制节点级稀疏度，适合写成“学习每个节点当前需要多少邻居”。这是目前更适合作为论文主线的阈值函数。

## SD 关键结果

| Model | Graph / module | Threshold | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---|---:|---:|---:|---:|---:|
| STID | original | / | / | 17.8422 | 31.1878 | 0.1203 | / |
| GWNet | original | / | yes | 18.7983 | 31.1842 | 0.1416 | 0.1217 |
| GWNet | physical + adaptive | / | yes | 18.1488 | 30.1777 | 0.1190 | 0.1057 |
| GWNet | original fixed | / | no | 20.4228 | 33.1292 | 0.1377 | 0.1244 |
| GWNet | GSP K64 fixed | / | no | 20.5546 | 33.3987 | 0.1359 | 0.1218 |
| GWNet | GSP K64 dynamic | exp_tanh | no | 18.7582 | 30.6328 | 0.1294 | 0.1137 |
| GWNet | GSP K64 dynamic | exp_tanh | yes | 17.9169 | 30.5399 | 0.1235 | 0.1083 |
| DCRNN | original | / | / | 19.0016 | 31.0799 | 0.1251 | / |
| DCRNN | GSP K64 dynamic | exp_tanh | / | 18.0222 | 29.7664 | 0.1300 | 0.1138 |
| STGCN | original | / | / | 21.2788 | 36.2263 | 0.1492 | 0.1346 |
| STGCN | GSP K64 dynamic | exp_tanh | / | 19.7075 | 34.1798 | 0.1403 | 0.1263 |
| MTGNN | OSRM K64 dynamic | degree-quantile | no original adaptive | 19.0145 | 31.8734 | 0.1306 | 0.1171 |

SD 上可以讲的点：

- `GSP K64 dynamic` 相对 `GSP K64 fixed` 的提升很明显：GWNet MAE `20.5546 -> 18.7582`。
- 加上 GWNet 原生 `addaptadj` 后，`GSP K64 dynamic` 进一步到 `17.9169`，强于 `GWNet original` 和 `GWNet physical + adaptive`。
- DCRNN 也从 `19.0016` 提升到 `18.0222`，说明这个模块不是只对 GWNet 有效。
- 但 SD 最强参考 `STID 17.8422` 仍略好于当前最佳动态图结果，因此不能说已经赢过强大规模 ST baseline。

## METR-LA 关键结果

| Model | Graph / module | Threshold | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---|---:|---:|---:|---:|---:|
| GWNet | OSRM full fixed | / | yes | 3.0718 | 6.2409 | 0.0837 | 0.0704 |
| GWNet | OSRM K64 fixed | / | yes | 3.0807 | 6.2159 | 0.0840 | 0.0707 |
| GWNet | GSP K64 fixed | / | yes | 3.0851 | 6.2335 | 0.0847 | 0.0708 |
| GWNet | fullPair dynamic | exp_tanh | yes | 3.0730 | 6.2099 | 0.0830 | 0.0700 |
| GWNet | OSRM K64 dynamic | exp_tanh | yes | 3.0729 | 6.1719 | 0.0829 | 0.0699 |
| GWNet | fullPair dynamic identity tau=0.5 | degree-quantile | yes | 3.0654 | 6.2058 | 0.0842 | 0.0705 |
| GWNet | fullPair dynamic probe | degree-quantile | yes | 3.0540 | 6.1530 | 0.0846 | 0.0705 |
| DCRNN | OSRM full fixed | / | / | 3.0266 | 6.2229 | 0.0822 | 0.0689 |
| DCRNN | GSP K64 dynamic | exp_tanh | / | 3.0732 | 6.3295 | 0.0845 | 0.0706 |
| STGCN | GSP K64 fixed | / | / | 3.1791 | 6.3908 | 0.0868 | 0.0731 |
| STGCN | GSP K64 dynamic | exp_tanh | / | 3.2221 | 6.5336 | 0.0894 | 0.0751 |

METR-LA 上可以讲的点：

- GWNet 上 `degree-quantile probe` 是当前最优，MAE `3.0540`。
- 它相对 `OSRM full fixed` 提升 `0.0178` MAE，相对 `GSP K64 fixed` 提升 `0.0311` MAE。
- 这个结果支持“学习节点级邻居预算”比固定图更灵活，但增益幅度不大。
- DCRNN/STGCN 上没有一致优势，所以 METR-LA 只能支持 backbone-dependent claim。

## PEMS-BAY 关键结果

| Dataset | Model | Graph / module | Threshold | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---:|---:|---:|---:|
| PEMS-BAY | GWNet | fullPair dynamic identity tau=0.5 | degree-quantile | 1.5880 | 3.6522 | 0.0355 | 0.0330 |
| PEMS-BAY | GWNet | fullPair dynamic probe | degree-quantile | 1.5916 | 3.6756 | 0.0351 | 0.0328 |

PEMS-BAY 上可以讲的点：

- 两个 `degree-quantile` 头都能跑通，`identity tau=0.5` 的 MAE 略好。
- 目前缺少同协议 fixed graph / original GWNet 对照，因此只能作为可迁移性参考，不能支撑强 claim。

## 空气质量关键结果

MAGE-aligned 设置：`KnowAir_MAGE13` 使用 `in_dim=13, horizon=24`，`CCAQ_MAGE10` 使用 `in_dim=10, horizon=24`。旧三通道和 fullcov 探索版不用于 claim。

| Dataset | Model | Graph / module | Threshold | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---:|---:|---:|---:|---:|
| KnowAir_MAGE13 | GWNet | KNN-K32 fixed | / | no | 15.4282 | 24.3760 | 0.5498 | 0.4087 |
| KnowAir_MAGE13 | GWNet | KNN-K32 adaptive | / | yes | 15.3720 | 23.6234 | 0.5809 | 0.4187 |
| KnowAir_MAGE13 | GWNet | full dynamic | degree-quantile | no | 15.4408 | 24.1921 | 0.5612 | 0.4166 |
| KnowAir_MAGE13 | GWNet | PM25GNN graph fixed | / | no | 15.5086 | 24.5147 | 0.5458 | 0.4093 |
| KnowAir_MAGE13 | GWNet | PM25GNN graph adaptive | / | yes | 15.4156 | 23.7896 | 0.5780 | 0.4179 |
| KnowAir_MAGE13 | GWNet | PM25GNN graph dynamic | degree-quantile | no | 15.6407 | 24.1581 | 0.5945 | 0.4297 |
| CCAQ_MAGE10 | GWNet | KNN-K32 fixed | / | no | 18.4675 | 31.9226 | 0.2821 | 0.2590 |
| CCAQ_MAGE10 | GWNet | KNN-K32 adaptive | / | yes | 18.4954 | 31.7794 | 0.2877 | 0.2629 |
| CCAQ_MAGE10 | GWNet | full dynamic | degree-quantile | no | 18.8322 | 32.0493 | 0.3054 | 0.2736 |

空气质量上可以讲的点：

- KnowAir 上 `KNN-K32 adaptive` 最好，动态阈值没有赢。
- CCAQ 上 `KNN-K32 fixed` 最好，动态阈值明显更差。
- 这说明动态阈值不是所有领域自然有效；空气质量可能需要把气象、风场、地形阻挡等先验更明确地放进候选池或边权，而不是只靠距离分位数。

## LargeST 大规模结果

| Dataset | Model | Graph / module | Threshold | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---:|---:|---:|---:|
| GBA | STID | original | / | 20.1298 | 34.4777 | 0.1598 | / |
| GBA | GWNet | OSRM K64 dynamic no-adaptive | degree-quantile | 21.9341 | 35.1577 | 0.2058 | 0.1626 |
| GLA | STID | original | / | 19.7113 | 34.1685 | 0.1220 | / |
| CA | STID | original | / | 18.3057 | 31.7102 | 0.1379 | / |

大规模上可以讲的点：

- 目前只说明 `GWNet + OSRM K64 dynamic no-adaptive` 还不足以打大规模强 baseline。
- 若要走 large-scale claim，需要更强 backbone 或更强候选池，不宜只靠当前 GWNet 动态阈值头。

## 稀疏化与效率

固定图稀疏算子已经有初步证据，但动态阈值训练还没有真正完成稀疏化：

| Variant | MAE | RMSE | 观察 |
|---|---:|---:|---|
| GWNet OSRM beta=1 fixed | 19.8386 | 32.2385 | dense/fixed reference |
| SparseGraphWaveNet OSRM beta=1 | 20.2352 | 32.8860 | 速度可提升，但精度有约 `+0.40` MAE 退化 |
| DCRNN OSRM beta=1 fixed | 18.6836 | 30.6549 | dense/fixed reference |
| SparseDCRNN OSRM beta=1 | 18.6593 | 30.4750 | 与 dense 基本一致 |

当前要谨慎讲：

- 固定稀疏图可以用 CSR/sparse support 加速。
- 动态 hard mask 当前主要还是 dense `B x N x N` 路径。
- 因此论文里不能直接 claim 动态阈值已经带来真实训练加速，除非后续实现 candidate-pool 内的 packed edge / sparse aggregation。

## 当前推荐主线

### 可以作为主线的 claim

1. 固定物理图的全局阈值很难同时适配不同节点，节点级动态邻居预算能改善部分 STGNN。
2. 候选池很重要，`GSP K64` 在 SD 上比纯 OSRM 候选更适合动态阈值。
3. `degree-quantile` 比半径倍数更适合作为最终模块，因为它直接控制节点度数和稀疏预算。
4. 动态阈值应被定位为 graph preprocessing / graph rewiring module，而不是万能 plug-and-play 模块。

### 暂时不建议讲的 claim

1. 不建议说动态阈值已经全面超过强 STGNN baseline；SD 上还略弱于 `STID`。
2. 不建议说模块对所有 backbone 都有效；METR-LA 上 STGCN/DCRNN 证据不一致。
3. 不建议说动态阈值天然适合空气质量；KnowAir/CCAQ 当前结果不支持。
4. 不建议说当前动态阈值已经实现训练加速；实现路径仍以 dense mask 为主。

## 下一步讨论重点

1. 是否把论文主线从“动态阈值 plug-in”收窄为“候选池约束的节点级邻居预算学习”。
2. 是否把最终阈值函数定为 `degree-quantile`，保留 `exp_tanh` 作为旧版 ablation。
3. 是否优先在 `GWNet + addaptadj` 和 `DCRNN` 上补 `GSP K64 / OSRM K64 / fullPair` 的同协议 ablation。
4. 是否先实现 candidate-pool 内 sparse aggregation，再继续做大规模 LargeST。
5. 空气质量是否转为反例分析或单独设计风场/地形候选池，不放进主表。

## 相关文件

- `RESULTS_SNAPSHOT_20260605.md`: 当前总结果快照。
- `EXPERIMENT_TRACKER.md`: 实验队列与有效/无效设置说明。
- `README.md`: 动态阈值机制、复杂度矛盾和早期实验记录。
- `BasicTS/baselines/AdaptiveGraph/dynamic_support.py`: 动态阈值核心实现。

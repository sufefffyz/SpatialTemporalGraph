# 自适应阈值动态图 MVP

本文件记录当前 `adaptive_threshold_dynamic_weight` 这条实验线的核心问题、已有结论和后续应优先验证的方向。

## 当前目标

我们希望在 STGNN 中替代固定 sensor graph，学习一个节点级动态邻居集合：

```math
A_{b,i,j} = \mathbf{1}[d_{i,j} \le r_{b,i}]
```

其中 `d_ij` 是 OSRM 路网距离，`r_{b,i}` 是由节点历史状态和节点嵌入动态计算的半径。目标不是单纯提升精度，而是同时满足：

- 节点级邻居感知：不同节点、不同样本可以选择不同邻居范围。
- 稀疏图聚合：消息传递只在少量候选边上进行。
- 可学习阈值：半径头能通过预测损失收到梯度。
- Backbone 可迁移：至少在 GWNet / DCRNN / STGCN 中能解释清楚。

## 当前动态阈值模块

当前实现位于：

- `BasicTS/baselines/AdaptiveGraph/dynamic_support.py`
- `BasicTS/baselines/AdaptiveGraph/dynamic_gwnet.py`
- `BasicTS/baselines/AdaptiveGraph/dynamic_dcrnn.py`
- `BasicTS/baselines/AdaptiveGraph/dynamic_stgcn.py`

半径头形式为：

```math
s_{b,i} = \tanh(W_h x_{b,:,i} + b_h)
```

```math
q_{b,i} = w^\top [s_{b,i}; e_i] + b
```

```math
r_{b,i} = r_0 \exp(\alpha \tanh(q_{b,i}))
```

当前默认：

- `d_model = 32`
- `alpha = 1.0`
- `weight_mode = binary`
- `radius_param = exp_tanh`
- `r0` 由 LargeST-SD 原始图的平均出度反推，约等于 beta=1 的 OSRM 距离阈值。
- 对 SD 来说，参考平均出度为 `24.1885`，约对应 `r0 = 7.2 km`。
- 半径倍数范围为 `[exp(-1), exp(1)] ~= [0.368, 2.718]`。

训练时使用 straight-through estimator：

```math
A^{ST}_{b,i,j}
=
A^{hard}_{b,i,j}
+ \sigma((r_{b,i}-d_{i,j})/\tau)
- \operatorname{stopgrad}(\sigma((r_{b,i}-d_{i,j})/\tau))
```

forward 看起来是 hard 0-1 图，backward 通过 sigmoid surrogate 给 radius head 传梯度。

## 关键矛盾

当前最大问题是：

> 全节点候选和稀疏化加速天然有张力。

如果每个节点可以从所有其他节点中动态选择邻居，那么候选集合是：

```math
\mathcal{E}_{all} = \{(i,j): i,j \in \mathcal V\}
```

候选规模为：

```math
|\mathcal{E}_{all}| = N^2
```

这给了模型完整探索空间，也让 inactive edge 能通过 sigmoid surrogate 收到“靠近/远离阈值”的梯度。但计算上仍然需要构造和处理：

```math
B \times N \times N
```

的 dense mask。

反过来，如果只在 active edges 上做 sparse message passing：

```math
\mathcal{E}_{b}^{active}
=
\{(i,j): d_{i,j} \le r_{b,i}\}
```

则计算可以接近：

```math
O(B|\mathcal{E}_{b}^{active}|C)
```

但 hard `nonzero` 选边几乎不可导：

```math
\frac{\partial \mathbf{1}[d_{i,j} \le r_{b,i}]}{\partial r_{b,i}} = 0
```

这会导致 radius head 很难学习，尤其是 inactive edges 不再参与 surrogate 梯度。

## 当前实现的真实复杂度

当前图在数学上是 hard sparse，但实现路径仍然是 dense。

在 `dynamic_support.py` 中，`logits / soft_mask / hard_mask / mask` 都是：

```math
B \times N \times N
```

在 dynamic DCRNN 中，图传播使用：

```python
torch.einsum("bij,bjf->bif", support, x)
```

因此复杂度仍接近：

```math
O(BN^2C)
```

而不是真正 sparse 图的：

```math
O(BEC)
```

也就是说，当前方法应称为：

> hard-valued dense mask，而不是真正 sparse computation。

## DCRNN 为什么特别慢

DCRNN 的慢不是单次图卷积慢，而是 recurrent seq2seq 结构导致图传播重复次数太多。

当前 SD 设置：

- `T_in = 12`
- `H = 12`
- `num_rnn_layers = 2`
- `max_diffusion_step = 2`
- support 数量为 2，forward/backward transition。

每个 DCGRUCell 内部有两次图卷积：

- reset/update gate 一次
- candidate state 一次

每次图卷积有：

```math
S K = 2 \times 2 = 4
```

次 support multiplication。

每个 batch 的 cell 调用数为：

```math
(T_{in}+H) \times L_r = (12+12) \times 2 = 48
```

所以每个 batch 大约有：

```math
48 \times 2 \times 2 \times 2 = 384
```

次 support multiplication。

对比：

- GWNet dynamic 大约是 `blocks * layers * support * order = 8 * 2 * 2 = 32` 次 support multiplication，并且时间维卷积可并行。
- STGCN dynamic 只有约 2 个 ST block，每个 ChebConv 做 2 次 support multiplication，数量级更小。

因此 dynamic DCRNN 同时吃亏在：

- 时间步和预测步串行展开。
- 每步都要过多层 DCGRU。
- 每个 DCGRU 又重复多次 diffusion support multiplication。
- 当前 dynamic support 是 dense `B x N x N`。

## 当前实验结果

截至 2026-05-25，主要结果都使用 best-val checkpoint 的 test 指标。`DynamicThreshold GWNet + addaptadj=True` 的 final checkpoint test 更好，但为保持 BasicTS 统一报告口径，主表仍使用 best-val。

### Backbone 结果

| 模型 | 图设置 | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---:|---:|---:|---:|---:|
| GWNet | original SD fixed aligned | False | 20.4228 | 33.1292 | 0.1377 | 0.1244 |
| GWNet | dynamic hard | False | 19.3330 | 31.6238 | 0.1278 | 0.1150 |
| GWNet | dynamic hard | True | 19.3685 | 32.4364 | 0.1247 | 0.1122 |
| DCRNN | dynamic hard | - | 18.2713 | 30.2007 | 0.1232 | 0.1089 |
| STGCN | dynamic hard, normlap 修正版 | - | 23.3167 | 38.4368 | 0.1650 | 0.1480 |

### GWNet 对照

| 模型 | 图设置 | addaptadj | MAE | RMSE | MAPE | WAPE |
|---|---|---:|---:|---:|---:|---:|
| GWNet | original SD fixed aligned | False | 20.4228 | 33.1292 | 0.1377 | 0.1244 |
| GWNet | dynamic hard | False | 19.3330 | 31.6238 | 0.1278 | 0.1150 |
| GWNet | original SD adaptive | True | 18.9315 | 31.4135 | 0.1264 | 0.1126 |
| GWNet | OSRM beta1 adaptive | True | 18.1753 | 30.6229 | 0.1330 | 0.1157 |
| GWNet | dynamic hard | True | 19.3685 | 32.4364 | 0.1247 | 0.1122 |

### 关键观察

1. 相比 `original SD fixed aligned`，`dynamic hard` 在 GWNet 上明显更好：MAE 从 `20.4228` 降到 `19.3330`，约 `-5.34%`。
2. 给 dynamic hard 加 `addaptadj=True` 后，MAE/RMSE 没有变好：MAE `19.3330 -> 19.3685`，RMSE `31.6238 -> 32.4364`；但 MAPE/WAPE 更好。
3. 和强 baseline 比，dynamic hard 仍弱于 `GWNet original SD adaptive` 和 `GWNet OSRM beta1 adaptive`。因此当前结果不能支持“动态阈值替代 adaptive adjacency”，只能支持“动态阈值优于 fixed original graph”。
4. DCRNN dynamic hard 的绝对精度最好，但训练极慢，当前 dense dynamic support 不适合继续扩大 sweep。
5. STGCN dynamic hard 明显偏弱，需要补 static control 后再判断是动态图机制问题还是 STGCN 图算子适配问题。

### 运行与日志

| 实验 | log/checkpoint |
|---|---|
| GWNet original fixed aligned | `BasicTS/logs/adaptive_threshold_dynamic_weight/original_fixed_aligned/original_fixed_aligned_20260522_gpu0/gwnet_original_fixed_aligned.log` |
| GWNet dynamic hard no addaptadj | `BasicTS/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/dynamic_threshold_hard_priority_20260519_gpu1/hard/gwnet_hard.log` |
| GWNet dynamic hard addaptadj | `BasicTS/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/dynamic_threshold_hard_addaptadj_20260522_gpu1/hard/gwnet_hard_addaptadj.log` |
| DCRNN dynamic hard | `BasicTS/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/dynamic_threshold_hard_priority_20260519_gpu1/hard/dcrnn_hard.log` |
| STGCN dynamic hard normlap | `BasicTS/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/dynamic_threshold_hard_stgcn_normlap_20260520_gpu0/hard/stgcn_hard.log` |

训练耗时的粗略观测：

| 模型 | 当前实现 | train/epoch |
|---|---|---:|
| STGCN hard normlap | dense dynamic mask | 86-88s |
| GWNet original fixed aligned | dense fixed support | 166-178s |
| GWNet hard dynamic | dense dynamic mask | 397-426s |
| GWNet hard dynamic + addaptadj | dense dynamic mask | 307-322s |
| DCRNN hard dynamic | dense dynamic mask | 1388-1474s |
| SparseDCRNN fixed beta1 | fixed CSR support | 274-276s |

这个对比说明：DCRNN 的 recurrent 结构确实慢，但 fixed CSR support 可以显著降耗；当前 dynamic hard 慢的主要原因是 dense dynamic support 路径。

## 可选路线

### 路线 A：全节点 dense STE

```math
\mathcal{E}_{cand}=\mathcal V \times \mathcal V
```

优点：

- radius head 可学习。
- 所有潜在邻居都有 surrogate 梯度。
- 方法最干净，探索空间最大。

缺点：

- 复杂度是 `O(BN^2C)`。
- 不能支撑稀疏加速 claim。
- 在 DCRNN 上尤其慢。

适合作为准确率原型，不适合作为最终高效方法。

### 路线 B：active-only hard sparse

```math
\mathcal{E}_b=\{(i,j): d_{i,j} \le r_{b,i}\}
```

优点：

- 推理最快。
- 计算规模接近 active edge 数。

缺点：

- hard `nonzero` 选边基本不可导。
- inactive edge 没有 surrogate 梯度。
- radius head 很可能学不动或学习不稳定。

适合推理，不适合直接训练动态阈值。

### 路线 C：固定候选池 + 候选池内 STE

先建一个较大的物理候选池：

```math
\mathcal{E}_{cand}
=
\{(i,j): d_{i,j} \le \tau_{cand}\}
```

例如 beta=2 或 beta=3 的 OSRM 图。训练时只在候选池内计算：

```math
m^{ST}_{b,i,j}
=
m^{hard}_{b,i,j}
+ \sigma((r_{b,i}-d_{i,j})/\tau)
- \operatorname{stopgrad}(\sigma((r_{b,i}-d_{i,j})/\tau))
```

```math
h'_{b,i}
=
\frac{
\sum_{(i,j)\in\mathcal{E}_{cand}} m^{ST}_{b,i,j} h_{b,j}
}{
\sum_{(i,j)\in\mathcal{E}_{cand}} m^{ST}_{b,i,j}
}
```

优点：

- 复杂度从 `O(BN^2C)` 降到 `O(BE_candC)`。
- radius head 仍能在候选池内收到 surrogate 梯度。
- 和物理路网先验一致。

缺点：

- 模型不能选择候选池外的边。
- `E_cand` 如果太大，加速有限；太小，又可能损失精度。
- 需要实现 packed edge/gather 或 batched sparse-like aggregation。

这是当前最合理的训练路线。

## 当前问题总结

1. 当前 dynamic hard 图虽然是 0-1 阈值图，但实现上仍然是 dense mask，因此没有真正利用稀疏加速。
2. 全节点候选保证可学习性和探索空间，但候选规模是 `N^2`，与稀疏效率目标冲突。
3. active-only sparse 可以加速，但 hard edge selection 不可导，会破坏 radius head 的训练。
4. DCRNN 的 seq2seq recurrent 结构会把这个问题放大，因为每个 batch 有大量重复 support multiplication。
5. STGCN 对图算子语义敏感，必须区分 adjacency、transition support 和 normalized Laplacian。
6. `dynamic hard + addaptadj=True` 已补，但没有超过 `dynamic hard + addaptadj=False` 的 MAE/RMSE，也没有超过 `GWNet adaptive` 强 baseline。
7. 当前结果更支持“动态阈值图优于 fixed original graph”，但还不能支持“动态阈值图优于 learned adaptive adjacency”，也不能支持“动态阈值图已经实现稀疏加速”。

## 建议的下一步

优先级从高到低：

1. 实现 candidate-pool STE：以 beta=2 或 beta=3 OSRM 图作为候选池，只在候选池内训练动态 hard mask。
2. 在 GWNet 上先验证 candidate-pool STE 的准确率和速度，再迁移到 DCRNN。
3. 补 STGCN static controls：原始 LargeST 图、OSRM beta=1 fixed 图，判断 STGCN weak 是 backbone 问题还是动态图问题。
4. 尝试 local-anchor + dynamic band rewiring：保留近邻固定锚点，同时学习中远距离候选边。
5. DCRNN 只在 candidate-pool 版本上继续做效率实验；不要继续用全节点 dense dynamic DCRNN 扩大 sweep。
6. 推理阶段可以单独导出 active-only sparse edge list，评估真实 sparse inference speed。

## 文件结构

```text
configs/               BasicTS configs for graph variants
scripts/               graph generation, launch, and analysis scripts
src/                   prototype code for graph support and edge weights
results/               compact tables, JSON summaries, figures
outputs/               raw logs and intermediate outputs
notes/                 related work notes and design decisions
```

## 相关文档

- `EXPERIMENT_PLAN.md`
- `EXPERIMENT_TRACKER.md`
- `notes/osrm_gaussian_global_graphs_2026-05-13.md`
- `notes/osrm_gaussian_global_runs_2026-05-13.md`
- `notes/osrm_gaussian_global_gwnet_adaptive_runs_2026-05-13.md`

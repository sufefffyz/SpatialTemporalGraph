# 自适应阈值图机制：理论支撑与下一步实验计划

日期：2026-06-02

## 0. 当前目标

把现在的自适应阈值图从“模型内部 plug-in 动态邻接”重新收束成一个更可解释的图预处理 / graph rewiring 模块：

- 输入：节点坐标或两两距离矩阵，以及训练集时序信号。
- 输出：每个节点的连接预算或阈值，形成稀疏消息传递图。
- 目标：在控制图稀疏度和计算成本的同时，让经典 STGNN backbone 获得稳定提升。

## 1. 理论文献线

### 1.1 Graphical Lasso / Sparse Dependency Graph

核心用途：支撑“从多变量观测中恢复稀疏依赖图”。

推荐文献：

- Friedman, Hastie, Tibshirani, 2008, Sparse inverse covariance estimation with the graphical lasso.
  - 作用：精度矩阵稀疏化，给“稀疏图是可解释条件依赖结构”提供统计学习基础。
  - 对我们的启发：可以把图学习表述为稀疏依赖恢复，但不要直接声称我们估计的是 Gaussian precision graph。
- Hallac et al., 2017, Network Inference via the Time-Varying Graphical Lasso.
  - 作用：同时约束数据拟合、稀疏性、时间一致性。
  - 对我们的启发：动态阈值如果要保留，最好加“时间平滑/日内稳定”的正则解释。
- Ahmed and Xing, 2010, Learning Spatial-Temporal Varying Graphs with Applications to Climate Data Analysis.
  - 作用：空间-时间变化图，强调相邻时间或空间位置的图结构相似。
  - 对我们的启发：适合支撑跨领域天气/气象数据，不局限交通。

### 1.2 Smooth Graph Signal Learning

核心用途：支撑“让训练集信号在学习图上平滑”。

推荐文献：

- Dong et al., 2016, Learning Laplacian Matrix in Smooth Graph Signal Representations.
  - 作用：学习一个 Laplacian，使观测信号在图上变化较平滑。
  - 对我们的启发：自适应阈值可以被写成距离约束下的稀疏 Laplacian 学习。
- Kalofolias, 2016, How to Learn a Graph from Smooth Signals.
  - 作用：把图学习写成带稀疏性的加权优化，并指出高斯距离图是其框架的特例。
  - 对我们的启发：这条线最适合连接 LargeST 原始高斯距离图和我们的“学习阈值/预算”。

### 1.3 GNN Graph Rewiring / Message-Passing Geometry

核心用途：支撑“消息传递图不必等于物理原图，需要按任务重连”。

推荐文献：

- Topping et al., 2022, Understanding over-squashing and bottlenecks on graphs via curvature.
  - 作用：说明图拓扑会限制消息传播，rewiring 可以缓解 bottleneck。
  - 对我们的启发：我们不是泛化地学 dense adjacency，而是在距离候选池中控制局部传播瓶颈。
- Karhadkar et al., 2022, FoSR: First-order spectral rewiring for addressing oversquashing in GNNs.
  - 作用：用 spectral expansion 加边，同时通过结构区分缓解 oversmoothing。
  - 对我们的启发：支持“加边/删边必须有预算”，不能只追求更密。
- Effective Resistance Rewiring, 2026.
  - 作用：固定边预算下用有效电阻添加/移除边，显式控制 densification。
  - 对我们的启发：可以作为 GSP/effective-resistance 候选图的理论对照。

### 1.4 Deep Graph Structure Learning

核心用途：作为“我们不想做的过复杂端到端图学习”的对照。

推荐文献：

- Pro-GNN, KDD 2020.
  - 关键词：sparsity, low-rank, feature smoothness。
  - 对我们的定位：它是模型训练时联合学图；我们更想做训练前或轻量外部图预处理。
- IDGL, NeurIPS 2020.
  - 关键词：iterative graph structure learning, adaptive graph regularization, scalable anchors。
  - 对我们的定位：它证明任务驱动学图有效，但复杂度和端到端性比我们的目标更重。

## 2. 我们的方法应如何改写

建议把方法名暂时写成：

Distance-Budgeted Graph Rewiring for STGNNs

核心公式可以先保持简单：

1. 以距离矩阵给候选边：
   - full-pair: `C_i = {j != i}`
   - candidate: `C_i = TopK_i(-D_ij)` 或 `TopK_i(score_ij)`
2. 学习节点级连接预算：
   - `k_i = k_min + (k_max - k_min) sigmoid(s_i / tau)`
3. 由预算反推距离阈值：
   - `r_i = Quantile({D_ij | j in C_i}, k_i / |C_i|)`
4. 构建稀疏消息传递图：
   - `A_ij = 1[D_ij <= r_i, j in C_i]`

当前结果提示：

- GWNet 在 METR-LA 上有小幅收益。
- DCRNN / STGCN full dynamic 在 METR-LA 上变差。
- DCRNN 的 degree-quantile identity 在 PEMS-BAY 上学到的日内变化几乎为 0，更像静态节点级阈值。
- 因此优先把方法写成“任务适配的节点级距离预算图”，暂时不要过度强调强时间动态。

## 3. LargeST 实验计划

### Phase A：先立大规模 baseline

目标：确认我们要打的靶子，而不是只和弱 baseline 比。

优先级：

1. 官方 LargeST 框架直接跑：HL, LSTM, STGCN, GWNet, DCRNN, STGODE。
2. 资源允许再跑：AGCRN, D2STGNN, DGCRN, ASTGCN, STTN。
3. 已合入的可关注模型：PatchSTG, BiST。
4. BasicTS 中的强模型候选：STID, STAEformer, BigST, MTGNN, D2STGNN。

数据顺序：

1. SD 15min：最快复现和调参。
2. GBA 或 GLA 15min：验证规模扩大后是否仍有效。
3. CA 15min：只在前两步有收益且算力允许时做。

注意：当前自适应阈值线的 SD 协议应使用 `BasicTS/datasets/SD`，其 shape 为
`[35040, 716, 3]`，frequency 为 15min。LargeST 官方
`generate_data_for_training.py --dataset sd --years 2019` 会从
`sd_his_2019.h5` 生成 `[105120, 716, 3]`，即 5min 全年数据，不应混入
当前 15min 对比表。

### Phase B：测试我们的图预处理模块

保持 backbone 不改，只换图：

1. Original LargeST graph。
2. OSRM / road-distance global threshold graph。
3. OSRM K64 candidate graph。
4. GSP / smooth-signal K64 candidate graph。
5. Distance-budgeted rewiring graph：学习 `k_i` 或 `r_i` 后固定为静态图。
6. 如果静态图有效，再测轻量动态版本：日内或 batch-level threshold。

### Phase C：构建我们自己的模型

不要一开始做复杂新 backbone。建议先做：

- Backbone: GraphWaveNet 或 STGCN/GWNet-style temporal convolution。
- Graph module: Distance-Budgeted Rewiring Preprocessor。
- Sparse operator: 固定图时使用 CSR sparse matmul；动态 hard graph 先只作为分析对照。
- Optional dynamic head: 只学习节点预算 `k_i`，避免同时学权重、上下限、候选池等过多自由度。

目标 claim：

- 不是“我们发明动态图学习”。
- 而是“给定物理距离，学习任务适配的稀疏消息传递图；在大规模 STGNN 中兼顾精度和效率”。

## 4. 如果 LargeST 打不动：跨领域数据路线

优先标准：有坐标或可计算两两距离、节点数不太小、任务是多站点时序预测。

候选：

1. PeakWeather
   - 优点：2025 新数据，明确支持 forecasting / graph structure learning，站点测量 + NWP/地形信息。
   - 风险：需要新预处理，可能不适配 BasicTS 标准交通输入。
2. KnowAir / HighAir / China AQI
   - 优点：空气质量扩散天然依赖距离、风向、地理因素；和动态边权有自然动机。
   - 风险：公开数据版本、站点元数据、复现实验协议需要核对。
3. WeatherBench2
   - 优点：公开、标准、规模大。
   - 风险：它是网格天气，不是站点图；若转图，容易变成规则网格实验，和“距离阈值传感器图”动机不完全一致。
4. Weather2K
   - 优点：大量气象站，适合地理距离图。
   - 风险：需要先查数据可得性和预处理成本。

推荐顺序：

1. 先继续交通：SD -> GBA/GLA。
2. 若交通结果不稳定，转 PeakWeather 或 KnowAir。
3. WeatherBench2 放在最后，除非我们想写成“地理网格/球面图 rewiring”。

## 5. 下一步行动

1. 整理理论 related work 小节草稿：graphical lasso / smooth graph learning / GNN rewiring 三段。
2. 在服务器上盘点 LargeST 官方框架已有数据和 baseline 是否可直接运行。
3. 先跑 SD 15min 上的 baseline：优先使用已存在的 BasicTS 15min SD 配置；
   若要使用 LargeST 官方代码，需要先确认或改造为 15min 输入，不能直接用
   官方 5min 生成文件。
4. 用同一协议补测我们的静态 distance-budgeted rewiring graph。
5. 只有静态图胜出后，再考虑动态 threshold head。

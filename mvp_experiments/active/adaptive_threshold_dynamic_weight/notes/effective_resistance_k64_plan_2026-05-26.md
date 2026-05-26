# 有效电阻 K64 候选图实验计划

## 目标

在 OSRM K64 / GSP K64 之外补一个 `effective_resistance K64` 候选图，检验“多跳拓扑可达性”是否比纯距离或训练信号平滑性更适合作为稀疏候选池。

## 图构建

- 基图：SD 原始物理图 `BasicTS/datasets/SD/adj_mx.pkl`。
- 有效电阻：先对基图去自环并对称化为 conductance graph，再由 Laplacian 伪逆计算
  `R_ij = L^+_ii + L^+_jj - 2 L^+_ij`。
- 候选边：每个节点保留 `R_ij` 最小的 64 个邻居。
- 边权：第一版沿用 OSRM Gaussian 距离权重，只改变候选边选择；为避免远距离边权下溢成 0，保留边做最小正值裁剪。
- 数据集：`SD_EFFRESTOPK_K064`。

## 首批训练

1. `GraphWaveNet + SD_EFFRESTOPK_K064`：固定候选图，和 OSRM/GSP K64 fixed 对比。
2. `DynamicThresholdGraphWaveNet + effective-resistance K64 candidate`：在同一候选池内动态开关边，和 OSRM/GSP K64 dynamic 对比。

DCRNN 先不抢当前 GPU；如果 GWNet 结果有信号，再排正常 `DCRNN` 和 `DynamicThresholdDCRNN`。

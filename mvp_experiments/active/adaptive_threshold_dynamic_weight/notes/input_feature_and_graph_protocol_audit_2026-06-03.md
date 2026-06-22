# 输入特征与固定图协议审计（2026-06-03）

## 结论

1. SD / LargeST 不能只用 `[target, tod]`。现有 `datasets/SD/desc.json` 是 3 通道：
   `[traffic flow, time of day, day of week]`。
2. LargeST 官方公共配置默认 `input_dim=3`，dataloader 使用 `data[..., :args.input_dim]`。
3. 因此 SD / GBA / GLA / CA 的 LargeST 对齐实验应使用 `FORWARD_FEATURES=[0,1,2]`。
4. KnowAir / CCAQ 不能把 `[target, tod, dow]` 称为多协变量或 MAGE-style。
5. 空气数据正式实验必须先恢复 full-covariate 输入，并配套固定 KNN 图。

## 当前有效/无效状态

| 数据集 | 当前数据 | 当前状态 | 原因 |
|---|---|---|---|
| SD | `datasets/SD`, 3 通道 | 可继续，但需三通道 | LargeST 官方默认 `input_dim=3` |
| SD 图变体 | `SD_*`，通常继承 3 通道 | 可继续，但需三通道 | 图变体不应改变输入协议 |
| KnowAir | 当前 BasicTS 版 3 通道 | 不用于正式 claim | 只含 PM2.5 + 时间特征，丢失气象协变量 |
| CCAQ | 当前 BasicTS 版 3 通道 | 不用于正式 claim | 只含 AQI + 时间特征，丢失官方 x/u 协变量 |
| KnowAir KNN | K32/K64 图已生成 | 可复用图，不可复用旧结果 | 图可作为固定候选池，数据需重造 |
| CCAQ KNN | K32/K64 图已生成 | 可复用图，不可复用旧结果 | 图可作为固定候选池，数据需重造 |

## 已核实的 raw 信息

### KnowAir

- `KnowAir.npy` shape: `(11688, 184, 18)`。
- 第 17 个 raw channel 是 PM2.5。
- 其他 17 个 raw channel 是气象/空间相关协变量。
- MAGE 官方 `get_dataset_info("knowair")` 使用 `input_dim=13`，因此不是简单的 `[PM2.5, tod, dow]`。

### CCAQ

- `train_x.npy` shape: `(14259, 24, 209, 8)`。
- `train_y.npy` shape: `(14259, 6, 209)`。
- `train_u.npy` shape: `(14259, 3)`。
- `x[..., 7]` 与 `y[:,0]` 的滑窗关系一致，因此第 7 个 x channel 是预测目标。
- MAGE 官方 `get_dataset_info("ccaq")` 使用 `input_dim=10`，很可能由 `8 个 x channel + 2 个时间特征` 构成，但具体时间特征选择需确认，不能猜。

## 下一步协议

### 先做

| 任务 | 数据 | 输入特征 | 图 | 备注 |
|---|---|---|---|---|
| SD baseline rerun | SD | `[0,1,2]` | original / KNN / dynamic candidate | 走 LargeST-aligned 或显式三通道 |
| SD dynamic rerun | SD | `[0,1,2]` | full / K32 / K64 | 与 baseline 完全一致输入 |
| KnowAir full-covariate data | raw KnowAir | PM2.5 first + covariates | KNN K32/K64 | 先生成并核对 shape/feature order |

### 暂缓

| 任务 | 原因 |
|---|---|
| CCAQ full-covariate BasicTS 训练 | 官方数据是预切 window，BasicTS 默认连续切窗会改变 split；需要 custom dataset 或严格重建连续序列 |
| CCAQ MAGE-style 10 维 | 需要确认 8 个 x channel 外到底使用哪 2 个 u/time feature |
| 空气数据 claim | full-covariate + fixed graph + dynamic graph 结果出来前不做 claim |

## 已做防呆

- `GWNet/NoAdaptive.py`：LargeST 命名数据默认使用 `[0,1,2]`。
- `AdaptiveGraph/sd_dynamic_common.py`：GWNet/DCRNN 动态阈值在 LargeST 命名数据上默认使用 `[0,1,2]`。
- 其他数据集不自动猜多通道；需要通过 `BASICTS_FORWARD_FEATURES` 显式指定。

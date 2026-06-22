# Implementation Blueprint: Physical Graph Benchmark (SD Dataset)

**目标：** 在 LargeST SD 数据集上验证物理有向图相对于距离阈值图的优势
**数据集：** LargeST SD（District 11，San Diego，716 传感器，2019年，5分钟，Flow only）
**模型：** DCRNN + GWNet
**验证 Claim：** 方向性 / 边语义 / 稠密度

---

## 一、整体数据流

```
LargeST 原始数据
  ca_his_raw_2019.h5          PeMS D11 全CA 5分钟 Flow 数据
  sd_meta.csv                 716传感器元信息（ID, Lat, Lng, Type, Fwy...）
  sd_rn_adj.npy               LargeST 原版路网距离图 (716×716)
        │
        ▼
[Step 0a] build_sd_5min.py
        │  过滤716个 SD 传感器，不做 resample，保留5分钟分辨率
        ▼
  SD_5min_raw/
    sd_his_5min_2019.h5       (T≈105120, 716) Flow 时序
    sensor_ids.npy            (716,) PeMS station ID
        │
        ▼
[Step 0c] generate_sd_5min_dataset.py
        │  添加 time_of_day / day_of_week 特征，转为 memmap 格式
        ▼
  SD_5min_full/
    data.dat                  (T, 716, 3) float32
                                channel 0 = flow
                                channel 1 = time_of_day
                                channel 2 = day_of_week
    shape.npy
    adj_mx_largeST_original.pkl  (716,716) LargeST 原版图
    sensor_ids.npy → 复制到 graphs/SD/
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[Step 1] build_stable_sensor_graph.py          [Step 2] build_road_dist_adj.py
         （已有 pipeline，需加 --fixed-sensor-ids）    （不需要，SD只用LargeST原版图）
         输入：D11 station_meta + SHN 文件
         输出：graphs/SD/
           adj_mx_physical_directed.pkl
           directed_edges.csv
        │
        ▼
[Step 3] build_graph_variants.py
         输入：adj_mx_physical_directed.pkl + directed_edges.csv
         输出：graphs/SD/
           adj_mx_physical_bidir.pkl
           adj_mx_physical_ML_only.pkl
           adj_mx_physical_forward.pkl   （同 directed，显式命名）
           adj_mx_physical_reverse.pkl   （转置）
        │
        ▼
[Step 4] build_dataset_dirs.py
         为每种图 × 每个训练窗口生成 BasicTS 数据集目录
         输出：BasicTS/datasets/SD_LargeST_{window}_{graph}/
           data.dat     （软链接或复制）
           adj_mx.pkl   （对应图）
           desc.json
        │
        ▼
[Step 5] generate_configs.py
         输出：BasicTS/baselines/{DCRNN,GWNet}/SD_LargeST_{window}_{graph}.py
        │
        ▼
[Step 6] run_pilot.sh
         训练 2模型 × 5图 × 2窗口 × 3seeds = 60次
        │
        ▼
[Step 7] stratified_eval.py
         输出：benchmark/eval/sd_results.csv
```

---

## 二、目录结构

```
/Users/richardo/Desktop/STproject/
│
├── data/                                    [已有]
│   └── （SD 数据无需下载，来自 LargeST）
│
└── SpatialTemporalGraph/
    │
    ├── LargeST/data/                        [已有]
    │   ├── ca/
    │   │   └── ca_his_raw_2019.h5           ← 需确认是否存在
    │   └── sd/
    │       ├── sd_meta.csv                  ← 716传感器信息
    │       ├── sd_rn_adj.npy                ← LargeST原版图
    │       └── generate_sd_dataset.ipynb
    │
    ├── BasicTS/                             [已有]
    │   ├── baselines/
    │   │   ├── DCRNN/
    │   │   │   ├── PEMS07.py               ← 参照模板
    │   │   │   └── SD_LargeST_*.py         ← 待生成（Step 5）
    │   │   └── GWNet/
    │   │       ├── PEMS07.py               ← 参照模板
    │   │       └── SD_LargeST_*.py         ← 待生成（Step 5）
    │   ├── datasets/                        ← 待生成（Step 4）
    │   │   ├── SD_LargeST_full_largeST_original/
    │   │   ├── SD_LargeST_full_physical_forward/
    │   │   ├── SD_LargeST_full_physical_reverse/
    │   │   ├── SD_LargeST_full_physical_bidir/
    │   │   ├── SD_LargeST_full_physical_ML_only/
    │   │   ├── SD_LargeST_1m_largeST_original/
    │   │   └── ...（共10个目录）
    │   └── experiments/train.py            ← 训练入口，不修改
    │
    ├── graphs/                              ← 待生成
    │   └── SD/
    │       ├── sensor_ids.npy              ← (716,) int64
    │       ├── directed_edges.csv          ← src_id, dst_id, edge_type
    │       ├── adj_mx_largeST_original.pkl ← (716,716) float32
    │       ├── adj_mx_physical_directed.pkl
    │       ├── adj_mx_physical_forward.pkl
    │       ├── adj_mx_physical_reverse.pkl
    │       ├── adj_mx_physical_bidir.pkl
    │       └── adj_mx_physical_ML_only.pkl
    │
    ├── datasets/                            ← 中间产物，待生成
    │   ├── SD_5min_raw/
    │   │   ├── sd_his_5min_2019.h5
    │   │   ├── sd_meta.csv
    │   │   ├── sd_rn_adj.npy
    │   │   └── sensor_ids.npy
    │   └── SD_5min_full/
    │       ├── data.dat                    ← (T, 716, 3) float32
    │       └── shape.npy
    │
    └── benchmark/                          ← 全部新建
        ├── IMPLEMENTATION_BLUEPRINT.md     ← 本文件
        ├── graph_baselines/
        │   └── build_graph_variants.py
        ├── data_prep/
        │   ├── build_sd_5min.py
        │   └── generate_sd_5min_dataset.py
        ├── configs/
        │   └── generate_configs.py
        └── eval/
            └── stratified_eval.py
```

---

## 三、图变体说明

SD 实验使用以下 5 种图，覆盖三个核心 Claim：

| 图名 | 方向 | 边语义 | 稠密度 | 验证目标 |
|---|---|---|---|---|
| `largeST_original` | 无向 | 无区分 | 稠密（距离阈值） | 基准（LargeST原版） |
| `physical_forward` | **有向**（上游→下游） | ML+OR+FR | 稀疏 | Claim 1：方向性 |
| `physical_reverse` | **有向**（下游→上游） | ML+OR+FR | 稀疏 | Claim 1：方向反转 |
| `physical_bidir` | 无向（对称化） | ML+OR+FR | 稀疏 | Claim 1：消融方向 |
| `physical_ML_only` | 有向 | **仅 ML→ML** | 稀疏 | Claim 2：边语义 |

### Claim 对应的关键对比

**Claim 1（方向性）：** `physical_forward` vs `physical_bidir`
> 两者边集相同，唯一差异是方向。若 forward > bidir，说明方向本身有贡献。

**Claim 2（边语义）：** `physical_forward` vs `physical_ML_only` vs `largeST_original`
> ML_only 去掉了 OR→ML 和 ML→FR 边，对比 Ramp 节点预测误差变化。

**Claim 3（稠密度）：** `physical_forward` vs `largeST_original`
> 同为有效连接，物理图极稀疏（~1.5边/节点），LargeST 图稠密（~20+边/节点）。

---

## 四、训练窗口设计

**固定 val + test，只改变 train 长度：**

```
|←────────── 全年2019（T≈105120步）────────────────→|
|                         |      val 20%  | test 20% |
|  train_full（60%）       |               |          |
|               train_1m  |               |          |
                ↑                         ↑          ↑
           T - T_val - T_test        T - T_test      T
```

| 窗口 | 训练步数 | 对应时长 |
|---|---|---|
| `full` | T × 60% ≈ 63072步 | 约219天 |
| `1m` | 31 × 288 = 8928步 | 约1个月 |

val 和 test 对所有窗口完全相同，结果可直接横向比较。

---

## 五、实验矩阵

**总训练次数：2模型 × 5图 × 2窗口 × 3seeds = 60次**

| 模型 | 图类型 | 训练窗口 | seeds |
|---|---|---|---|
| DCRNN | largeST_original | full, 1m | 0,1,2 |
| DCRNN | physical_forward | full, 1m | 0,1,2 |
| DCRNN | physical_reverse | full, 1m | 0,1,2 |
| DCRNN | physical_bidir | full, 1m | 0,1,2 |
| DCRNN | physical_ML_only | full, 1m | 0,1,2 |
| GWNet | largeST_original | full, 1m | 0,1,2 |
| GWNet | physical_forward | full, 1m | 0,1,2 |
| GWNet | physical_reverse | full, 1m | 0,1,2 |
| GWNet | physical_bidir | full, 1m | 0,1,2 |
| GWNet | physical_ML_only | full, 1m | 0,1,2 |

**评估指标：** MAE / RMSE / MAPE，分层报告（ALL / ML / Ramp / Interchange）

---

## 六、接口约定

### adj_mx.pkl
- 内容：`numpy.ndarray`，shape `(N, N)`，dtype `float32`
- 直接 `pickle.dump(ndarray, file)`，不包装为 list
- 存原始权重，归一化由 BasicTS 的 `load_adj()` 处理

### data.dat
- 格式：`numpy.memmap`，dtype `float32`，shape `(T, N, 3)`
- Channel 0 = flow（整数值，单位：辆/5min）
- Channel 1 = time_of_day（0.0 ~ 1.0）
- Channel 2 = day_of_week（0/7 ~ 6/7）

### desc.json 必须字段
```json
{
  "name": "SD_LargeST_{window}_{graph}",
  "shape": [T, 716, 3],
  "num_time_steps": T,
  "num_nodes": 716,
  "num_features": 3,
  "frequency (minutes)": 5,
  "has_graph": true,
  "regular_settings": {
    "INPUT_LEN": 12,
    "OUTPUT_LEN": 12,
    "TRAIN_VAL_TEST_RATIO": [train_ratio, val_ratio, test_ratio],
    "NORM_EACH_CHANNEL": false,
    "RESCALE": true,
    "METRICS": ["MAE", "RMSE", "MAPE"],
    "NULL_VAL": 0.0
  }
}
```

### sd_meta.csv 字段说明
```
列名：ID, Lat, Lng, District, County, Fwy, Lanes, Type, Direction, ID2
Type 取值：Mainline（对应ML）/ Onramp（对应OR）/ Offramp（对应FR）
注意：LargeST 的 Type 命名与 PeMS 不同（Mainline vs ML），处理时需映射
```

### directed_edges.csv 字段说明
```
列名：src_id, dst_id, edge_type
edge_type 取值：ML_ML / OR_ML / ML_FR
src_id / dst_id：PeMS station ID（整数），与 sensor_ids.npy 中的值对应
```

---

## 七、各脚本说明

### Step 0a：`data_prep/build_sd_5min.py`

**作用：** 从 LargeST 原始 H5 中提取 SD 传感器的5分钟 Flow 数据

**输入：**
- `LargeST/data/ca/ca_his_raw_2019.h5`：pandas HDF5，index=DatetimeIndex(5min)，columns=station_id(str)，值=flow
- `LargeST/data/sd/sd_meta.csv`：716传感器信息
- `LargeST/data/sd/sd_rn_adj.npy`：LargeST原版图

**输出：** `datasets/SD_5min_raw/`
- `sd_his_5min_2019.h5`：shape (T, 716)，columns=station_id(str)
- `sensor_ids.npy`：shape (716,)，dtype int64，按 sd_meta.csv 行顺序
- `sd_meta.csv`、`sd_rn_adj.npy`：直接复制

**关键逻辑：**
1. 读取 sd_meta.csv，按原始行顺序提取 ID 列（不排序）
2. 将整数 ID 转为字符串，匹配 ca_his_raw 的列名
3. `ca_his_raw.reindex(columns=sensor_ids_str, fill_value=0)`
4. `fillna(0)`
5. **不做 `resample('15T')`**（这是与 LargeST 原版的唯一差异）

> 参考代码见原始 plan 中 Script 0a

---

### Step 0c：`data_prep/generate_sd_5min_dataset.py`

**作用：** 将 H5 转为 BasicTS memmap 格式，添加时间特征，构建 LargeST 图 pkl

**输入：** `datasets/SD_5min_raw/`

**输出：** `datasets/SD_5min_full/`
- `data.dat`：shape (T, 716, 3)，channel=[flow, time_of_day, day_of_week]
- `shape.npy`
- `adj_mx_largeST_original.pkl`：由 `sd_rn_adj.npy` 转换

同时输出：`graphs/SD/sensor_ids.npy`、`graphs/SD/adj_mx_largeST_original.pkl`

**time_of_day 计算（与 LargeST 完全一致）：**
```
time_ind[t] = (timestamp[t] - 当天00:00) / 24h   ∈ [0.0, 1.0)
```

**day_of_week 计算（与 LargeST 完全一致）：**
```
dow[t] = df.index.dayofweek[t] / 7   ∈ {0/7, 1/7, ..., 6/7}
```

> 参考代码见原始 plan 中 Script 0c

---

### Step 1：`build_stable_sensor_graph.py`（已有 pipeline，需修改）

**作用：** 为 SD 的 716 个传感器构建物理有向图

**现有 pipeline 的问题：** 默认会按 `presence_ratio` 自动筛选稳定传感器，会改变节点集。需要增加 `--fixed-sensor-ids` 参数，跳过稳定性筛选，强制使用指定的 716 个传感器。

**需要修改的逻辑：**
- 新增参数 `--fixed-sensor-ids <path>`：若指定，跳过 presence_ratio 过滤，直接使用该文件中的 ID
- 输入 `graphs/SD/sensor_ids.npy`（来自 Step 0c）
- 调用命令：`--districts 11 --fixed-sensor-ids graphs/SD/sensor_ids.npy --shn-shapefile ... --output-dir graphs/SD/`

**输出：** `graphs/SD/`
- `adj_mx_physical_directed.pkl`
- `directed_edges.csv`（含 edge_type：ML_ML / OR_ML / ML_FR）

> 注意：sd_meta.csv 中 Type 为 "Mainline"/"Onramp"/"Offramp"，pipeline 内部使用 "ML"/"OR"/"FR"，需做映射

---

### Step 3：`graph_baselines/build_graph_variants.py`

**作用：** 从 `adj_mx_physical_directed.pkl` 派生其余4种图变体

**输入：** `graphs/SD/`
- `adj_mx_physical_directed.pkl`：shape (716, 716)，有向图
- `sensor_ids.npy`
- `directed_edges.csv`

**输出：** `graphs/SD/`
- `adj_mx_physical_bidir.pkl`：`max(A, A.T)`
- `adj_mx_physical_ML_only.pkl`：只保留 `edge_type==ML_ML` 的边
- `adj_mx_physical_forward.pkl`：同 directed（显式复制，命名统一）
- `adj_mx_physical_reverse.pkl`：`A.T`

**adj_mx 内容格式检查：** pickle load 后若为 list，取 `obj[0]`；若为 ndarray，直接使用。统一转为 `np.float32` ndarray 后再保存。

> 参考代码见原始 plan 中 Script 1

---

### Step 4：`data_prep/build_dataset_dirs.py`

**作用：** 为每种图 × 每个训练窗口生成 BasicTS 数据集目录

**输入：**
- `datasets/SD_5min_full/data.dat`：shape (T, 716, 3)
- `datasets/SD_5min_full/shape.npy`
- `graphs/SD/adj_mx_*.pkl`：5种图

**输出：** `BasicTS/datasets/SD_LargeST_{window}_{graph}/`（共10个目录）
- `data.dat`：截取对应时间窗口，shape (T_window, 716, 3)
- `adj_mx.pkl`：对应图的 pkl 文件（复制）
- `desc.json`：包含 `name`、`shape`、`regular_settings` 等字段

**窗口切分逻辑：**
- `T_test = int(T * 0.2)`，`T_val = int(T * 0.2)`（固定，不随窗口变化）
- `full`：`T_train = T - T_test - T_val`
- `1m`：`T_train = min(31 * 288, T - T_test - T_val)`
- 数据截取：`data[-( T_train + T_val + T_test ):]`

> 参考代码见原始 plan 中 Script 4

---

### Step 5：`configs/generate_configs.py`

**作用：** 批量生成 DCRNN 和 GWNet 的 config 文件

**输入：**
- `BasicTS/datasets/SD_LargeST_*/desc.json`（读取 num_nodes）
- `BasicTS/baselines/DCRNN/PEMS07.py`（参照模板）
- `BasicTS/baselines/GWNet/PEMS07.py`（参照模板）

**输出：** `BasicTS/baselines/{DCRNN,GWNet}/SD_LargeST_{window}_{graph}.py`（共20个文件）

**DCRNN 关键参数：**
- `load_adj(..., "doubletransition")`：生成正向+反向两个扩散矩阵
- `FORWARD_FEATURES = [0, 1, 2]`，`TARGET_FEATURES = [0]`（只预测 flow）
- `input_dim = 1`（DCRNN 只用 flow channel 作为 GNN 输入，时间特征由 runner 处理）

**GWNet 关键参数：**
- `load_adj(..., "doubletransition")`
- `addaptadj = True`（保留自适应图，对比固定图效果）
- `FORWARD_FEATURES = [0, 1, 2]`，`TARGET_FEATURES = [0]`

**注意：** `n_vertex` / `num_nodes` 从对应的 `desc.json` 读取，不硬编码。

> 参考代码见原始 plan 中 Script 5，DCRNN/GWNet template

---

### Step 6：`run_pilot.sh`

**作用：** 串行运行全部60次训练

```bash
#!/usr/bin/env bash
cd SpatialTemporalGraph/BasicTS

GRAPHS=(largeST_original physical_forward physical_reverse physical_bidir physical_ML_only)
WINDOWS=(full 1m)
MODELS=(DCRNN GWNet)
SEEDS=(0 1 2)
GPU=${1:-0}

for MODEL in "${MODELS[@]}"; do
  for WINDOW in "${WINDOWS[@]}"; do
    for GRAPH in "${GRAPHS[@]}"; do
      DATASET="SD_LargeST_${WINDOW}_${GRAPH}"
      CFG="baselines/${MODEL}/${DATASET}.py"
      [ -f "$CFG" ] || { echo "SKIP: $CFG"; continue; }
      for SEED in "${SEEDS[@]}"; do
        echo "▶ $MODEL | $DATASET | seed=$SEED"
        python experiments/train.py --cfg "$CFG" --gpus "$GPU" --seed "$SEED" \
          2>&1 | tee "logs/${DATASET}_${MODEL}_seed${SEED}.log"
      done
    done
  done
done
```

---

### Step 7：`eval/stratified_eval.py`

**作用：** 读取训练结果，按传感器类型分层汇报 MAE/RMSE/MAPE

**输入：**
- `BasicTS/logs/*/test_results.npz`：`prediction` (T,N,H)，`target` (T,N,H)
- `graphs/SD/sensor_ids.npy`
- `graphs/SD/directed_edges.csv`
- `LargeST/data/sd/sd_meta.csv`

**传感器分层维度：**

| 分组 | 定义 |
|---|---|
| ALL | 全部716个节点 |
| ML | `Type == "Mainline"` |
| Ramp | `Type in ["Onramp", "Offramp"]` |
| Interchange | 有 OR→ML 或 ML→FR 边连接的节点（交汇口） |

**输出：** `benchmark/eval/sd_results.csv`

列：`model, graph, window, group, mae_mean, mae_std, rmse_mean, rmse_std, mape_mean, mape_std, n_seeds`

> 参考代码见原始 plan 中 Script 7，注意 sd_meta.csv 的 Type 字段为 "Mainline"/"Onramp"/"Offramp"（非 ML/OR/FR）

---

## 八、执行顺序

```bash
cd /Users/richardo/Desktop/STproject

# [前置] 确认 ca_his_raw_2019.h5 存在
ls SpatialTemporalGraph/LargeST/data/ca/ca_his_raw_2019.h5

# Step 0a：提取 SD 5分钟数据（~5分钟）
python SpatialTemporalGraph/benchmark/data_prep/build_sd_5min.py

# Step 0c：转为 BasicTS 格式（~5分钟）
python SpatialTemporalGraph/benchmark/data_prep/generate_sd_5min_dataset.py

# Step 1：构建 SD 物理图（需 SHN 文件，~30分钟）
python SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_stable_sensor_graph.py \
  --districts 11 \
  --fixed-sensor-ids SpatialTemporalGraph/graphs/SD/sensor_ids.npy \
  --shn-shapefile <SHN_SHAPEFILE_PATH> \
  --shn-postmiles-geojson <SHN_POSTMILES_PATH> \
  --output-dir SpatialTemporalGraph/graphs/SD/

# Step 3：生成图变体（~1分钟）
python SpatialTemporalGraph/benchmark/graph_baselines/build_graph_variants.py

# Step 4：生成数据集目录（~5分钟）
python SpatialTemporalGraph/benchmark/data_prep/build_dataset_dirs.py

# Step 5：生成 config 文件（~1分钟）
python SpatialTemporalGraph/benchmark/configs/generate_configs.py

# Step 6：训练（60次，约40~80 GPU小时）
bash run_pilot.sh 0

# Step 7：评估
python SpatialTemporalGraph/benchmark/eval/stratified_eval.py
```

---

## 九、预期结果表结构

```
model   graph               window  group        mae_mean  mae_std
──────────────────────────────────────────────────────────────────
DCRNN   largeST_original    full    ALL           x.xx      x.xx
DCRNN   largeST_original    full    ML            x.xx      x.xx
DCRNN   largeST_original    full    Ramp          x.xx      x.xx
DCRNN   largeST_original    full    Interchange   x.xx      x.xx
DCRNN   physical_forward    full    ALL           x.xx      x.xx
...
GWNet   largeST_original    full    ALL           x.xx      x.xx
...
```

**决策规则：**

| 结果 | 结论 |
|---|---|
| `physical_forward` < `physical_bidir`（Ramp/Interchange 行） | Claim 1 成立：方向性有贡献 |
| `physical_forward` < `physical_ML_only`（Ramp 行） | Claim 2 成立：OR/FR 边语义有贡献 |
| `physical_forward` < `largeST_original`（ALL 行） | 物理图整体优于 LargeST 距离图 |
| GWNet 各图差距 < DCRNN 各图差距 | 自适应图学习能补偿图质量差异 |
| `1m` 窗口差距 > `full` 窗口差距 | 低数据量下物理图优势更显著 |

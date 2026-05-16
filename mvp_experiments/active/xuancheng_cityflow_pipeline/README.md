# Xuancheng CityFlow Road-Aggregation MVP

One-sentence thesis:

> Use the official Xuancheng CityFlow release as a simulation-backed traffic dataset, then preprocess per-second simulated lane states into PeMS-style road-level tensors for stress-test experiments.

## Dataset Status

- Source paper/data: Scientific Data 2026 Xuancheng release, Figshare article `29925824`, DOI `10.6084/m9.figshare.29925824.v5`.
- Official code: `sysu-mqz/Hierarchical_traffic_control_platform`.
- Released Xuancheng time span: `2023-04-01` through `2023-04-30`, exactly 30 daily flow files.
- Road network scale inspected from released files: 1,744 CityFlow roads, 3,546 SUMO lanes in the SUMO export.
- Important limitation: this is one month, not three months or one year. It is suitable for a pilot/robustness benchmark, but too short for claiming LargeST-scale temporal coverage.

## Scope

- Primary dataset: Xuancheng CityFlow daily flow files.
- Simulator: CityFlow first; SUMO export is kept as secondary metadata.
- Output: dense road-level aggregation every 60 seconds by default.
- Must not claim: full-city all-vehicle ground truth, 3-month or 1-year temporal coverage, or raw roadside sensor observations.

## Official-Code Alignment

The pipeline keeps the official CityFlow release format:

- Uses the released `roadnet_xuancheng250319.json` and daily `data_2023_04_DD_type_filtered.json` flow files.
- Uses CityFlow's `Engine.next_step()`, `get_lane_vehicles()`, `get_vehicle_speed()`, and `get_current_time()` APIs, matching the official API test style.
- Avoids CityFlow replay logs by default because official replay splitting code treats them as bulky artifacts.

One deliberate preprocessing choice is exposed in `make_cityflow_config.py`: default traffic-light mode is fixed-time (`rlTrafficLight=false`) so a no-agent replay follows roadnet signal phases. Pass `--tl-mode official_rl` to preserve the official training config's `rlTrafficLight=true`.

## Storage Estimate

For the released 30 days:

- Raw daily flow JSON files: about 5.41 GiB.
- Road-level dense float32 tensor, 1-min buckets, 1,744 roads:
  - 3 features: about 862 MiB/month.
  - 4 features: about 1.12 GiB/month.
  - 5 features: about 1.40 GiB/month.
- Long road CSV, gzip-compressed: depends on values, but budget roughly 5.61 GiB uncompressed for one month.
- Replay logs: avoid for this pipeline unless debugging; official utility splits logs around 0.4 GB chunks.

## First Run Order

1. R001 one-day smoke: `2023-04-03`, 1,800 seconds, 1-min buckets, verify schema and speed.
2. R002 one-day full: `2023-04-03`, 86,400 seconds, dense CSV plus NPZ tensor.
3. R003 one-week pilot: `2023-04-01` through `2023-04-07`, check weekday/weekend variation and missing/zero roads.
4. R004 full released month: all 30 days, then convert to BasicTS-style dataset.
5. R005 downstream stress-test: train/evaluate simple STGNN baselines on normal versus event/weather-like perturbation scenarios.

## Commands

Estimate size:

```bash
python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/estimate_storage.py
```

Download one day plus roadnet/configs:

```bash
python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/download_xuancheng_figshare.py \
  --data-root /data/yuzhang_fei/xuancheng_cityflow \
  --days 2023-04-03
```

Build a CityFlow config:

```bash
python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/make_cityflow_config.py \
  --data-root /data/yuzhang_fei/xuancheng_cityflow \
  --date 2023-04-03 \
  --output /data/yuzhang_fei/xuancheng_cityflow/configs/config_xuancheng_2023-04-03_roadagg.json
```

Run road aggregation in an environment with `cityflow` installed:

```bash
python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_cityflow_road_aggregation.py \
  --config /data/yuzhang_fei/xuancheng_cityflow/configs/config_xuancheng_2023-04-03_roadagg.json \
  --date 2023-04-03 \
  --duration 86400 \
  --bucket-seconds 60 \
  --output-dir /data/yuzhang_fei/xuancheng_cityflow/road_agg
```

Server convenience wrapper:

```bash
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_one_day.sh 2023-04-03
```

The wrapper first tries native Python with `cityflow`; if unavailable, it tries the official Docker image `kingsleycl/cityflow_env:latest`.

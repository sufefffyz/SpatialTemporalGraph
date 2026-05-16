# Paper-Style Max-Pressure First Run

Date: 2026-05-16

## Official Paper Setting Target

- Paper validation setting: Xuancheng state-data validation uses max-pressure signal control for the 17:00-18:00 window.
- CityFlow config mode: `rlTrafficLight=true`.
- Official config timing carried over: `step_time=10`, `all_red_time=3`.
- This is signal-control state generation only. It does not include the paper's separate vehicle route guidance or perimeter-control demonstrations.

## Completed First Artifact

Command family:

```bash
/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python \
  mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_cityflow_paper_mp_road_aggregation.py \
  --config /data/yuzhang_fei/xuancheng_cityflow/configs/config_xuancheng_2023-04-03_paper_mp.json \
  --date 2023-04-03 \
  --start-second 0 \
  --duration 3600 \
  --bucket-seconds 60 \
  --output-dir /data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp
```

Artifacts:

- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur3600.csv.gz`
- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur3600.npz`
- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur3600.summary.json`

Observed stats:

- Tensor shape: `(60, 1744, 5)`.
- Features: `entered_veh`, `exited_veh`, `mean_active_veh`, `mean_speed_kmh`, `density_veh_per_lane_km`.
- Signal intersections controlled: 116.
- Max-pressure decisions: 360.
- Phase changes counted across intersections: 15441.
- Total entered over all road buckets: 42732.
- Nonzero entered road-buckets: 28151.
- Mean nonzero speed: about 38.45 km/h.

## Strict Paper Window Status

Strict 17:00-18:00 command was launched in screen:

- screen: `xuancheng_paper_mp_1700`
- log: `/data/yuzhang_fei/xuancheng_cityflow/logs/paper_mp_1700_1800_2023-04-03.log`
- target output suffix: `start61200_dur3600`

This run must simulate from second 0 to 61200 before the aggregation window starts, so it is much slower than the `start-second 0` first artifact.

Observed result: the strict window run failed before aggregation with the CityFlow C++ router assertion:

```text
router.cpp:84: void CityFlow::Router::update(): Assertion `iCurRoad < route.end()` failed.
```

This happened with `/data/yuzhang_fei/xuancheng_cityflow/raw/data_2023_04_03_type_filtered.valid.json`, i.e. after the current route filtering/expansion preprocessing.

## Non-Official / AI-Supplemented Parts To Report

- `run_cityflow_paper_mp_road_aggregation.py` implements the paper-aligned max-pressure setting directly on CityFlow roadnet phases. It does not import the official repository's `CityFlowEnv` and `MPAgent` classes.
- The route filtering/expansion file used as input remains AI-supplemented preprocessing from this MVP pipeline.
- The road-level dense tensor builder is also AI-supplemented; the official repository exposes state APIs and validation scripts, but not a PeMS/LargeST-style tensor export.
- The completed first artifact uses the official max-pressure signal setting but starts at second 0. The strict paper time window is the separate running `start-second 61200` job.

## First Difficulties

- Strictly matching the paper's 17:00-18:00 window is expensive because CityFlow needs to advance through the earlier part of the day.
- The official flow still triggers CityFlow router assertions before reaching the 17:00 aggregation window, even after route filtering/expansion.
- Only 116 intersections are controlled by the max-pressure script because it uses non-virtual intersections with more than one traffic-light phase and available road links. The full road tensor still covers all 1744 roads.

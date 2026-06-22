# Xuancheng CityFlow Hierarchical Control Update

Date: 2026-06-02

## Current Viewable Artifacts

- [No-agent official-RL road aggregation animation](../07_Assets/xuancheng_cityflow_20260602/xuancheng_2023-04-03_roadagg_traffic_animation.html)
- [Paper max-pressure, first hour, route-repaired animation](../07_Assets/xuancheng_cityflow_20260602/xuancheng_2023-04-03_paper_mp_1h_repaired_traffic_animation.html)
- [Paper max-pressure, 1 day, nocycle-repaired demand animation](../07_Assets/xuancheng_cityflow_20260602/xuancheng_2023-04-03_paper_mp_nocycle_1d_traffic_animation.html)

![](../07_Assets/xuancheng_cityflow_20260602/xuancheng_paper_mp_nocycle_1d_network_diagnostics.png)

## What Was Actually Run

This is not the full paper hierarchical controller yet. The completed full-day run is:

- CityFlow with `rlTrafficLight=true`
- signal-level max-pressure control
- 10 s decision interval
- 3 s all-red transition
- 1 min road-level aggregation
- `dtignn_repaired_nocycle` demand, not raw official demand

Server artifacts:

- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp_nocycle_1d/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur86400.csv.gz`
- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp_nocycle_1d/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur86400.npz`
- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp_nocycle_1d/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur86400.summary.json`

## Main Results

| item | value |
|---|---:|
| tensor shape | `(1440, 1744, 5)` |
| roads | 1744 |
| lanes | 3546 |
| signal intersections controlled | 116 |
| decisions | 8640 |
| phase changes | 342567 |
| CSV size | 25 MB |
| NPZ size | 9.7 MB |

Demand usage:

| flow | records | start `<86400` | cyclic routes |
|---|---:|---:|---:|
| raw official | 377322 | 377319 | 0 |
| old `.valid.json` | 350041 | 350038 | 15602 |
| nocycle used | 334472 | 334469 | 0 |

The final used demand is smaller than raw official by 42850 trips: 27281 unreachable-anchor routes plus 15569 cyclic routes removed.

## Diagnosis

The paper max-pressure signal setting makes early traffic much more realistic than no-agent RL. However, signal-only control still gridlocks under full-day Xuancheng demand:

| hour | demand starts | entered-exited gap | stock end | mean speed |
|---:|---:|---:|---:|---:|
| 6 | 14498 | +4255 | 4900 | 33.7 |
| 7 | 33551 | +15552 | 20416 | 20.4 |
| 8 | 23351 | +8118 | 28575 | 5.39 |
| 9 | 17868 | +4801 | 33392 | 2.11 |
| 10 | 17470 | +3147 | 36539 | 0.93 |
| 17 | 34155 | +861 | 46776 | 0.18 |
| 23 | 3324 | +75 | 48121 | 0.036 |

End-of-day summary:

- `total_road_entered = 602451`
- `total_road_exited = 554331`
- `end_network_stock ~= 48121`
- `zero_rate_entered ~= 89.77%`
- `zero_rate_exited ~= 90.84%`

Interpretation: max-pressure-only is not the complete paper control scheme. The paper also discusses network-level perimeter control and vehicle-level route guidance. The current result should be described as **paper signal-control setting / max-pressure-only**, not full hierarchical control.

## Implementation Direction

Next implementation target:

1. keep signal-level max-pressure as the lower-layer controller;
2. add network-level perimeter control that meters inflow into a protected congested core when network accumulation exceeds a target;
3. add a route-guidance hook using CityFlow `set_vehicle_route`, initially optional and diagnostic-only because the public code does not provide a full official route-guidance agent.

Important deviation to report:

- The official public repository supports `set_vehicle_route` and network MFD-style metrics, but does not include a directly runnable Xuancheng full hierarchical controller.
- The perimeter and route-guidance implementation below is therefore an MVP heuristic aligned with the paper concept, not an official as-is reproduction.

## Implementation Status

Added script:

- `/home/yuzhang_fei/code/SpatialTemporalGraph/mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_cityflow_hierarchical_control_road_aggregation.py`
- local source: `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_cityflow_hierarchical_control_road_aggregation.py`

Implemented controls:

1. lower layer: paper-style 10 s max-pressure signal control, same roadnet/phase parsing as the previous paper-MP road aggregation script;
2. upper layer: MFD-style perimeter gating for boundary phases that move traffic from outside the protected core into the core;
3. optional route-guidance hook via CityFlow `set_vehicle_route`, disabled by default because the official public code does not provide a complete runnable route-guidance agent.

Smoke tests:

| test | setting | result |
|---|---|---|
| normal early-window smoke | `start=0`, `duration=7200`, target `55 veh/lane-km`, release `40 veh/lane-km` | completed, 120 buckets, 720 decisions, no perimeter activation |
| forced gating smoke | `start=0`, `duration=600`, target `0`, release `-1` | completed, perimeter activated once, 3632 boundary overrides, 2940 all-red fallbacks |
| forced route-guidance smoke | `start=0`, `duration=300`, forced perimeter, route guidance on | completed, 290 reroute attempts, 65 route changes |

Server outputs:

- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_hierarchical_mvp_smoke/xuancheng_2023-04-03_hier_mp_road_agg_60s_start0_dur7200.summary.json`
- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_hierarchical_mvp_forced/xuancheng_2023-04-03_hier_mp_road_agg_60s_start0_dur600.summary.json`
- `/data/yuzhang_fei/xuancheng_cityflow/road_agg_hierarchical_mvp_route_smoke/xuancheng_2023-04-03_hier_mp_road_agg_60s_start0_dur300.summary.json`

MFD threshold estimate from the previous full-day paper-MP run:

| item | value |
|---|---:|
| core selection | top 250 roads by `mean_active_veh` during 07:00-10:00 |
| core lane-km | 262.959 |
| peak smoothed core outflow bucket | 24060 s |
| peak stock | 1872.9 |
| peak stock density | 7.12 veh/lane-km |
| peak smoothed exited flow | 1234.2 veh/min |

Interpretation: the first default target `55 veh/lane-km` is too conservative for this core definition. A more reasonable full-day candidate is around `target=7.1`, `release=5.5`, with periodic release such as `--perimeter-open-every-n 6`.

## Running Candidate

Launched a full-day perimeter-only candidate on the server:

- screen: `xuancheng_hier_mfd_1d`
- log: `/data/yuzhang_fei/xuancheng_cityflow/logs/hier_mfd_1d_2023-04-03.log`
- output dir: `/data/yuzhang_fei/xuancheng_cityflow/road_agg_hierarchical_mfd_1d`
- target density: `7.1 veh/lane-km`
- release density: `5.5 veh/lane-km`
- periodic release: `--perimeter-open-every-n 6`
- route guidance: off for this first full-day candidate

Rationale: isolate the network-level perimeter-control effect first. The route-guidance hook passed a short interface smoke test, but it is still a heuristic implementation rather than an official public agent.

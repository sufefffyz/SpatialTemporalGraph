# Official-First Reproduction Check, 2026-05-18

Goal: pause the AI-supplemented route-filtering path and verify what the
official Xuancheng CityFlow release can do before any input modification.

## Server Setup

- Official repository checkout:
  `/home/yuzhang_fei/code/Hierarchical_traffic_control_platform_official`
- Official commit: `81e64b2`
- Data bridge: symlinks from official `data/xuancheng/` to raw Figshare files
  under `/data/yuzhang_fei/xuancheng_cityflow/raw/`
- Task Python: `/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python`
- Dependency added for official `test.py`: `orjson==3.11.9`

The data symlinks are an environment bridge only. No flow routes were filtered,
expanded, or rewritten in the strict-official checks.

## Results

| Check | Status | Evidence |
|---|---|---|
| Official `test.py` as-is | BLOCKED | Fails after `orjson` install because `data/xuancheng/trip_2023_04_04_17.json` is missing from the public raw release. |
| Official `from agent import MPAgent` | BLOCKED | Fails before simulation: `agent/__init__.py` imports baseline agents that require missing `agent.dqn_agent` source. |
| Raw `config_xuancheng_test.json` + direct CityFlow Engine | PASS | `/data/yuzhang_fei/xuancheng_cityflow/logs/official_repro_20260518_114411/direct_engine_3600s.log`; 3,600s, return code 0, 292 `Invalid route` warnings omitted by CityFlow, elapsed 41.31s. |
| Official `CityFlowEnv` + official `MPAgent` source via launcher | PASS WITH REPORTED WORKAROUND | `/data/yuzhang_fei/xuancheng_cityflow/logs/official_mp_launcher_20260518_114536/mpagent_3600s_launcher.log`; 3,600s, return code 0, elapsed 120.35s. |

## Important Deviations

- `mpagent-launcher` is not the official as-is entrypoint. It bypasses only
  `agent/__init__.py`, because that package initializer imports broken/unneeded
  baseline modules before `MPAgent` can be used.
- The earlier `*.valid.json` route filtering/expansion path is not official and
  introduced duplicate/cycle-like routes in prior checks. It should not be the
  mainline until the official raw path is exhausted.
- The failed 30-day aggregation run downloaded all 30 raw daily files but then
  failed on 2023-04-01 after the non-official route filtering/expansion step.

## Next Strict Step

Use raw official flow files and either:

1. stay with direct CityFlow Engine to build a no-agent road-aggregation
   baseline, or
2. continue with the reported `mpagent-launcher` workaround to aggregate
   official `MPAgent` simulations, while clearly labeling the launcher bypass.

Adopted immediate solution:

- Official-raw direct-engine road aggregation uses raw daily
  `data_2023_04_DD_type_filtered.json` files with `FILTER_FLOW=0`.
- The wrapper preserves the official `rlTrafficLight=true` config through
  `TL_MODE=official_rl`.
- Output is isolated under `road_agg_official_raw_direct` or
  `road_agg_30d_official_raw_direct`, separate from earlier `.valid.json`
  experiments.
- A 3,600-second smoke on `2023-04-03` succeeded with tensor shape
  `(60, 1744, 5)` and `total_entered=5408`.
- The 30-day official-raw direct-engine aggregation is running in screen
  `xuancheng_30d_official_raw_direct`.

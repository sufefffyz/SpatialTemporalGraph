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

Current strict-official status, checked on the server against official commit
`81e64b2`:

- The released raw `cfg/xuancheng/config_xuancheng_test.json` path can run
  directly with `cityflow.Engine` for at least 3,600 seconds using the original
  `data_2023_04_03_type_filtered.json`. CityFlow emits official
  `Invalid route ... Omitted by default` warnings and continues.
- The official `test.py` entrypoint is not directly reproducible from the
  public release in the current environment: after installing its direct
  `orjson` dependency, it fails because `trip_2023_04_04_17.json` is not in
  the downloaded public Xuancheng release.
- The official `from agent import MPAgent` import path currently fails before
  MPAgent is reached because `agent/__init__.py` imports baseline agents that
  require missing `agent.dqn_agent` source.
- A diagnostic launcher can run the official `CityFlowEnv` plus official
  `agent/base_agent.py` and `agent/mp_agent.py` for 3,600 seconds, but only by
  bypassing the broken `agent/__init__.py`. This is reported as a launcher
  workaround, not a fully official as-is entrypoint.

The pipeline keeps the official CityFlow release format:

- Uses the released `roadnet_xuancheng250319.json` and daily `data_2023_04_DD_type_filtered.json` flow files.
- Uses CityFlow's `Engine.next_step()`, `get_lane_vehicles()`, `get_vehicle_speed()`, and `get_current_time()` APIs, matching the official API test style.
- Avoids CityFlow replay logs by default because official replay splitting code treats them as bulky artifacts.

One deliberate preprocessing choice is exposed in `make_cityflow_config.py`: default traffic-light mode is fixed-time (`rlTrafficLight=false`) so a no-agent replay follows roadnet signal phases. Pass `--tl-mode official_rl` to preserve the official training config's `rlTrafficLight=true`.

Earlier route filtering/expansion created `*.valid.json` files and is now
treated as an experimental deviation, not the strict-official mainline. The
strict-official reproduction path must first use raw released flow files and
CityFlow's own invalid-route omission behavior.

## AI-Supplemented Pieces

These pieces are not official repository logic and should be reported as added preprocessing:

- `download_xuancheng_figshare.py`: a convenience downloader built from the Figshare file manifest.
- `make_cityflow_config.py`: a config generator that rewrites paths into CityFlow-friendly `dir + relative filename` form.
- `filter_cityflow_flow.py`: an added guard that removes flows whose route anchors are unreachable in the released roadnet and expands reachable anchor routes into full shortest paths. CityFlow supports anchor routes and fills shortest paths internally, but the full-day Xuancheng flow triggered a CityFlow C++ router assertion without this expansion.
- `run_cityflow_road_aggregation.py`: the actual 1-minute road-level aggregation into dense CSV/NPZ tensors. The official repo exposes CityFlow state APIs but does not provide this STGNN tensor builder.
- `run_official_xuancheng_repro.py`: a strict-official diagnostic runner. Its
  `direct-engine` mode uses raw official config/data. Its `mpagent-launcher`
  mode requires `--allow-bypass-agent-init` and only exists because the official
  package import chain fails before MPAgent is reached.
- `render_xuancheng_osm_map.py`: an added visualization utility that converts the released SUMO/CityFlow local coordinates back to lon/lat and renders the roadnet or road-aggregation values on an OpenStreetMap Leaflet basemap.
- Server environment workaround: Docker Hub timed out, so CityFlow was built from official `cityflow-project/CityFlow` source, then installed into the dedicated `/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow` conda environment. The earlier `/data/yuzhang_fei/xuancheng_cityflow/pydeps` directory remains the source-built staging area.

The main behavioral deviation from the official config is the default `rlTrafficLight=false` fixed-time replay. Use `TL_MODE=official_rl` only when an official-compatible signal-control loop is added; otherwise the raw official RL mode has no controlling agent in this pipeline.

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
On the current server, Docker Hub timed out, so CityFlow was built from official source and installed into a dedicated conda env. The wrapper auto-detects this Python when it exists:

```bash
DATA_ROOT=/data/yuzhang_fei/xuancheng_cityflow \
DURATION=1800 \
BUCKET_SECONDS=60 \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_one_day.sh 2023-04-03
```

To force the dedicated env explicitly:

```bash
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_one_day.sh 2023-04-03
```

For smoke checks, avoid overwriting the main `road_agg` directory:

```bash
OUTPUT_DIR=/data/yuzhang_fei/xuancheng_cityflow/road_agg_envcheck \
DURATION=60 \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_one_day.sh 2023-04-03
```

Strict-official diagnostics against the separate official checkout:

```bash
/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python \
  mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_official_xuancheng_repro.py \
  --official-repo /home/yuzhang_fei/code/Hierarchical_traffic_control_platform_official \
  --mode direct-engine \
  --duration 3600

/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python \
  mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_official_xuancheng_repro.py \
  --official-repo /home/yuzhang_fei/code/Hierarchical_traffic_control_platform_official \
  --mode mpagent-launcher \
  --allow-bypass-agent-init \
  --duration 3600
```

Run all released 30 days as 1-minute road aggregation, one file pair per day:

```bash
DATA_ROOT=/data/yuzhang_fei/xuancheng_cityflow \
OUTPUT_DIR=/data/yuzhang_fei/xuancheng_cityflow/road_agg_30d_fixed_time \
DURATION=86400 \
BUCKET_SECONDS=60 \
TL_MODE=fixed_time \
FILTER_FLOW=1 \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_month.sh
```

Current official-raw mainline, with no route filtering or expansion:

```bash
DATA_ROOT=/data/yuzhang_fei/xuancheng_cityflow \
DURATION=3600 \
BUCKET_SECONDS=60 \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_official_raw_one_day.sh 2023-04-03

DATA_ROOT=/data/yuzhang_fei/xuancheng_cityflow \
DURATION=86400 \
BUCKET_SECONDS=60 \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_official_raw_month.sh
```

These official-raw wrappers force `FILTER_FLOW=0` and preserve the official
`rlTrafficLight=true` setting through `TL_MODE=official_rl`. CityFlow's own
`Invalid route ... Omitted by default` behavior is therefore kept intact.

Lane-level official-raw smoke, NPZ only by default:

```bash
DATA_ROOT=/data/yuzhang_fei/xuancheng_cityflow \
OUTPUT_DIR=/data/yuzhang_fei/xuancheng_cityflow/lane_agg_official_raw_direct_smoke \
DURATION=3600 \
BUCKET_SECONDS=60 \
bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_official_raw_lane_one_day.sh 2023-04-03
```

Lane-level output has shape `(time_buckets, 3546 lanes, 5 features)` and can be
aggregated to road-level later. Dense lane CSV is intentionally opt-in through
`WRITE_CSV=1`.

Render the roadnet or an aggregation result on OpenStreetMap:

```bash
python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/download_xuancheng_figshare.py \
  --data-root /data/yuzhang_fei/xuancheng_cityflow \
  --days 2023-04-03 \
  --include-sumo

python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/render_xuancheng_osm_map.py \
  --roadnet /data/yuzhang_fei/xuancheng_cityflow/raw/roadnet_xuancheng250319.json \
  --sumo-net /data/yuzhang_fei/xuancheng_cityflow/raw/xuancheng.net.xml \
  --npz /data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur3600.npz \
  --feature entered_veh \
  --time-agg sum \
  --output-html /data/yuzhang_fei/xuancheng_cityflow/maps/xuancheng_2023-04-03_paper_mp_entered_sum_osm.html
```

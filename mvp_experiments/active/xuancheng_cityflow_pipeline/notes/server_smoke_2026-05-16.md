# Server Smoke Run 2026-05-16

- Server: `ssh -p 5102 yuzhang_fei@183.174.228.180`.
- Repo: `/home/yuzhang_fei/code/SpatialTemporalGraph`.
- Code commit: `25ae270`.
- Data root: `/data/yuzhang_fei/xuancheng_cityflow`.
- Native `cityflow` was absent from system Python, `STGraph`, and `graph_ml`.
- Docker was installed, but Docker Hub timed out while pulling `kingsleycl/cityflow_env:latest`.
- Workaround: built official `cityflow-project/CityFlow` source with official `rapidjson` and `pybind11` submodules bundled locally, installed to `/data/yuzhang_fei/xuancheng_cityflow/pydeps`.
- Build detail: pip-installed `cmake==3.27.9` into the same `pydeps` directory; CMake 4.x rejected CityFlow's old `cmake_minimum_required(VERSION 3.0)`.

## R001 Output

- Command: `PYTHONPATH=/data/yuzhang_fei/xuancheng_cityflow/pydeps PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python DATA_ROOT=/data/yuzhang_fei/xuancheng_cityflow DURATION=1800 BUCKET_SECONDS=60 bash mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_server_xuancheng_one_day.sh 2023-04-03`.
- Config: `rlTrafficLight=false`, `laneChange=false`, 1-second CityFlow interval.
- Output CSV: `/data/yuzhang_fei/xuancheng_cityflow/road_agg/xuancheng_2023-04-03_road_agg_60s.csv.gz`, 684K.
- Output NPZ: `/data/yuzhang_fei/xuancheng_cityflow/road_agg/xuancheng_2023-04-03_road_agg_60s.npz`, 209K.
- CSV rows: `52321`, equal to one header plus `30 * 1744` dense road rows.
- NPZ shape: `(30, 1744, 5)`.
- Feature names: `entered_veh`, `exited_veh`, `mean_active_veh`, `mean_speed_kmh`, `density_veh_per_lane_km`.
- Smoke stats: `nonzero_entered=14253`, `total_entered=24682`, nonzero `mean_speed_kmh` mean approximately `34.16`.
- Caveat: CityFlow emitted many `Invalid route ... Omitted by default` warnings while loading the official flow file. This is consistent with the official test code being prepared to catch invalid routes, but R002 should quantify the omitted-flow rate before using full-day results as a benchmark.

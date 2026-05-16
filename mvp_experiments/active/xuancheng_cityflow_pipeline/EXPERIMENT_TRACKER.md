# Xuancheng CityFlow Pipeline Tracker

| Run ID | Command | Status | Artifact | Notes |
|---|---|---|---|---|
| ENV001 | `xuancheng_cityflow` conda env import/run check | DONE | `/data/yuzhang_fei/xuancheng_cityflow/road_agg_envcheck/xuancheng_2023-04-03_road_agg_60s.*` | Dedicated Python `/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python`; 60s smoke shape `(1, 1744, 5)`. |
| R001 | `run_server_xuancheng_one_day.sh 2023-04-03` with `DURATION=1800 BUCKET_SECONDS=60` | DONE | `/data/yuzhang_fei/xuancheng_cityflow/road_agg/xuancheng_2023-04-03_road_agg_60s.*` | Shape `(30, 1744, 5)`; CSV rows `52321`; `total_entered=24682`. |
| R002 | `run_server_xuancheng_one_day.sh 2023-04-03` with `DURATION=86400 BUCKET_SECONDS=60` | TODO | daily `csv.gz` and `.npz` | Full first day at 1-min resolution. |
| R003 | loop `2023-04-01..2023-04-07` | TODO | seven daily tensors | Pilot week. |
| R004 | `download_xuancheng_figshare.py --all-days` plus daily aggregation loop | TODO | monthly tensor files | Full official release. |
| R005 | `run_cityflow_paper_mp_road_aggregation.py --start-second 0 --duration 3600` | DONE | `/data/yuzhang_fei/xuancheng_cityflow/road_agg_paper_mp/xuancheng_2023-04-03_paper_mp_road_agg_60s_start0_dur3600.*` | Paper-style max-pressure control smoke/full-hour check; shape `(60, 1744, 5)`, 116 signal intersections, 360 decisions, `total_entered=42732`. |
| R006 | `run_cityflow_paper_mp_road_aggregation.py --start-second 61200 --duration 3600` | FAILED | `/data/yuzhang_fei/xuancheng_cityflow/logs/paper_mp_1700_1800_2023-04-03.log` | Strict paper window 17:00-18:00 failed before aggregation with CityFlow `router.cpp:84` assertion, even using the `.valid.json` route-filtered flow. |
| VIS001 | `render_xuancheng_osm_map.py --feature entered_veh --time-agg sum` | DONE | `/data/yuzhang_fei/xuancheng_cityflow/maps/xuancheng_2023-04-03_paper_mp_entered_sum_osm.html` | Renders all 1,744 roads on OpenStreetMap by converting SUMO local coordinates through `xuancheng.net.xml`; local copy in `outputs/maps/`. |

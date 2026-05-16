# Xuancheng CityFlow Pipeline Tracker

| Run ID | Command | Status | Artifact | Notes |
|---|---|---|---|---|
| ENV001 | `xuancheng_cityflow` conda env import/run check | DONE | `/data/yuzhang_fei/xuancheng_cityflow/road_agg_envcheck/xuancheng_2023-04-03_road_agg_60s.*` | Dedicated Python `/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python`; 60s smoke shape `(1, 1744, 5)`. |
| R001 | `run_server_xuancheng_one_day.sh 2023-04-03` with `DURATION=1800 BUCKET_SECONDS=60` | DONE | `/data/yuzhang_fei/xuancheng_cityflow/road_agg/xuancheng_2023-04-03_road_agg_60s.*` | Shape `(30, 1744, 5)`; CSV rows `52321`; `total_entered=24682`. |
| R002 | `run_server_xuancheng_one_day.sh 2023-04-03` with `DURATION=86400 BUCKET_SECONDS=60` | TODO | daily `csv.gz` and `.npz` | Full first day at 1-min resolution. |
| R003 | loop `2023-04-01..2023-04-07` | TODO | seven daily tensors | Pilot week. |
| R004 | `download_xuancheng_figshare.py --all-days` plus daily aggregation loop | TODO | monthly tensor files | Full official release. |

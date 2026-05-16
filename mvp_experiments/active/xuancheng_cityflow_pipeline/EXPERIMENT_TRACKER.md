# Xuancheng CityFlow Pipeline Tracker

| Run ID | Command | Status | Artifact | Notes |
|---|---|---|---|---|
| R001 | `run_server_xuancheng_one_day.sh 2023-04-03` with `DURATION=1800` | TODO | `/data/yuzhang_fei/xuancheng_cityflow/road_agg/` | Server smoke test. |
| R002 | `run_server_xuancheng_one_day.sh 2023-04-03` with `DURATION=86400 BUCKET_SECONDS=60` | TODO | daily `csv.gz` and `.npz` | Full first day at 1-min resolution. |
| R003 | loop `2023-04-01..2023-04-07` | TODO | seven daily tensors | Pilot week. |
| R004 | `download_xuancheng_figshare.py --all-days` plus daily aggregation loop | TODO | monthly tensor files | Full official release. |

# Experiment Tracker

| Run ID | Purpose | Command | Priority | Status | Notes |
|---|---|---|---|---|---|
| R000 | Check available Python envs | `bash zero_aware_mvp/scripts/check_env.sh` | MUST | DONE | Diagnostics env works; current default BasicTS training env is missing torch/easytorch/easydict. |
| R001 | Prepare 5min BasicTS dataset | `RESOLUTIONS=5min bash zero_aware_mvp/scripts/prepare_mvp_datasets.sh` | MUST | TODO | Must create `TRAFFIC_VOLUME_FULL_5MIN` from `/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz`. |
| R002 | Prepare aggregation datasets | `RESOLUTIONS="15min 30min 60min" bash zero_aware_mvp/scripts/prepare_mvp_datasets.sh` | MUST | TODO | Must create `TRAFFIC_VOLUME_FULL_15MIN`, `TRAFFIC_VOLUME_FULL_30MIN`, and `TRAFFIC_VOLUME_FULL_1H`. |
| R010 | Zero diagnostics | `bash zero_aware_mvp/scripts/run_00_diagnostics.sh` | MUST | TODO | Must run on full NPZ. Previous category-subgraph outputs are smoke-only. |
| R020 | Naive baselines | `bash zero_aware_mvp/scripts/run_01_naive_baselines.sh` | MUST | TODO | Must run on full BasicTS datasets. Previous category-subgraph outputs are smoke-only. |
| R030 | AGCRN smoke | `ZA_NUM_EPOCHS=3 ZA_MODEL=AGCRN bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh` | MUST | BLOCKED | Requires BasicTS env with torch/easytorch/easydict. |
| R031 | GWNet smoke | `ZA_NUM_EPOCHS=3 ZA_MODEL=GWNET bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh` | MUST | BLOCKED | Requires BasicTS env with torch/easytorch/easydict. |
| R040 | Post-hoc eval | `bash zero_aware_mvp/scripts/run_03_posthoc_eval.sh <ckpt_dir>` | MUST | TODO | Converts BasicTS predictions to zero-aware metrics. |
| R041 | Collect ranking | `bash zero_aware_mvp/scripts/run_04_collect_results.sh` | MUST | DONE | Built `results/summary/mvp_ranking.csv`. |
| R050 | Longer BasicTS runs | `ZA_NUM_EPOCHS=30 bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh` | NICE | TODO | Run after smoke passes. |
| R060 | Aggregation model repeats | `ZA_DATASET_NAME=TRAFFIC_VOLUME_15MIN ...` | NICE | TODO | Run only if diagnostics support the claim. |

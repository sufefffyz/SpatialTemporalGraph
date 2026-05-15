# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M1 | Discover server-side saved predictions | 12 PropMLP SD variants | SD test | found run count | MUST | DONE | Found under `BasicTS/checkpoints/PropMLP/SD_*/*/test_results` on server. |
| R002 | M2 | Frequency decomposition diagnostics | 12 PropMLP SD variants | SD test | full/low/high MAE, RMSE, WAPE | MUST | DONE | Output: `results/server_sd_propmlp_20260515_1650_named`. |
| R003 | M2 | Peak-window diagnostics | 12 PropMLP SD variants | SD test | peak/normal MAE and WAPE | MUST | DONE | Peak windows use per-node q=0.90 target threshold. |
| R004 | M3 | Spatial residual diagnostics | 12 PropMLP SD variants | SD h12 | edge corr, Dirichlet energy | NICE | DONE | Lightweight h12 run with 1000 sampled graph edges. |
| R005 | M1 | Discover old-SD baseline checkpoints | all old-SD checkpoint candidates | SD test | checkpoint count, metrics availability | MUST | DONE | Found 76 old-SD candidate best checkpoints, including non-PropMLP baselines. |
| R006 | M2 | Export saved predictions for model baselines | 10 non-PropMLP baselines | SD test | BasicTS `test_results` files | MUST | DONE | Exported STID, GraphWaveNet, DCRNN, ITransformer4D, Autoformer, Crossformer, FEDformer, PatchTST, STAEformer, STGCN. FlowNet load failed. |
| R007 | M2 | Frequency and peak diagnostics for model baselines | 10 non-PropMLP baselines | SD test | full/low/high MAE, peak MAE, W1 | MUST | DONE | Output: `results/server_sd_baselines_20260515_models_named`. |
| R008 | M3 | Spatial residual diagnostics for model baselines | 10 non-PropMLP baselines | SD h12 | edge corr, Dirichlet energy | NICE | DONE | Output: `results/server_sd_baselines_20260515_models_spatial_h12`. |

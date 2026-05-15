# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M1 | Discover server-side saved predictions | 12 PropMLP SD variants | SD test | found run count | MUST | DONE | Found under `BasicTS/checkpoints/PropMLP/SD_*/*/test_results` on server. |
| R002 | M2 | Frequency decomposition diagnostics | 12 PropMLP SD variants | SD test | full/low/high MAE, RMSE, WAPE | MUST | DONE | Output: `results/server_sd_propmlp_20260515_1650_named`. |
| R003 | M2 | Peak-window diagnostics | 12 PropMLP SD variants | SD test | peak/normal MAE and WAPE | MUST | DONE | Peak windows use per-node q=0.90 target threshold. |
| R004 | M3 | Spatial residual diagnostics | 12 PropMLP SD variants | SD h12 | edge corr, Dirichlet energy | NICE | DONE | Lightweight h12 run with 1000 sampled graph edges. |

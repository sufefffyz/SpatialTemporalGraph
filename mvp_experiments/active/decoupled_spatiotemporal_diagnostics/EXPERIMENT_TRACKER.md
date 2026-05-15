# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M1 | Discover server-side saved predictions | all found BasicTS runs | SD test | found run count | MUST | TODO | Search `BasicTS/checkpoints` and optional data roots. |
| R002 | M2 | Frequency decomposition diagnostics | all discovered runs | SD test | full/low/high MAE, RMSE, WAPE | MUST | TODO | Moving-average low-pass, default window 12. |
| R003 | M2 | Peak-window diagnostics | all discovered runs | SD test | peak/normal MAE and WAPE | MUST | TODO | Per-node target quantile threshold. |
| R004 | M3 | Spatial residual diagnostics | all discovered runs | SD test | edge corr, Dirichlet energy | NICE | TODO | Requires matching adjacency. |

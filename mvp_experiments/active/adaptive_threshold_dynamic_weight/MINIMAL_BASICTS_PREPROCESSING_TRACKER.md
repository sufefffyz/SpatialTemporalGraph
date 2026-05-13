# Minimal BasicTS LargeST-15min Preprocessing Tracker

| Run ID | Block | Purpose | BasicTS system / variant | Dataset | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| MBP001 | B0 | Validate node order, 15min splits, scaler, graph stats | diagnostics only | SD 15min | coverage, NaN rate, degree, isolated nodes | MUST | TODO | No training until this passes. |
| MBP002 | B0 | One-epoch training smoke | GWNet original graph, LargeST-15min protocol | SD 15min | finite loss, MAE sanity, sec/epoch | MUST | TODO | Confirms BasicTS path. |
| MBP003 | B1 | Baseline preprocessing | GWNet, LargeST-15min, train-split scaler, time features | SD 15min | MAE/RMSE/WAPE/per-horizon | MUST | TODO | Candidate default. |
| MBP004 | B1 | Time-feature confound | GWNet, LargeST-15min, train-split scaler, target-only | SD 15min | MAE/RMSE/WAPE/per-horizon | MUST | TODO | Required before dynamic weights. |
| MBP005 | B1 | Scaler confound | GWNet, LargeST-15min, node/channel-wise scaler, time features | SD 15min | MAE/RMSE/WAPE/per-horizon | MUST | TODO | Do not use leakage scaler. |
| MBP006 | B2 | Straight-line graph reference | GWNet, straight global threshold, beta 1.0 of LargeST-SD degree | SD 15min | MAE/RMSE/WAPE, edges, memory | MUST | TODO | Decides whether OSRM is necessary at matched original sparsity. |
| MBP007 | B2 | OSRM fixed threshold | GWNet, OSRM Gaussian global threshold, beta 1.0 of LargeST-SD degree | SD 15min | MAE/RMSE/WAPE, edges, memory | MUST | IN_PROGRESS | Fixed-threshold comparator; score quantile targets avg degree 24.1885. Launched as part of the 5-beta GWNet/DCRNN sweep on 2026-05-13. |
| MBP008 | B2 | Sparse heuristic reference | GWNet, OSRM top-k, matched beta 1.0 edge budget | SD 15min | MAE/RMSE/WAPE, edges, memory | MUST | TODO | Strong non-threshold baseline. |
| MBP009 | B2 | Minimal adaptive threshold | GWNet, OSRM local-density threshold, matched beta 1.0 edge budget | SD 15min | MAE/RMSE/WAPE, density quartile MAE | MUST | TODO | Core MVP signal. |
| MBP010 | B2 | Degree sensitivity, global threshold | GWNet, OSRM Gaussian global threshold, beta 0.25/0.5/1.5/2.0 | SD 15min | MAE/RMSE/WAPE, graph stats | NICE | IN_PROGRESS | Launched in the same 5-beta GWNet/DCRNN sweep on 2026-05-13; threshold chosen by Gaussian score quantile. |
| MBP011 | B2 | Degree sensitivity, top-k | GWNet, OSRM top-k, matched beta 0.25/0.5/1.5/2.0 | SD 15min | MAE/RMSE/WAPE, graph stats | NICE | TODO | Equal edge-budget comparator. |
| MBP012 | B2 | Degree sensitivity, local threshold | GWNet, OSRM local-density threshold, matched beta 0.25/0.5/1.5/2.0 | SD 15min | MAE/RMSE/WAPE, graph stats | NICE | TODO | Run only after global/top-k sweep is interpretable. |
| MBP013 | B3 | Static weight isolation | GWNet, best support, static learned edge weights | SD 15min | MAE/RMSE/WAPE, peak MAE | MUST | TODO | Same support as fixed baseline. |
| MBP014 | B3 | Dynamic weight isolation | GWNet, best support, dynamic edge weights | SD 15min | MAE/RMSE/WAPE, peak MAE | MUST | TODO | Compare with/without time features. |
| MBP015 | M4 | Seed confirmation | finalists only, seed 2023/2024/2025 | SD 15min | mean/std MAE/RMSE/WAPE | MUST | TODO | Only after positive one-seed signal. |
| MBP016 | B4 | Sparse efficiency proof | dense masked vs sparse edge-index/CSR | SD + GLA/GBA 15min | sec/epoch, throughput, memory | MUST | TODO | Required for efficiency claims. |
| MBP017 | B5 | Tiny PEMS replication | best graph variant vs reference | coordinate-validated PEMS | MAE/RMSE/WAPE | NICE | TODO | Run only after SD story is clean. |

# Server SD H1-H12 Test Results, 2026-05-15

## Scope

This pass recomputed BasicTS-style test metrics from the corrected old-SD baseline `test_results/predictions.npy` and `targets.npy` files.

- Dataset: `SD`
- Models: `STID`, `GraphWaveNet`, `DCRNN`, `ITransformer4D`, `Autoformer`, `Crossformer`, `FEDformer`, `PatchTST`, `STAEformer`, `STGCN`
- Metrics: `MAE`, `RMSE`, `MAPE`, `WAPE`
- Horizons: `H1` to `H12`
- Mask: `NULL_VAL=0.0`
- Batch size used for BasicTS-style aggregation: `64`

## Artifacts

Server output:

`mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_baselines_20260515_horizon_test_results`

Local copy:

`mvp_experiments/active/decoupled_spatiotemporal_diagnostics/outputs/server_sd_baselines_20260515_horizon_test_results`

Important files:

- `horizon_test_metrics.md`: full H1-H12 tables for MAE/RMSE/MAPE/WAPE.
- `horizon_test_metrics.csv`: long-form table.
- `horizon_mae_wide.csv`, `horizon_rmse_wide.csv`, `horizon_mape_wide.csv`, `horizon_wape_wide.csv`: wide tables.
- `mae_by_horizon.png`, `rmse_by_horizon.png`, `mape_by_horizon.png`, `wape_by_horizon.png`: line plots.
- `horizon_metric_heatmaps.png`: model-by-horizon heatmaps.
- `average_metric_bars.png`: average metric comparison bars.

## Average Across H1-H12

Lower is better.

| System | MAE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|
| STID | 17.8834 | 29.1351 | 0.1187 | 0.0943 |
| STAEformer | 18.6655 | 30.3472 | 0.1229 | 0.0967 |
| GraphWaveNet | 18.9227 | 30.1199 | 0.1264 | 0.1004 |
| DCRNN | 19.0212 | 30.0428 | 0.1256 | 0.0990 |
| STGCN | 21.4720 | 35.5951 | 0.1485 | 0.1163 |
| FEDformer | 21.8661 | 34.7845 | 0.1653 | 0.1284 |
| Crossformer | 21.9629 | 34.5469 | 0.1458 | 0.1160 |
| Autoformer | 28.1944 | 42.6223 | 0.2344 | 0.1881 |
| ITransformer4D | 32.2080 | 50.6379 | 0.2194 | 0.1763 |
| PatchTST | 47.5463 | 70.4623 | 0.3028 | 0.2638 |

## Sanity Check

The recomputed H3/H6/H12 values were checked against the BasicTS-generated `test_metrics.json` for representative models (`STID`, `DCRNN`, `GraphWaveNet`). Values match within small floating-point or batch aggregation differences.

## Initial Reading

- `STID` is the best by average MAE and MAPE.
- `STAEformer` is second by average MAE and has the second-best MAPE.
- `GraphWaveNet` and `DCRNN` remain close; DCRNN has slightly better RMSE than GraphWaveNet, while GraphWaveNet has slightly better MAE.
- `ITransformer4D` and `PatchTST` degrade sharply as horizon grows, especially after H6.

# Server SD Decoupled H1-H12 Diagnostics, 2026-05-16

## Scope

This pass addresses the concern that plain full-horizon test metrics can hide the actual failure mode. It recomputes the corrected old-SD baseline diagnostics across all horizons.

- Dataset: `SD`
- Models: `STID`, `GraphWaveNet`, `DCRNN`, `ITransformer4D`, `Autoformer`, `Crossformer`, `FEDformer`, `PatchTST`, `STAEformer`, `STGCN`
- Horizons: `H1` to `H12`
- Decomposition: centered moving average with window `12`; `full`, `low`, and `high` components.
- Peak windows: per-node `q=0.90` target threshold.
- Distribution metric: 1D Wasserstein distance (`W1`) between prediction and target distributions.
- Residual metrics: bias, residual std, mean absolute residual, under-prediction rate, temporal lag-1 residual correlation, node bias std, edge residual correlation, and residual Dirichlet energy.

The `Full MAE` here is the diagnostic script's finite-only raw reference, not the BasicTS `NULL_VAL=0` masked official test metric from R009.

## Artifacts

Server output:

`mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_baselines_20260516_decoupled_horizon_diagnostics`

Local copy:

`mvp_experiments/active/decoupled_spatiotemporal_diagnostics/outputs/server_sd_baselines_20260516_decoupled_horizon_diagnostics`

Important files:

- `decoupled_diagnostic_report.md`: average decoupled summary across H1-H12.
- `component_metrics.csv`: full/low/high MAE/RMSE/WAPE/W1 by horizon.
- `peak_window_metrics.csv`: peak/normal MAE/RMSE/WAPE/W1 by horizon.
- `residual_structure_metrics.csv`: residual bias, lag-1 correlation, under-prediction rate, node bias std.
- `spatial_residual_metrics.csv`: edge residual correlation and Dirichlet energy.
- `component_mae_heatmaps.png`, `component_w1_heatmaps.png`: component diagnostics.
- `peak_over_normal_mae_by_horizon.png`, `peak_w1_by_horizon.png`: peak diagnostics.
- `high_residual_structure_heatmaps.png`: high-component residual structure.

## Average Decoupled Diagnostics Across H1-H12

Lower is better for MAE, W1, edge corr, and Dirichlet. Peak/Normal measures relative peak-window difficulty.

| System | Full MAE | Low MAE | High MAE | High/Full | Full W1 | High W1 | Peak MAE | Peak/Normal | Peak W1 | High Lag1 | High Edge Corr | High Dirichlet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STID | 20.3541 | 14.0138 | 12.3450 | 0.6065 | 6.9632 | 3.1858 | 41.2459 | 2.3888 | 25.7077 | 0.3487 | 0.1916 | 672.7460 |
| DCRNN | 20.6286 | 14.3325 | 12.6468 | 0.6131 | 4.0739 | 3.0263 | 37.4464 | 2.0640 | 22.6754 | 0.3461 | 0.2105 | 656.0171 |
| STAEformer | 20.8651 | 14.8937 | 11.9373 | 0.5721 | 6.7329 | 3.9273 | 41.7106 | 2.3454 | 26.5169 | 0.3338 | 0.2040 | 617.0872 |
| GraphWaveNet | 21.3660 | 15.0251 | 12.7416 | 0.5963 | 6.8377 | 3.4446 | 42.6792 | 2.3430 | 25.8575 | 0.3430 | 0.2163 | 668.9234 |
| FEDformer | 23.5018 | 16.0153 | 14.1020 | 0.6000 | 2.9908 | 2.6141 | 40.0954 | 1.9049 | 24.2180 | 0.3848 | 0.3174 | 693.9356 |
| Crossformer | 24.4820 | 17.0839 | 14.2211 | 0.5809 | 5.5168 | 4.1215 | 49.0784 | 2.3543 | 29.0893 | 0.3712 | 0.2629 | 778.3193 |
| STGCN | 26.1517 | 19.7034 | 13.4246 | 0.5133 | 8.4845 | 4.4616 | 62.6845 | 3.0207 | 38.6588 | 0.3739 | 0.2620 | 687.5113 |
| Autoformer | 29.6312 | 19.2752 | 19.6606 | 0.6635 | 8.6172 | 4.1760 | 43.6165 | 1.5824 | 20.4482 | 0.1278 | 0.5270 | 832.8268 |
| ITransformer4D | 31.3038 | 18.8173 | 22.6683 | 0.7241 | 4.0295 | 5.6041 | 40.4105 | 1.3489 | 15.3170 | 0.4018 | 0.1904 | 2474.7183 |
| PatchTST | 46.0485 | 30.9167 | 29.6092 | 0.6430 | 8.2716 | 11.0073 | 59.7686 | 1.3577 | 37.1028 | 0.4535 | 0.2435 | 3897.2437 |

## Initial Reading

- `DCRNN` is strongest on peak-window absolute errors and distribution W1, even though `STID` remains competitive on average full error.
- `STAEformer` has the lowest high-component MAE, which supports the earlier impression that it handles high-frequency residual variation well.
- `ITransformer4D` has deceptively modest Full W1 and Peak W1, but its high-component MAE, high W1, lag-1 residual correlation, and Dirichlet energy are poor. This is a useful example where distribution-only or full-only metrics can mislead.
- `PatchTST` is consistently weak in high-frequency and residual-structure diagnostics; its errors grow sharply with horizon.
- `Autoformer` has a relatively low peak ratio but high full/low/high MAE and high edge residual correlation, so the ratio is not a quality ranking by itself.

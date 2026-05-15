# Server SD Model-Baseline Diagnostics, 2026-05-15

## Correction

The first diagnostic pass used PropMLP propagation variants. The intended target was old-SD model baselines such as STID, GraphWaveNet, DCRNN, and iTransformer. This note records the corrected pass.

## Checkpoint Discovery

- Server: `ssh -p 5102 yuzhang_fei@183.174.228.180`
- Repo: `/home/yuzhang_fei/code/SpatialTemporalGraph`
- Query: `BasicTS/checkpoints/*/*SD*/*/*best_val*.pt`
- Filter: old `SD` dataset only; excluded `SD_5min_full`, `SD_OSRMGG`, and `PEMSD*`.
- Found: 76 old-SD candidate best checkpoints.
- Already had saved predictions before this pass: mainly PropMLP variants.
- Needed export: most model baselines had `test_metrics.json` and checkpoints but no `test_results/predictions.npy`.

## Prediction Export

Export tag: `diagnostic_export_models_20260515`

Successfully exported predictions for:

- `STID`
- `GraphWaveNet`
- `DCRNN`
- `ITransformer4D`
- `Autoformer`
- `Crossformer`
- `FEDformer`
- `PatchTST`
- `STAEformer`
- `STGCN`

FlowNet was attempted but skipped from diagnostics. The checkpoint `BasicTSFlowNet_best_val_MAE.pt` failed strict loading against the current FlowNet code with missing keys:

```text
backbone.seq_est.linear.weight
backbone.seq_est.linear.bias
backbone.seq_est.linear.router.weight
backbone.seq_est.linear.router.bias
```

Do not include FlowNet in this diagnostic table unless the matching historical code/config is restored or a verified non-strict load policy is justified.

## Output Paths

- Export logs: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/eval_logs_20260515`
- Main diagnostics: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_baselines_20260515_models_named`
- Spatial h12 diagnostics: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_baselines_20260515_models_spatial_h12`

## H12 Component Ranking

Lower is better.

| System | Full MAE | Low MAE | High MAE | High / Full | Full W1 |
|---|---:|---:|---:|---:|---:|
| STID | 25.2409 | 20.0688 | 12.4121 | 0.4917 | 10.2365 |
| DCRNN | 25.7677 | 20.2808 | 13.0995 | 0.5084 | 6.1202 |
| STAEformer | 25.7786 | 20.7563 | 12.0153 | 0.4661 | 10.7209 |
| GraphWaveNet | 27.3651 | 22.0060 | 12.9611 | 0.4736 | 11.4202 |
| FEDformer | 30.0699 | 22.7366 | 15.4592 | 0.5141 | 6.3534 |
| STGCN | 32.7678 | 27.0949 | 13.6662 | 0.4171 | 14.2193 |
| Crossformer | 32.9834 | 26.8308 | 14.3738 | 0.4358 | 10.0551 |
| Autoformer | 36.9861 | 25.2668 | 23.9102 | 0.6465 | 10.1561 |
| ITransformer4D | 48.0148 | 33.8634 | 29.6189 | 0.6169 | 5.9508 |
| PatchTST | 78.5366 | 62.3170 | 39.2128 | 0.4993 | 18.7566 |

## Average Across Horizons

| System | Full MAE | Low MAE | High MAE | High / Full |
|---|---:|---:|---:|---:|
| STID | 19.0233 | 12.3224 | 12.2671 | 0.6448 |
| DCRNN | 19.3945 | 12.8519 | 12.5475 | 0.6470 |
| STAEformer | 19.5032 | 13.1686 | 11.9711 | 0.6138 |
| GraphWaveNet | 19.8148 | 13.1248 | 12.6362 | 0.6377 |
| FEDformer | 22.1796 | 14.7429 | 13.7360 | 0.6193 |
| Crossformer | 22.5009 | 14.7771 | 14.0489 | 0.6244 |
| STGCN | 24.8676 | 18.3052 | 13.3646 | 0.5374 |
| ITransformer4D | 28.0432 | 16.1360 | 20.9811 | 0.7482 |
| Autoformer | 28.7493 | 18.4781 | 19.6152 | 0.6823 |
| PatchTST | 40.4173 | 26.2925 | 26.7443 | 0.6617 |

## H12 Peak-Window Ranking

Peak windows use per-node `q=0.90` target thresholds.

| System | Normal MAE | Peak MAE | Peak / Normal | Normal W1 | Peak W1 |
|---|---:|---:|---:|---:|---:|
| DCRNN | 22.5717 | 47.3916 | 2.0996 | 5.0645 | 29.0966 |
| FEDformer | 27.0648 | 50.4021 | 1.8623 | 7.1078 | 30.8029 |
| STAEformer | 22.0609 | 50.9329 | 2.3087 | 9.6898 | 30.6971 |
| STID | 21.4193 | 51.0977 | 2.3856 | 8.4123 | 33.1027 |
| Autoformer | 34.4429 | 54.1937 | 1.5734 | 10.5172 | 27.3988 |
| GraphWaveNet | 23.0859 | 56.3181 | 2.4395 | 9.3368 | 34.3346 |
| ITransformer4D | 46.2269 | 60.1120 | 1.3004 | 10.0534 | 30.1444 |
| Crossformer | 27.0063 | 73.4244 | 2.7188 | 7.1524 | 38.1081 |
| STGCN | 26.1972 | 77.2248 | 2.9478 | 9.6517 | 46.9066 |
| PatchTST | 74.2607 | 107.4676 | 1.4472 | 15.5730 | 88.6182 |

## H12 High-Residual Spatial Metrics

Lightweight diagnostic with 1000 sampled graph edges from `BasicTS/datasets/SD/adj_mx.pkl`.

| System | Edge Residual Corr | Dirichlet | Sampled Edges |
|---|---:|---:|---:|
| STID | 0.1993 | 697.0058 | 1000 |
| ITransformer4D | 0.2059 | 4087.6916 | 1000 |
| STAEformer | 0.2277 | 606.7275 | 1000 |
| DCRNN | 0.2301 | 682.6593 | 1000 |
| PatchTST | 0.2455 | 6404.8335 | 1000 |
| GraphWaveNet | 0.2532 | 655.8267 | 1000 |
| STGCN | 0.2829 | 685.0266 | 1000 |
| Crossformer | 0.2964 | 731.8813 | 1000 |
| FEDformer | 0.3791 | 731.2675 | 1000 |
| Autoformer | 0.6425 | 976.7257 | 1000 |

## Initial Interpretation

- The corrected old-SD model-baseline ranking is led by `STID`, `DCRNN`, `STAEformer`, and `GraphWaveNet`.
- `STID` has the best average MAE, while `DCRNN` has the best h12 peak-window MAE and the lowest h12 full-distribution W1 among the top models.
- `STAEformer` has the lowest average high-component MAE, which suggests it handles residual/high-frequency variation relatively well.
- `ITransformer4D` and `PatchTST` are weak on this old-SD setup despite being modern temporal baselines; their high-component errors are especially large.
- Peak windows are substantially harder. For the top models, h12 peak MAE is roughly `2.1x` to `2.4x` normal MAE.
- Spatial residual correlation alone is not a forecast-quality ranking. `STID` has the lowest sampled edge residual correlation, while `DCRNN` and `GraphWaveNet` remain competitive in both forecast error and residual structure.

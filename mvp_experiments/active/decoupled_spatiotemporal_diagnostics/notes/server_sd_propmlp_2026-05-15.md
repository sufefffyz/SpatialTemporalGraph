# Server SD PropMLP Diagnostics, 2026-05-15

## Run Context

- Server: `ssh -p 5102 yuzhang_fei@183.174.228.180`
- Repo: `/home/yuzhang_fei/code/SpatialTemporalGraph`
- Code sync: local commit and push, then server `git pull --ff-only origin 0.5.8`
- Commits:
  - `def9ef3` adds the decoupled diagnostics MVP.
  - `c7284ea` improves automatic run naming.
- Dataset: `BasicTS/datasets/SD`
- Saved predictions: `BasicTS/checkpoints/PropMLP/SD_*/*/test_results`
- Main output: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_propmlp_20260515_1650_named`
- Spatial h12 output: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_propmlp_20260515_1650_spatial_h12`

## Main H12 Component Ranking

Lower is better.

| Variant | Full MAE | Low MAE | High MAE | High / Full | Full W1 |
|---|---:|---:|---:|---:|---:|
| distthre_x_pfpb_k2 | 58.9141 | 46.5519 | 28.4406 | 0.4827 | 20.4983 |
| distthre_x_pfpb_k3 | 59.2642 | 46.9018 | 28.3297 | 0.4780 | 19.8199 |
| distthre_x_pfpb_k1 | 59.8297 | 46.7013 | 29.2910 | 0.4896 | 15.3828 |
| physical_dir_x_pfpb_k3 | 62.1744 | 49.0487 | 29.4619 | 0.4739 | 14.0962 |
| physical_dir_x_pfpb_k2 | 64.1203 | 50.6279 | 30.4474 | 0.4748 | 14.4885 |
| physical_dir_x_pfpb_k1 | 64.3375 | 50.5279 | 31.1948 | 0.4849 | 15.2655 |
| distthre_pfpb_k3 | 69.1585 | 59.4574 | 26.8520 | 0.3883 | 24.4276 |
| distthre_pfpb_k2 | 71.2624 | 61.3940 | 27.3095 | 0.3832 | 22.8214 |
| distthre_pfpb_k1 | 78.0414 | 68.6441 | 28.1499 | 0.3607 | 27.9627 |
| physical_dir_pfpb_k3 | 88.6884 | 79.5407 | 26.3745 | 0.2974 | 41.5851 |
| physical_dir_pfpb_k2 | 90.4847 | 81.3873 | 26.6624 | 0.2947 | 43.2910 |
| physical_dir_pfpb_k1 | 91.9189 | 82.2473 | 28.2053 | 0.3068 | 43.1050 |

## Average Across Horizons

| Variant | Full MAE | Low MAE | High MAE | High / Full |
|---|---:|---:|---:|---:|
| distthre_x_pfpb_k2 | 33.0691 | 21.0325 | 21.9489 | 0.6637 |
| distthre_x_pfpb_k3 | 33.2223 | 21.1742 | 21.8638 | 0.6581 |
| distthre_x_pfpb_k1 | 33.5588 | 20.9969 | 22.5401 | 0.6717 |
| physical_dir_x_pfpb_k3 | 34.7955 | 22.1249 | 22.8241 | 0.6560 |
| physical_dir_x_pfpb_k2 | 35.4181 | 22.6399 | 23.2051 | 0.6552 |
| physical_dir_x_pfpb_k1 | 35.6330 | 22.7553 | 23.5747 | 0.6616 |
| distthre_pfpb_k3 | 51.1446 | 43.8286 | 21.2033 | 0.4146 |
| distthre_pfpb_k2 | 54.3172 | 47.1323 | 21.3510 | 0.3931 |
| distthre_pfpb_k1 | 62.7089 | 56.4088 | 21.5600 | 0.3438 |
| physical_dir_pfpb_k3 | 74.1871 | 67.5729 | 21.2095 | 0.2859 |
| physical_dir_pfpb_k2 | 75.6492 | 69.0677 | 21.3626 | 0.2824 |
| physical_dir_pfpb_k1 | 76.8061 | 69.9821 | 22.0697 | 0.2873 |

## Peak Windows at H12

Peak windows use per-node `q=0.90` target thresholds. Lower is better.

| Variant | Normal MAE | Peak MAE | Peak / Normal | Normal W1 | Peak W1 |
|---|---:|---:|---:|---:|---:|
| distthre_x_pfpb_k1 | 54.3069 | 97.1932 | 1.7897 | 10.4736 | 81.4492 |
| distthre_x_pfpb_k3 | 53.4695 | 98.4683 | 1.8416 | 13.1601 | 81.4827 |
| distthre_x_pfpb_k2 | 52.9923 | 98.9775 | 1.8678 | 13.3837 | 84.8728 |
| physical_dir_x_pfpb_k2 | 58.4356 | 102.5804 | 1.7554 | 10.9316 | 87.7473 |
| physical_dir_x_pfpb_k1 | 58.6762 | 102.6380 | 1.7492 | 9.0436 | 89.8144 |
| physical_dir_x_pfpb_k3 | 55.8266 | 105.1204 | 1.8830 | 8.4693 | 89.1436 |
| distthre_pfpb_k3 | 59.4251 | 135.0088 | 2.2719 | 12.7922 | 110.4919 |
| distthre_pfpb_k2 | 61.3431 | 138.3704 | 2.2557 | 12.0522 | 110.4650 |
| distthre_pfpb_k1 | 66.3376 | 157.2216 | 2.3700 | 16.1511 | 117.6499 |
| physical_dir_pfpb_k3 | 75.9057 | 175.1676 | 2.3077 | 28.6632 | 129.0065 |
| physical_dir_pfpb_k2 | 77.5691 | 177.8642 | 2.2930 | 30.5298 | 129.6253 |
| physical_dir_pfpb_k1 | 79.1969 | 177.9883 | 2.2474 | 29.9638 | 132.0109 |

## Spatial H12 High-Residual Metrics

Lightweight diagnostic with 1000 sampled graph edges from `BasicTS/datasets/SD/adj_mx.pkl`.

| Variant | Edge Residual Corr | Dirichlet | Sampled Edges |
|---|---:|---:|---:|
| physical_dir_x_pfpb_k1 | 0.3077 | 2685.4086 | 1000 |
| physical_dir_pfpb_k1 | 0.3250 | 2097.3472 | 1000 |
| physical_dir_x_pfpb_k2 | 0.3339 | 2466.7839 | 1000 |
| physical_dir_x_pfpb_k3 | 0.3365 | 2344.2454 | 1000 |
| physical_dir_pfpb_k2 | 0.3436 | 1844.5766 | 1000 |
| physical_dir_pfpb_k3 | 0.3639 | 1778.0025 | 1000 |
| distthre_x_pfpb_k1 | 0.4990 | 1630.3345 | 1000 |
| distthre_x_pfpb_k3 | 0.5149 | 1540.5407 | 1000 |
| distthre_x_pfpb_k2 | 0.5177 | 1515.9044 | 1000 |
| distthre_pfpb_k2 | 0.6011 | 1119.5558 | 1000 |
| distthre_pfpb_k3 | 0.6085 | 1069.0608 | 1000 |
| distthre_pfpb_k1 | 0.6212 | 1095.5023 | 1000 |

## Initial Interpretation

- Adding raw input `x` is the dominant difference: `x_pfpb` variants roughly halve average full MAE compared with `pfpb`.
- The largest model gap is low-frequency/trend error. The `pfpb` variants have similar or even lower high-component MAE, but much worse low-component MAE.
- Peak windows are much harder than normal windows. At h12, peak MAE is about `1.75x` to `2.37x` normal MAE.
- `distthre_x_pfpb` is strongest on full MAE, while `physical_dir_x_pfpb` can have lower h12 distribution W1 in some cases. This suggests distribution-shape diagnostics may not always agree with MAE ranking.
- Spatial residual correlation is lower for physical-dir variants, but their absolute errors are often higher. Treat this as a residual-structure signal, not as evidence of better forecasting by itself.

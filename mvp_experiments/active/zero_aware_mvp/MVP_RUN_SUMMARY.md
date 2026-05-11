# MVP Run Summary

Date: 2026-05-07

## Status Correction

The completed results below were produced with the local category-filtered file
`data/city_traffic_m_volume__category__1_0.npz`. They are useful only as code
smoke checks. They should not be used as paper-facing evidence.

Paper-facing runs must use:

```text
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz
```

## Completed Smoke Checks

- Created the zero-aware MVP experiment scaffold under `zero_aware_mvp/`.
- Prepared BasicTS datasets:
  - `TRAFFIC_VOLUME_5MIN`
  - `TRAFFIC_VOLUME_15MIN`
  - `TRAFFIC_VOLUME_30MIN`
  - `TRAFFIC_VOLUME_1H`
- Ran full zero diagnostics on the raw city-traffic-M category=1.0 volume subgraph.
- Ran naive baselines on `5MIN`, `15MIN`, `30MIN`, and `1H`.
- Collected all current rows into `results/summary/mvp_ranking.csv`.

## Key Early Findings

### Zero Diagnostics

On the 5min target data:

- overall zero rate: `0.2281`
- variance / mean: `14.33`
- positive q90: `26`
- positive q95: `34`
- max: `147`

Aggregation sensitivity:

| Resolution | Zero Rate |
|---|---:|
| 5min | 0.2281 |
| 15min | 0.1381 |
| 30min | 0.1062 |
| 60min | 0.0862 |

Expected-zero checks on the test split:

| Model | Expected Zero Rate | Observed - Expected |
|---|---:|---:|
| road-hour Poisson | 0.2167 | 0.0179 |
| road-level NB | 0.1865 | 0.0480 |

Transition dynamics:

- `P(0 -> 0) = 0.7225`
- `P(0 -> +) = 0.2775`
- `P(+ -> 0) = 0.0820`
- `P(+ -> +) = 0.9180`

### Naive Ranking Reversal

Best MAE vs best road-q90 high-flow F1:

| Dataset | Best MAE System | MAE | Its High-F1 | Best High-F1 System | Best High-F1 |
|---|---|---:|---:|---|---:|
| `TRAFFIC_VOLUME_5MIN` | slot_median | 2.5685 | 0.0303 | previous_step | 0.3584 |
| `TRAFFIC_VOLUME_15MIN` | slot_median | 5.7381 | 0.0690 | seasonal_day_ago | 0.4407 |
| `TRAFFIC_VOLUME_30MIN` | slot_median | 10.1038 | 0.1008 | seasonal_day_ago | 0.5032 |
| `TRAFFIC_VOLUME_1H` | slot_median | 18.3480 | 0.1229 | seasonal_day_ago | 0.5618 |

This is already a useful MVP signal: the best low-error seasonal median baseline
does not capture road-normalized high-flow events well.

## Blocked

BasicTS neural smoke runs are not launched yet because the currently available
default training Python is missing:

- `torch`
- `easydict`
- `easytorch`

Use an environment with those packages, then run:

```bash
cd /Users/richardo/Desktop/STproject/SpatialTemporalGraph
export BASICTS_PYTHON=python
ZA_MODEL=AGCRN ZA_NUM_EPOCHS=3 bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh
ZA_MODEL=GWNET ZA_NUM_EPOCHS=3 bash zero_aware_mvp/scripts/run_02_basicts_smoke.sh
```

Then evaluate:

```bash
bash zero_aware_mvp/scripts/run_03_posthoc_eval.sh <BasicTS checkpoint dir>
bash zero_aware_mvp/scripts/run_04_collect_results.sh
```

# Zero-Aware MVP Experiments

This folder contains the MVP experiment scaffold for:

> Beyond MAE: zero-aware fine-grained urban traffic forecasting.

The goal is to validate the idea in the cheapest useful order before spending
GPU time. BasicTS is used for dataset preparation and neural forecasting runs;
the scripts here add zero-aware diagnostics, naive baselines, and post-hoc
metrics over BasicTS prediction files.

By default, the MVP uses the full Urban Traffic Benchmark file:

```text
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz
```

Do not use `data/city_traffic_m_volume__category__1_0.npz` for paper-facing
evidence. That file is a local category-filtered smoke dataset and is only
acceptable for quick code checks.

## Run Order

The order below is sorted by both paper importance and expected turnaround time.

| Order | Stage | Why first | Cost |
|---|---|---|---|
| 0 | Dependency check | Avoids wasting runs in the wrong Python env. | seconds |
| 1 | Prepare BasicTS datasets | Creates 5/15/30/60min bundles used by all later runs. | minutes, disk-heavy |
| 2 | Zero diagnostics | Tests whether the data really has structured zeros. | minutes |
| 3 | Naive baselines | Tests whether MAE and zero-aware metrics disagree before GPUs. | minutes |
| 4 | BasicTS smoke runs | Confirms AGCRN/GWNet can train and save predictions. | low GPU |
| 5 | Post-hoc zero-aware eval | Converts BasicTS outputs into benchmark evidence. | minutes |
| 6 | Longer BasicTS runs | Only after stages 2-5 look promising. | GPU hours |
| 7 | Probe thinning / extra ablations | Strong but not needed before the first go/no-go. | extra |

## Quick Start

From the `SpatialTemporalGraph` repository root:

```bash
cd /Users/richardo/Desktop/STproject/SpatialTemporalGraph
```

Use a Python with `numpy` and `pandas` for data preparation and diagnostics:

```bash
export PYTHON_BIN=/Users/richardo/Desktop/STproject/SpatialTemporalGraph/.conda/causalrivers-macos/bin/python
```

Prepare BasicTS datasets from the full dataset:

```bash
bash mvp_experiments/active/zero_aware_mvp/scripts/prepare_mvp_datasets.sh
```

If the full dataset is mounted elsewhere:

```bash
INPUT_NPZ=/absolute/path/to/city_traffic_m_volume.npz \
bash mvp_experiments/active/zero_aware_mvp/scripts/prepare_mvp_datasets.sh
```

For the full city-M graph, keep the default `ADJ_MODE=none`. A dense adjacency
matrix for 53k nodes is too large for the low-cost MVP and is not needed by the
diagnostics or naive baselines. If disk under the repo is tight on the server,
write datasets under `/data` and point the evaluators there:

```bash
OUTPUT_ROOT=/data/yuzhang_fei/zero_aware_mvp_datasets \
BASICTS_DATASETS_ROOT=/data/yuzhang_fei/zero_aware_mvp_datasets \
bash mvp_experiments/active/zero_aware_mvp/scripts/prepare_mvp_datasets.sh
```

Run the cheapest must-pass checks:

```bash
bash mvp_experiments/active/zero_aware_mvp/scripts/run_00_diagnostics.sh
bash mvp_experiments/active/zero_aware_mvp/scripts/run_01_naive_baselines.sh
```

Run short BasicTS smoke experiments once a BasicTS environment with `torch`,
`easytorch`, and `easydict` is active:

```bash
export BASICTS_PYTHON=python
export ZA_NUM_EPOCHS=3
bash mvp_experiments/active/zero_aware_mvp/scripts/run_02_basicts_smoke.sh
```

Evaluate saved BasicTS predictions:

```bash
bash mvp_experiments/active/zero_aware_mvp/scripts/run_03_posthoc_eval.sh checkpoints/zero_aware_mvp/GWNet/TRAFFIC_VOLUME_FULL_5MIN_3_12_12_seed42
```

If the exact checkpoint folder has an EasyTorch hash suffix, pass that concrete
directory instead.

## Main Outputs

- `mvp_experiments/active/zero_aware_mvp/results/diagnostics/zero_diagnostics_*.json`
- `mvp_experiments/active/zero_aware_mvp/results/naive_baselines/naive_metrics_*.json`
- `mvp_experiments/active/zero_aware_mvp/results/posthoc/basicts_zero_metrics_*.json`
- `mvp_experiments/active/zero_aware_mvp/results/summary/mvp_ranking.csv`

Full-dataset BasicTS bundles are named:

- `TRAFFIC_VOLUME_FULL_5MIN`
- `TRAFFIC_VOLUME_FULL_15MIN`
- `TRAFFIC_VOLUME_FULL_30MIN`
- `TRAFFIC_VOLUME_FULL_1H`

## Important Metric Semantics

BasicTS traffic configs often use `NULL_VAL=0.0`, which masks zeros in MAE/RMSE.
That is not acceptable for this benchmark because zero is the object of study.
The MVP BasicTS configs override `cfg.METRICS.NULL_VAL = np.nan`, so zeros are
included in full-sample point metrics.

The zero-aware metrics are computed post-hoc because PR-style and high-flow
metrics need global test-set aggregation rather than per-batch averaging.

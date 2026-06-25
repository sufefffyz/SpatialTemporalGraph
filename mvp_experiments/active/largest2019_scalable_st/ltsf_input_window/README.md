# LargeST-LTSF Input Window Probe

This folder contains isolated tooling for testing true LTSF-style settings on
LargeST 2019 without modifying baseline config files.

Default protocol:

- First probe forecast horizon: `H=12`.
- Long-horizon follow-up: `H=672` only after the `H=12` input-window probe is useful.
- Input windows: `L in {96, 192, 336, 672}`.
- First wave: SD for all selected models; GBA for scalable models after smoke.
- Dynamic-threshold work is out of scope.

## Files

- `scripts/estimate_ltsf_resources.py`: creates complexity and tensor-memory estimates.
- `scripts/prepare_largest_ltsf_npz.py`: creates BiST/LargeST `his.npz` caches under a non-default tag such as `2019_L96_H12`.
- `scripts/generate_basicts_ltsf_configs.py`: writes isolated BasicTS configs for STID/DLinear/CycleNet/TimeMixer/PatchTST.
- `scripts/run_basicts_ltsf_grid.sh`: runs generated BasicTS configs with wall-time and GPU-memory sampling.
- `scripts/run_bist_ltsf_grid.sh`: runs BiST LTSF jobs after the matching `his.npz` cache exists.
- `scripts/collect_ltsf_results.py`: summarizes BasicTS metrics and BiST log rows into one CSV.
- `scripts/plot_input_window_curve.py`: plots MAE/RMSE curves versus input length.

## Quick Start

From `SpatialTemporalGraph`:

```bash
python mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/scripts/estimate_ltsf_resources.py

PHASE=smoke \
DATASETS=SD \
MODELS=STID,DLinear,CycleNet,TimeMixer,PatchTST \
INPUT_LENGTHS=96,192,336,672 \
HORIZON=12 \
GPU=0 \
bash mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/scripts/run_basicts_ltsf_grid.sh
```

For BiST, prepare the cache first:

```bash
python mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/scripts/prepare_largest_ltsf_npz.py \
  --dataset SD --input-len 96 --horizon 12
```

Then launch:

```bash
DATASETS=SD INPUT_LENGTHS=96,192,336,672 HORIZON=12 GPU=0 \
bash mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/scripts/run_bist_ltsf_grid.sh
```

Generated configs, logs, metrics, and plots are written under `outputs/`.

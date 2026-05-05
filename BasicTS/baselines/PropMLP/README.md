# PropMLP Results on Old SD

This note summarizes the PropMLP experiments found under:

```text
BasicTS/checkpoints/PropMLP/
```

The results were collected from each run's `test_metrics.json` on the server on 2026-05-05.

## Setup

- Dataset: `SD`
- Input length: `12`
- Output length: `12`
- Model: `PropMLP`
- Epoch budget: `100`
- Hidden dimension: `128`
- Dropout: `0.1`
- Seed: `42`
- Metrics below are `overall` test metrics unless otherwise noted.
- `x_only` ignores graph propagation and is graph-agnostic.
- `pfpb` uses only propagated forward/backward graph features.
- `x_pfpb` concatenates the original signal with propagated forward/backward graph features.

## Main Takeaways

1. Best overall run:

   ```text
   distthre + x_pfpb + K=4 + no self-loop
   MAE=37.3438, RMSE=60.3250, MAPE=0.2680, WAPE=0.2410
   ```

2. Adding graph-propagated features to the original signal is clearly useful. The `x_only` baseline has `MAE=42.2939`, while the best `x_pfpb` runs are around `37.34-37.59`.

3. Pure propagated features (`pfpb`) are much weaker than `x_pfpb`, but self-loops help them a lot. For example, `physical_dir + pfpb + K=5` improves from `MAE=68.9791` without self-loop to `MAE=43.7565` with self-loop.

4. `distthre` is currently stronger than `physical_dir` for the final predictive model when using `x_pfpb`. The best `physical_dir + x_pfpb` result is `MAE=38.8375`, while the best `distthre + x_pfpb` result is `MAE=37.3438`.

5. For `x_pfpb`, larger propagation order helps up to around `K=4/K=5`; the gain is much larger from `K=1` to `K=4` than from `K=4` to `K=5`.

## Best Runs

| Rank | Graph | Mode | K | Self-loop | MAE | RMSE | MAPE | WAPE |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | distthre | x_pfpb | 4 | no | 37.3438 | 60.3250 | 0.2680 | 0.2410 |
| 2 | distthre | x_pfpb | 5 | no | 37.3500 | 59.8173 | 0.2735 | 0.2479 |
| 3 | distthre | x_pfpb | 4 | yes | 37.4299 | 60.5163 | 0.2653 | 0.2417 |
| 4 | distthre | x_pfpb | 5 | yes | 37.5911 | 60.2105 | 0.2762 | 0.2476 |
| 5 | distthre | x_pfpb | 3 | yes | 37.7298 | 60.1568 | 0.2905 | 0.2608 |
| 6 | distthre | x_pfpb | 2 | no | 37.9624 | 60.0858 | 0.2972 | 0.2670 |
| 7 | distthre | x_pfpb | 2 | yes | 38.0447 | 60.3398 | 0.2948 | 0.2638 |
| 8 | distthre | x_pfpb | 3 | no | 38.1105 | 60.5381 | 0.2973 | 0.2696 |
| 9 | distthre | x_pfpb | 1 | no | 38.6227 | 61.4954 | 0.2903 | 0.2568 |
| 10 | distthre | x_pfpb | 1 | yes | 38.6803 | 61.6319 | 0.2887 | 0.2552 |
| 11 | physical_dir | x_pfpb | 5 | no | 38.8375 | 62.2023 | 0.3028 | 0.2684 |
| 12 | physical_dir | x_pfpb | 4 | no | 39.2206 | 62.9108 | 0.3030 | 0.2703 |
| 13 | physical_dir | x_pfpb | 4 | yes | 39.4446 | 63.3291 | 0.3085 | 0.2712 |
| 14 | physical_dir | x_pfpb | 5 | yes | 39.5011 | 63.5520 | 0.3163 | 0.2784 |
| 15 | physical_dir | x_pfpb | 3 | no | 39.7135 | 63.5103 | 0.2979 | 0.2613 |

## Baseline

| Graph | Mode | K | Self-loop | MAE | RMSE | MAPE | WAPE | H3 MAE | H6 MAE | H12 MAE |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| distthre | x_only | 1 | no | 42.2939 | 67.3608 | 0.3370 | 0.2888 | 25.1931 | 40.4161 | 67.8622 |

## `x_pfpb` Results

| Graph | K | Self-loop | MAE | RMSE | MAPE | WAPE | H3 MAE | H6 MAE | H12 MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| distthre | 1 | no | 38.6227 | 61.4954 | 0.2903 | 0.2568 | 23.7549 | 36.9897 | 60.9931 |
| distthre | 2 | no | 37.9624 | 60.0858 | 0.2972 | 0.2670 | 23.4177 | 36.3644 | 59.8561 |
| distthre | 3 | no | 38.1105 | 60.5381 | 0.2973 | 0.2696 | 23.4682 | 36.5015 | 60.1756 |
| distthre | 4 | no | 37.3438 | 60.3250 | 0.2680 | 0.2410 | 23.2668 | 35.8993 | 58.3463 |
| distthre | 5 | no | 37.3500 | 59.8173 | 0.2735 | 0.2479 | 23.3139 | 35.8550 | 58.1750 |
| distthre | 1 | yes | 38.6803 | 61.6319 | 0.2887 | 0.2552 | 23.8073 | 37.0510 | 61.0746 |
| distthre | 2 | yes | 38.0447 | 60.3398 | 0.2948 | 0.2638 | 23.5086 | 36.4770 | 59.9345 |
| distthre | 3 | yes | 37.7298 | 60.1568 | 0.2905 | 0.2608 | 23.3733 | 36.1258 | 59.2574 |
| distthre | 4 | yes | 37.4299 | 60.5163 | 0.2653 | 0.2417 | 23.2386 | 35.9422 | 58.6803 |
| distthre | 5 | yes | 37.5911 | 60.2105 | 0.2762 | 0.2476 | 24.0620 | 35.9910 | 58.4525 |
| physical_dir | 1 | no | 41.1449 | 65.7564 | 0.3083 | 0.2665 | 24.7804 | 39.4205 | 65.6680 |
| physical_dir | 2 | no | 40.6024 | 64.8826 | 0.3336 | 0.2910 | 24.4649 | 38.7840 | 64.8759 |
| physical_dir | 3 | no | 39.7135 | 63.5103 | 0.2979 | 0.2613 | 24.3024 | 38.1665 | 62.7502 |
| physical_dir | 4 | no | 39.2206 | 62.9108 | 0.3030 | 0.2703 | 23.9992 | 37.6059 | 61.9480 |
| physical_dir | 5 | no | 38.8375 | 62.2023 | 0.3028 | 0.2684 | 23.9490 | 37.2077 | 61.2858 |
| physical_dir | 1 | yes | 40.9458 | 65.2088 | 0.3036 | 0.2642 | 24.6805 | 39.1872 | 65.2091 |
| physical_dir | 2 | yes | 40.9517 | 65.5965 | 0.3466 | 0.3044 | 24.6223 | 39.3014 | 65.1504 |
| physical_dir | 3 | yes | 40.2490 | 63.6084 | 0.3381 | 0.2916 | 24.9172 | 38.5263 | 63.2248 |
| physical_dir | 4 | yes | 39.4446 | 63.3291 | 0.3085 | 0.2712 | 24.0294 | 37.8037 | 62.4672 |
| physical_dir | 5 | yes | 39.5011 | 63.5520 | 0.3163 | 0.2784 | 24.1177 | 37.8530 | 62.5057 |

## `pfpb` Results

| Graph | K | Self-loop | MAE | RMSE | MAPE | WAPE | H3 MAE | H6 MAE | H12 MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| distthre | 1 | no | 60.9045 | 88.4952 | 0.5013 | 0.4487 | 52.6366 | 58.8818 | 74.7609 |
| distthre | 2 | no | 54.2800 | 80.5682 | 0.4117 | 0.3716 | 44.7984 | 52.3633 | 69.4263 |
| distthre | 3 | no | 52.0029 | 77.9005 | 0.3986 | 0.3572 | 41.9941 | 50.1705 | 67.8759 |
| distthre | 4 | no | 50.1667 | 75.0435 | 0.4064 | 0.3601 | 40.1914 | 48.3489 | 65.8536 |
| distthre | 5 | no | 49.7227 | 74.9285 | 0.3696 | 0.3326 | 39.7225 | 47.9496 | 65.4804 |
| distthre | 1 | yes | 56.7366 | 83.0622 | 0.4601 | 0.4107 | 47.3361 | 54.6122 | 72.1742 |
| distthre | 2 | yes | 47.3278 | 71.5075 | 0.3472 | 0.3133 | 35.9734 | 45.4749 | 64.9766 |
| distthre | 3 | yes | 43.8513 | 67.5539 | 0.3261 | 0.2907 | 31.4241 | 42.0742 | 62.8665 |
| distthre | 4 | yes | 42.1105 | 65.4429 | 0.3086 | 0.2801 | 29.5375 | 40.4362 | 61.2628 |
| distthre | 5 | yes | 41.8674 | 64.9560 | 0.3085 | 0.2807 | 29.4513 | 40.1719 | 60.7212 |
| physical_dir | 1 | no | 74.2041 | 109.7298 | 0.8317 | 0.6996 | 65.7420 | 72.4305 | 87.8815 |
| physical_dir | 2 | no | 72.6776 | 108.0681 | 0.8029 | 0.6800 | 64.3179 | 70.9321 | 86.1270 |
| physical_dir | 3 | no | 71.3580 | 106.9928 | 0.7440 | 0.6342 | 63.2645 | 69.6155 | 84.5340 |
| physical_dir | 4 | no | 70.5459 | 105.8347 | 0.7600 | 0.6503 | 62.4084 | 68.8460 | 83.5978 |
| physical_dir | 5 | no | 68.9791 | 103.7053 | 0.7448 | 0.6359 | 61.0809 | 67.2758 | 81.8751 |
| physical_dir | 1 | yes | 49.6716 | 74.7557 | 0.4037 | 0.3525 | 36.1454 | 47.5771 | 70.4967 |
| physical_dir | 2 | yes | 46.2360 | 69.9992 | 0.3591 | 0.3160 | 32.6392 | 44.2874 | 67.0635 |
| physical_dir | 3 | yes | 44.9880 | 68.8964 | 0.3330 | 0.2948 | 31.2973 | 43.0470 | 65.9230 |
| physical_dir | 4 | yes | 44.0888 | 67.2346 | 0.3474 | 0.3084 | 30.5646 | 42.1594 | 64.7907 |
| physical_dir | 5 | yes | 43.7565 | 66.7479 | 0.3319 | 0.2953 | 30.2912 | 41.9322 | 64.1641 |

## Interpretation

### Why `x_pfpb` wins

`pfpb` alone forces the model to predict only from propagated signals. That can remove or dilute node-local information. `x_pfpb` keeps the raw node history and adds graph-propagated context, so it can use propagation when useful without losing the original temporal signal.

### Self-loop effect

Self-loop has opposite effects depending on whether the original signal is included:

- For `pfpb`, self-loop is very helpful because it injects each node's own signal into the propagated feature blocks.
- For `x_pfpb`, self-loop is usually neutral or slightly worse because the original `x` is already present. Adding self-loop to propagated blocks can make the propagated features more redundant with `x`.

### Graph choice

The `physical_dir` graph can look stronger in propagation diagnostics such as MADGap, but in this trained PropMLP setting the best predictive results still come from `distthre + x_pfpb`. This suggests that stronger graph-separation diagnostics do not automatically translate to better forecasting features.

## Reproduction

All runs use `BasicTS/baselines/PropMLP/SD.py` and are controlled by environment variables:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS

PROP_GRAPH_VARIANT=distthre \
PROP_FEATURE_MODE=x_pfpb \
PROP_MAX_ORDER=4 \
PROP_ADD_SELF_LOOP=0 \
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py \
  -c baselines/PropMLP/SD.py \
  --gpus 0
```

Useful variables:

```text
PROP_GRAPH_VARIANT: distthre | physical_dir
PROP_FEATURE_MODE:  x_only | pfpb | x_pfpb
PROP_MAX_ORDER:     integer K
PROP_ADD_SELF_LOOP: 0 | 1
```

Evaluation metrics are stored in:

```text
BasicTS/checkpoints/PropMLP/<experiment>/<run_id>/test_metrics.json
```


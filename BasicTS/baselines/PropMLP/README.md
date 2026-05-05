# PropMLP: distthre vs physical_dir on Old SD

This README compares `distthre` and `physical_dir` under matched PropMLP settings. Each row fixes the same feature mode, propagation order `K`, and self-loop setting, then compares the two graph choices directly.

Results were collected from server-side `test_metrics.json` files under:

```text
BasicTS/checkpoints/PropMLP/
```

Collected on: `2026-05-05`

## Experimental Setup

- Dataset: `SD`
- Input length: `12`
- Output length: `12`
- Model: `PropMLP`
- Epoch budget: `100`
- Hidden dimension: `128`
- Dropout: `0.1`
- Seed: `42`
- Primary comparison metric: overall test `MAE`

Feature modes:

```text
x_only  : raw node history only; graph-agnostic baseline
pfpb    : propagated forward/backward graph features only
x_pfpb  : raw node history + propagated forward/backward graph features
```

Delta convention:

```text
Delta MAE = physical_dir MAE - distthre MAE
Delta > 0 means distthre is better.
Delta < 0 means physical_dir is better.
```

## Summary

The strongest PropMLP result is:

```text
distthre + x_pfpb + K=4 + no self-loop
MAE=37.3438, RMSE=60.3250, MAPE=0.2680, WAPE=0.2410
```

Graph comparison takeaways:

- For `x_pfpb`, `distthre` beats `physical_dir` in all 10 matched settings.
- For `x_pfpb`, the average MAE advantage of `distthre` over `physical_dir` is `2.1746`.
- For `pfpb`, `distthre` beats `physical_dir` in 8 of 10 matched settings.
- The only `pfpb` settings where `physical_dir` wins are self-loop `K=1` and self-loop `K=2`.
- Pure `pfpb` is much more sensitive to self-loops than `x_pfpb`.

The graph-agnostic baseline is:

| Mode | MAE | RMSE | MAPE | WAPE | H3 MAE | H6 MAE | H12 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| x_only | 42.2939 | 67.3608 | 0.3370 | 0.2888 | 25.1931 | 40.4161 | 67.8622 |

## Matched Comparison: `x_pfpb`

`x_pfpb` keeps the original node signal and appends graph-propagated features. This is the most important setting for forecasting performance.

| K | Self-loop | dist MAE | phys MAE | Delta MAE | Winner | dist RMSE | phys RMSE | dist WAPE | phys WAPE |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | no | 38.6227 | 41.1449 | +2.5223 | distthre | 61.4954 | 65.7564 | 0.2568 | 0.2665 |
| 2 | no | 37.9624 | 40.6024 | +2.6401 | distthre | 60.0858 | 64.8826 | 0.2670 | 0.2910 |
| 3 | no | 38.1105 | 39.7135 | +1.6030 | distthre | 60.5381 | 63.5103 | 0.2696 | 0.2613 |
| 4 | no | 37.3438 | 39.2206 | +1.8768 | distthre | 60.3250 | 62.9108 | 0.2410 | 0.2703 |
| 5 | no | 37.3500 | 38.8375 | +1.4875 | distthre | 59.8173 | 62.2023 | 0.2479 | 0.2684 |
| 1 | yes | 38.6803 | 40.9458 | +2.2655 | distthre | 61.6319 | 65.2088 | 0.2552 | 0.2642 |
| 2 | yes | 38.0447 | 40.9517 | +2.9070 | distthre | 60.3398 | 65.5965 | 0.2638 | 0.3044 |
| 3 | yes | 37.7298 | 40.2490 | +2.5191 | distthre | 60.1568 | 63.6084 | 0.2608 | 0.2916 |
| 4 | yes | 37.4299 | 39.4446 | +2.0147 | distthre | 60.5163 | 63.3291 | 0.2417 | 0.2712 |
| 5 | yes | 37.5911 | 39.5011 | +1.9100 | distthre | 60.2105 | 63.5520 | 0.2476 | 0.2784 |

`x_pfpb` verdict:

```text
distthre wins: 10 / 10
physical_dir wins: 0 / 10
mean Delta MAE: +2.1746
best distthre: K=4, no self-loop, MAE=37.3438
best physical_dir: K=5, no self-loop, MAE=38.8375
```

Interpretation:

`physical_dir` improves as `K` increases, but it never catches `distthre` in matched `x_pfpb` settings. The raw node history already carries strong local information; in this model, `distthre` provides more useful additional propagated context.

## Matched Comparison: `pfpb`

`pfpb` removes the raw node history and gives the MLP only propagated graph features. This setting tests whether graph propagation alone preserves enough predictive information.

| K | Self-loop | dist MAE | phys MAE | Delta MAE | Winner | dist RMSE | phys RMSE | dist WAPE | phys WAPE |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | no | 60.9045 | 74.2041 | +13.2996 | distthre | 88.4952 | 109.7298 | 0.4487 | 0.6996 |
| 2 | no | 54.2800 | 72.6776 | +18.3976 | distthre | 80.5682 | 108.0681 | 0.3716 | 0.6800 |
| 3 | no | 52.0029 | 71.3580 | +19.3551 | distthre | 77.9005 | 106.9928 | 0.3572 | 0.6342 |
| 4 | no | 50.1667 | 70.5459 | +20.3791 | distthre | 75.0435 | 105.8347 | 0.3601 | 0.6503 |
| 5 | no | 49.7227 | 68.9791 | +19.2564 | distthre | 74.9285 | 103.7053 | 0.3326 | 0.6359 |
| 1 | yes | 56.7366 | 49.6716 | -7.0651 | physical_dir | 83.0622 | 74.7557 | 0.4107 | 0.3525 |
| 2 | yes | 47.3278 | 46.2360 | -1.0918 | physical_dir | 71.5075 | 69.9992 | 0.3133 | 0.3160 |
| 3 | yes | 43.8513 | 44.9880 | +1.1367 | distthre | 67.5539 | 68.8964 | 0.2907 | 0.2948 |
| 4 | yes | 42.1105 | 44.0888 | +1.9782 | distthre | 65.4429 | 67.2346 | 0.2801 | 0.3084 |
| 5 | yes | 41.8674 | 43.7565 | +1.8891 | distthre | 64.9560 | 66.7479 | 0.2807 | 0.2953 |

`pfpb` verdict:

```text
distthre wins: 8 / 10
physical_dir wins: 2 / 10
mean Delta MAE: +8.7535
best distthre: K=5, self-loop, MAE=41.8674
best physical_dir: K=5, self-loop, MAE=43.7565
```

Interpretation:

Without self-loops, `physical_dir + pfpb` is very weak because pure propagation loses too much node-local information. Adding self-loops fixes much of that by letting each node retain its own signal in the propagated blocks. Even then, `distthre` becomes better again from `K=3` onward.

## Best Matched Settings by Feature Mode

| Feature mode | Best distthre setting | Best dist MAE | Best physical_dir setting | Best phys MAE | Better graph |
|---|---|---:|---|---:|---|
| x_pfpb | K=4, no self-loop | 37.3438 | K=5, no self-loop | 38.8375 | distthre |
| pfpb | K=5, self-loop | 41.8674 | K=5, self-loop | 43.7565 | distthre |

## Relation to Propagation Diagnostics

The graph diagnostics can favor `physical_dir` in some cases, especially MADGap under no-self-loop propagation. That does not directly imply better forecasting in PropMLP.

The predictive comparison here says:

```text
For final forecasting with PropMLP, distthre is consistently stronger under matched settings.
```

One likely reason is that `physical_dir` creates a more selective propagation structure, but the trained MLP benefits more from the dense distance-threshold neighborhood when the raw signal is included.

## Reproduction

All runs use:

```text
BasicTS/baselines/PropMLP/SD.py
```

Example:

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

Useful environment variables:

```text
PROP_GRAPH_VARIANT: distthre | physical_dir
PROP_FEATURE_MODE:  x_only | pfpb | x_pfpb
PROP_MAX_ORDER:     integer K
PROP_ADD_SELF_LOOP: 0 | 1
```

Metrics are stored in:

```text
BasicTS/checkpoints/PropMLP/<experiment>/<run_id>/test_metrics.json
```

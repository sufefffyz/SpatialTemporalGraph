# Decoupled Spatiotemporal Diagnostics MVP

One-sentence thesis:

> Existing traffic forecasting baselines can look acceptable on aggregate MAE while mostly fitting low-frequency traffic cycles and missing high-frequency perturbations, spatially correlated residuals, or distribution tails.

## Scope

- Primary dataset: `SD_5min_full`
- Inputs: existing BasicTS saved `test_results/predictions.npy` and `test_results/targets.npy`
- Backbones: any baseline that has saved BasicTS test results, initially GWNet / DCRNN / temporal baselines if available
- Must not claim: a new forecasting model, causal traffic mechanism discovery, or event detection without event labels

## First Run Order

1. Discover existing SD baseline `test_results` directories on the server.
2. Run low-frequency / high-frequency decomposition diagnostics on each available baseline.
3. Compare normal windows against high-traffic windows.
4. Run standardized performance diagnostics that do not depend on a decomposition method.
5. Compare decomposition methods so the conclusion does not depend on one arbitrary low-pass filter.
6. Run alignment diagnostics to separate true failures from small time/node shifts.
7. Keep optional graph residual diagnostics as structural analysis rather than performance ranking metrics.
8. Use the signal to decide whether to pursue event-aware, retrieval-based, or graph-contribution follow-up work.

## Success Gate

Proceed if:

- At least two baselines produce usable saved predictions.
- The low/high-frequency split is stable enough under moving-average and FFT low-pass decompositions that model rankings or failure modes are interpretable.
- The summary table shows a clear difference between aggregate error and one diagnostic dimension.

Stop or pivot if:

- Available predictions are missing or not aligned with the same SD split.
- Low/high decomposition is unstable across moving-average windows.
- Moving-average and FFT low-pass decompositions produce incompatible conclusions.
- All diagnostic dimensions reproduce the same ordering as aggregate MAE with no extra signal.

## Quick Start

From the repository root on the server:

```bash
bash mvp_experiments/active/decoupled_spatiotemporal_diagnostics/scripts/run_server_sd_diagnostics.sh
```

By default this scans `BasicTS/checkpoints`. If saved results live elsewhere, pass one or more search roots:

```bash
bash mvp_experiments/active/decoupled_spatiotemporal_diagnostics/scripts/run_server_sd_diagnostics.sh \
  /data/yuzhang_fei/some_result_root \
  /home/yuzhang_fei/another_result_root
```

For a specific checkpoint directory:

```bash
python mvp_experiments/active/decoupled_spatiotemporal_diagnostics/scripts/run_frequency_diagnostics.py \
  --dataset-name SD_5min_full \
  --run "GWNet_distthre=BasicTS/checkpoints/GraphWaveNet/SD_5min_full_distthre_1m_100_12_12" \
  --decomp-methods moving_average fft_lowpass \
  --moving-window 12 \
  --fft-cutoff-period 12 \
  --output-dir mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/manual
```

The script accepts either a checkpoint directory containing `test_results/` or the `test_results/` directory itself.

## Low / High Decomposition Methods

- `moving_average`: low frequency is a centered moving average over `--moving-window` time steps; high frequency is original minus low.
- `fft_lowpass`: low frequency is reconstructed from the real FFT after keeping only frequencies with period at least `--fft-cutoff-period` time steps; high frequency is original minus low.
- The server wrapper now defaults to both methods: `DECOMP_METHODS="moving_average fft_lowpass"`.

## Outputs

```text
summary.json                         machine-readable report
component_metrics.csv                full / low / high component errors
standard_performance_metrics.csv     MAE/RMSE/WAPE/W1/MASE/RMSSE/Worst-k/Peak-F1 by horizon
standard_rank_details.csv            GIFT-style per-horizon rank details
standard_rank_summary.csv            GIFT-style average rank over standard metrics
peak_window_metrics.csv              standard normal-vs-high-traffic errors
distribution_metrics.csv             Wasserstein-style distribution distances
alignment_metrics.csv                ShiftGain / PeakLag / relaxed peak hit metrics by horizon
time_shift_curve.csv                 MAE under small prediction-time shifts
spatial_residual_metrics.csv         optional structural diagnostics, not performance ranking metrics
diagnostic_summary.md                compact human-readable summary
standard_average_summary.csv          standard metrics averaged across horizons
alignment_average_summary.csv         alignment diagnostics averaged across horizons
decomposition_average_summary.csv     decomposition-dependent low/high metrics averaged across horizons
decoupled_method_comparison.csv       metric deltas between decomposition methods
```

## Tracker

| Run ID | Purpose | Dataset | Metrics | Status | Notes |
|---|---|---|---|---|---|
| R001 | Discover server-side baseline predictions | SD_5min_full | found run count | TODO | Uses `run_server_sd_diagnostics.sh`. |
| R002 | Frequency decomposition diagnostics | SD_5min_full | low/high MAE, RMSE, WAPE, W1 | TODO | Decomposition-dependent table only. |
| R003 | Standardized performance diagnostics | SD_5min_full | WAPE, MASE/RMSSE, worst-k MAE, peak F1, GIFT-style average rank | TODO | Does not depend on low/high split. |
| R004 | Peak-window diagnostics | SD_5min_full | peak vs normal MAE/WAPE/W1 and peak F1 | TODO | Uses per-node target quantile threshold. |
| R005 | Alignment diagnostics | SD_5min_full | ShiftGain, PeakLag, relaxed hit@k,dt | TODO | Detects small time/node shifts; k>0 needs adjacency. |
| R006 | Optional spatial residual diagnostics | SD_5min_full | edge residual correlation / smoothness | TODO | Structural signal only; not a performance metric. |

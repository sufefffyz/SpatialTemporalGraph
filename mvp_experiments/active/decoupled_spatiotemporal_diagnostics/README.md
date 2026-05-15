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
4. Add optional graph residual diagnostics if an SD adjacency file is available.
5. Use the signal to decide whether to pursue event-aware, retrieval-based, or graph-contribution follow-up work.

## Success Gate

Proceed if:

- At least two baselines produce usable saved predictions.
- The low/high-frequency split is stable enough that model rankings or failure modes are interpretable.
- The summary table shows a clear difference between aggregate error and one diagnostic dimension.

Stop or pivot if:

- Available predictions are missing or not aligned with the same SD split.
- Low/high decomposition is unstable across moving-average windows.
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
  --output-dir mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/manual
```

The script accepts either a checkpoint directory containing `test_results/` or the `test_results/` directory itself.

## Outputs

```text
summary.json                         machine-readable report
component_metrics.csv                full / low / high component errors
peak_window_metrics.csv              normal-vs-high-traffic errors
distribution_metrics.csv             Wasserstein-style distribution distances
spatial_residual_metrics.csv         optional graph residual diagnostics
diagnostic_summary.md                compact human-readable summary
```

## Tracker

| Run ID | Purpose | Dataset | Metrics | Status | Notes |
|---|---|---|---|---|---|
| R001 | Discover server-side baseline predictions | SD_5min_full | found run count | TODO | Uses `run_server_sd_diagnostics.sh`. |
| R002 | Frequency decomposition diagnostics | SD_5min_full | low/high MAE, RMSE, WAPE | TODO | Runs on all discovered saved predictions. |
| R003 | Peak-window diagnostics | SD_5min_full | peak vs normal MAE/WAPE | TODO | Uses per-node target quantile threshold. |
| R004 | Optional spatial residual diagnostics | SD_5min_full | edge residual correlation / smoothness | TODO | Requires adjacency path. |

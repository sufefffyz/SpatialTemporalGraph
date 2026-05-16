# SD Decomposition Method Comparison

Date: 2026-05-16

## Purpose

Check whether the low/high-frequency diagnostic depends too much on the original centered moving-average split.

## Setup

- Dataset: `SD`
- Models: 10 non-PropMLP baselines with saved `diagnostic_export_models_20260515` test results.
- Horizons: H1-H12.
- Methods:
  - `moving_average`: centered 12-step moving average as low frequency.
  - `fft_lowpass`: real FFT low-pass with cutoff period 12 steps; high frequency is original minus reconstructed low frequency.
- Spatial residual graph: `BasicTS/datasets/SD/adj_mx.pkl`, 1000 sampled edges.

## Outputs

- Server: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/results/server_sd_baselines_20260516_decomp_method_comparison`
- Local copy: `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/outputs/server_sd_baselines_20260516_decomp_method_comparison`
- Main files:
  - `decoupled_average_summary.csv`
  - `decoupled_method_comparison.csv`
  - `decoupled_diagnostic_report.md`
  - `decomposition_method_metric_heatmaps.png`
  - `component_mae_heatmaps_moving_average.png`
  - `component_mae_heatmaps_fft_lowpass.png`

## Key Observations

- FFT low-pass gives lower average High MAE than moving average for every model.
- FFT low-pass also lowers high-component lag-1 residual correlation and high Dirichlet for every model, meaning this split extracts a less temporally/spatially sticky residual.
- High W1 is not uniformly lower under FFT. It increases for STID, DCRNN, STAEformer, GraphWaveNet, FEDformer, Crossformer, and STGCN, but decreases for Autoformer, ITransformer4D, and PatchTST.
- The worst high-frequency models remain suspicious under both methods: PatchTST and ITransformer4D are still high on high-component errors or spatial residual structure.

## Interpretation

Moving-average high frequency is a harsher residual because edge padding and local smoothing leave more short-term structure in the residual. FFT low-pass is cleaner as a frequency-domain split, but it can expose distribution mismatch in the extracted high-frequency component, which is why High W1 can rise even when High MAE falls.

Practical next step: when reporting high/low diagnostics, show both methods or use FFT low-pass as the primary frequency-domain check and moving average as a robustness baseline.

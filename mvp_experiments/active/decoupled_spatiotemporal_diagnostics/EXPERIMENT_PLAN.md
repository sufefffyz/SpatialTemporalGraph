# Experiment Plan

**Problem**: Aggregate traffic forecasting metrics hide whether a baseline fits low-frequency cycles, high-frequency perturbations, spatial propagation, or distribution shape.
**Method Thesis**: Reuse saved BasicTS predictions and decompose each model's errors into frequency, peak-window, distributional, and optional graph-residual diagnostics.
**Date**: 2026-05-15

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Runs |
|---|---|---|---|
| C1: Baselines differ beyond aggregate MAE. | A better average MAE may still hide high-frequency or tail failure. | At least two baselines have different low/high or peak/normal profiles. | R002, R003 |
| C2: High-frequency errors explain hard traffic failures. | This determines whether event-aware or retrieval-based follow-up is worth doing. | High-frequency component error is disproportionately large or ranks models differently from full-series MAE under more than one decomposition method. | R002, R011 |
| C3: Graph structure should reduce spatially correlated residuals if it is useful. | This tests whether STGNNs really help beyond temporal smoothing. | Graph baselines show lower edge residual correlation or smoother high-frequency residuals than temporal baselines. | R004 |

## Experiment Blocks

### Block 1: Saved Prediction Discovery

- Claim tested: server results are reusable without retraining.
- Dataset / split / task: `SD_5min_full`, BasicTS test split.
- Compared systems: all discovered directories with `test_results/predictions.npy` and `targets.npy`.
- Metrics: run count, inferred tensor shape, missing-file diagnostics.
- Success criterion: at least two usable saved prediction directories.
- Priority: MUST-RUN.

### Block 2: Frequency Component Diagnostics

- Claim tested: existing baselines mostly fit low-frequency traffic patterns.
- Dataset / split / task: each horizon of saved SD predictions.
- Compared systems: each discovered baseline.
- Metrics: MAE, RMSE, WAPE on full, low, and high components; high/full error ratio.
- Setup details: centered moving-average low-pass filter and FFT low-pass filter. Defaults use a 12-step moving window and a 12-step FFT cutoff period.
- Success criterion: high-frequency errors reveal a different failure profile from aggregate MAE.
- Priority: MUST-RUN.

### Block 2b: Decomposition-Method Robustness

- Claim tested: low/high conclusions are not an artifact of one arbitrary filter.
- Dataset / split / task: each horizon of saved SD predictions.
- Compared systems: each discovered baseline.
- Metrics: per-method low/high MAE, W1, residual lag1, edge residual correlation, residual Dirichlet; method deltas relative to moving average.
- Setup details: compare `moving_average` against `fft_lowpass` with matched 12-step scale.
- Success criterion: model ranking or qualitative failure modes are stable enough to interpret, or instability is explicitly reported.
- Priority: MUST-RUN.

### Block 3: Peak-Window Diagnostics

- Claim tested: hard traffic windows expose failures hidden by average metrics.
- Dataset / split / task: saved SD predictions, per-node high-flow threshold from test targets.
- Compared systems: each discovered baseline.
- Metrics: normal MAE/WAPE, peak MAE/WAPE, peak/normal error ratio.
- Success criterion: peak windows are materially harder, or rankings change under peak-only scoring.
- Priority: MUST-RUN.

### Block 4: Optional Spatial Residual Diagnostics

- Claim tested: graph-aware models reduce spatially structured residuals.
- Dataset / split / task: saved SD predictions plus optional adjacency file.
- Compared systems: temporal vs graph baselines if both are available.
- Metrics: edge residual correlation and residual Dirichlet energy on full / low / high residuals.
- Success criterion: graph baselines reduce high-frequency residual structure on neighboring nodes.
- Priority: NICE-TO-HAVE.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Pull latest code on server | `git pull --ff-only` | server reaches pushed commit | seconds | remote dirty worktree may block pull |
| M1 | Discover saved predictions | R001 | at least two runs found | seconds-minutes | results are under a different root |
| M2 | Run diagnostics | R002/R003 | CSV/JSON/MD generated | minutes | raw memmap shape inference mismatch |
| M3 | Optional graph diagnostics | R004 | adjacency is available | minutes | graph node order may not match predictions |
| M4 | Compare decomposition methods | R011 | method comparison table generated | minutes | FFT and moving average may disagree |
| M5 | Interpret next direction | summary table | clear failure profile | manual | signal may be weak |

## Risks and Mitigations

- Risk: BasicTS writes raw memmap files with `.npy` suffix.
- Mitigation: infer tensor shape from file size and dataset metadata if `np.load` fails.

- Risk: discovered runs use different SD windows or splits.
- Mitigation: record inferred shape and require comparable `num_samples`, `horizon`, and `num_nodes`.

- Risk: optional graph diagnostics use a mismatched adjacency.
- Mitigation: run graph diagnostics only when adjacency shape equals prediction node count.

- Risk: a moving-average residual is not a reliable high-frequency definition.
- Mitigation: add an FFT low-pass split and report both method-specific metrics plus paired deltas.

## Final Checklist

- [ ] Existing server predictions discovered
- [ ] Low/high component table generated
- [ ] Peak-window table generated
- [ ] Optional spatial diagnostics generated or explicitly skipped
- [ ] Moving-average vs FFT low-pass comparison generated
- [ ] Next modeling direction chosen from diagnostic evidence

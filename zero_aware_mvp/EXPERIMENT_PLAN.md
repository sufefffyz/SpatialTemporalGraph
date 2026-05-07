# Experiment Plan

Date: 2026-05-07

## Problem

Fine-grained urban link-level traffic forecasting may be poorly evaluated by
full-sample MAE/RMSE because observed zeros, nonzero occurrence, positive-flow
intensity, road-normalized high-flow events, and calibration are different
abilities.

All paper-facing MVP runs should use the full Urban Traffic Benchmark volume
NPZ at `/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz`.
Category-filtered local NPZ files are smoke-only.

For full city-M, the low-cost MVP intentionally saves edge lists instead of
dense adjacency matrices. Dense 53k-node graph baselines should be treated as a
separate scalability problem, not part of the first diagnostics/naive pass.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Runs |
|---|---|---|---|
| C1: Structured observed zeros exist. | Without this, the benchmark is unnecessary. | Zero rate changes by resolution, hour, and road attributes; Poisson/NB diagnostics do not fully explain zeros. | R010, R011 |
| C2: MAE hides important failures. | This is the "Beyond MAE" paper thesis. | Model ranking changes between MAE, occurrence F1/AP, MAE+, and road-q90 high-flow recall/F1. | R020, R030, R040 |
| C3: Traffic-specific event definitions matter. | Separates this from precipitation/OD/crash work. | Road-normalized q90 high-flow produces different evidence than global high-flow; speed-volume can be added after MVP. | R021, R050 |

## Must-Run Blocks

### Block 1: Zero Diagnostics

- Claim tested: C1.
- Dataset: full city-traffic-M volume.
- Systems: no model.
- Metrics: zero rate, overdispersion, positive tail, aggregation sensitivity, road/hour strata, transition rates, Poisson/NB expected-zero gap.
- Success criterion: zero behavior is structured and not fully explained by a simple low-mean count process.
- Failure interpretation: pivot from "zero-inflated" to "sparse overdispersed count forecasting."
- Priority: MUST-RUN.

### Block 2: Naive Ranking Reversal

- Claim tested: C2.
- Dataset: BasicTS `TRAFFIC_VOLUME_FULL_5MIN`.
- Systems: all-zero, previous step, node mean, node median, seasonal median, one-day seasonal.
- Metrics: MAE, RMSE, WAPE, occurrence precision/recall/F1/AP, MAE+, WAPE+, road-q90 high-flow precision/recall/F1/AP.
- Success criterion: all-zero or conservative baselines look deceptively strong under MAE but fail occurrence / positive / high-flow metrics.
- Priority: MUST-RUN.

### Block 3: BasicTS Neural Smoke

- Claim tested: pipeline validity for C2.
- Dataset: BasicTS `TRAFFIC_VOLUME_FULL_5MIN`.
- Systems: AGCRN and GWNet with 3-5 epochs.
- Metrics: BasicTS point metrics plus post-hoc zero-aware metrics.
- Success criterion: training completes, predictions are saved, and post-hoc metrics are comparable with naive baselines.
- Priority: MUST-RUN before longer GPU runs.

### Block 4: BasicTS Main MVP

- Claim tested: C2 and C3.
- Dataset: `TRAFFIC_VOLUME_FULL_5MIN`, optionally `15MIN` for aggregation sensitivity.
- Systems: best naive baselines, AGCRN, GWNet, optional MTGNN.
- Metrics: same as Block 2.
- Success criterion: at least one zero-aware metric changes model conclusions relative to full MAE/RMSE.
- Priority: MUST-RUN after smoke.

### Block 5: Extensions

- Claim tested: C3.
- Dataset: volume plus optional speed NPZ.
- Systems: post-hoc definitions over existing predictions.
- Metrics: speed-volume high-flow-low-speed event F1; probe-thinning robustness.
- Success criterion: adds traffic-specific evidence.
- Priority: NICE-TO-HAVE.

## Run Order

| Run ID | Stage | Purpose | Cost | Go/No-Go Gate |
|---|---|---|---|---|
| R000 | dependency | Check Python envs. | seconds | `numpy/pandas` for scripts; `torch/easytorch` for BasicTS training. |
| R001 | data | Prepare BasicTS 5min dataset. | minutes / disk-heavy | `BasicTS/datasets/TRAFFIC_VOLUME_FULL_5MIN/desc.json` exists. |
| R002 | data | Prepare 15/30/60min datasets. | minutes | aggregation summaries exist. |
| R010 | diagnostics | Full zero diagnostics. | minutes | structured zero patterns exist. |
| R011 | diagnostics | Poisson/NB expected-zero gap. | minutes | gap remains meaningful after road/time conditioning. |
| R020 | naive | Run naive baselines on 5min. | minutes | zero-aware ranking differs from MAE ranking. |
| R030 | BasicTS | AGCRN 3-epoch smoke. | low GPU | training and `test_results/` complete. |
| R031 | BasicTS | GWNet 3-epoch smoke. | low GPU | training and `test_results/` complete. |
| R040 | eval | Post-hoc zero-aware eval of BasicTS outputs. | minutes | comparable table against naive baselines. |
| R050 | main | Longer AGCRN/GWNet runs. | GPU hours | run only if R010/R020/R040 pass. |
| R060 | extension | 15/30/60min model repeats. | GPU hours | run if aggregation claim is promising. |
| R070 | extension | Probe thinning. | extra | run if observed-zero story needs support. |

## Stop Rules

- Stop or reframe if 5/15/30/60min zero rates barely change.
- Stop or reframe if MAE ranking and zero-aware rankings are nearly identical across naive and neural systems.
- Stop or reframe if road-q90 thresholds are invalid for too many roads.

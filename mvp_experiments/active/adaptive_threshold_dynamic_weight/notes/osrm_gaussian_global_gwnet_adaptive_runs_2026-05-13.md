# OSRM Gaussian Global GWNet Adaptive Runs - 2026-05-13

## Purpose

Add the GraphWaveNet reference with `addaptadj=True` on the same five OSRM Gaussian global threshold graphs used by the fixed-support sweep. This isolates whether the learned adaptive adjacency improves or rescues performance when the physical OSRM graph is sparse or over-dense.

## Variants

- `beta0p25`: `SD_OSRMGG_B025`
- `beta0p50`: `SD_OSRMGG_B050`
- `beta1p00`: `SD_OSRMGG_B100`
- `beta1p50`: `SD_OSRMGG_B150`
- `beta2p00`: `SD_OSRMGG_B200`

## Config

- BasicTS config: `BasicTS/baselines/GWNet/SD_osrm_gaussian_global_adaptive.py`
- Runner script: `mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/run_osrm_gaussian_global_gwnet_adaptive_sweep.sh`
- Difference from fixed-support GWNet: same physical transition supports, but `CFG.MODEL.PARAM["addaptadj"] = True`.
- W&B project: `adaptive_threshold_dynamic_weight`

## Notes

This is not an adaptive-only graph. It is the physical OSRM Gaussian threshold graph plus GraphWaveNet's learned adaptive adjacency.

# CD-003 Completion Report: LargeST-LTSF Complexity And Protocol

Status: `TOOLING_READY_NOT_RUN_FULL`

Date: 2026-06-24

## Result Summary

This branch defines the LargeST input-window protocol and adds a resource estimator. After the master correction, the first probe fixes `H=12` and sweeps longer input windows; standard LTSF-style long-input/long-output forecasting with `H=672` is a later follow-up, not the immediate default.

The plan's target hardware assumption is A6000 48G, but the current smoke jobs are being run on A100 80G servers. Memory observed in smoke should be treated as a feasibility upper-bound reference, not as a direct A6000 pass/fail result.

Default protocol:

- Dataset wave: SD first; GBA for scalable models after smoke; GLA only after probe.
- First probe horizon: `H=12`.
- Later long-horizon reference: `H=672`.
- Input lengths: `L={96,192,336,672}`.
- First probe report horizon: `{12}` plus overall metrics when available.

## Added Artifacts

- `ltsf_input_window/scripts/estimate_ltsf_resources.py`
- `ltsf_input_window/manifests/smoke_grid.json`
- `ltsf_input_window/manifests/pilot_grid.json`

## Model Triage

| Model | First-wave role | Scalability risk |
|---|---|---|
| STID | required baseline | low |
| DLinear | low-cost sanity baseline | low |
| CycleNet | periodic LTSF baseline | low |
| TimeMixer | recent multiscale LTSF baseline | medium |
| PatchTST | SD/probe only | medium-to-high as patch count grows |
| BiST | scalable ST baseline with profile guard | medium on SD, high on larger graphs |
| iTransformer | not first-wave full | high due to node-quadratic attention |

## Suggested Merge Note

Use `ltsf_input_window` as the artifact home for the LargeST input-window probe. The smoke stage should run SD first with `H=12` and `L={96,192,336,672}`. GBA should only receive scalable models after the SD smoke/probe confirms no shape, memory, or metric failures. The `H=672` setting remains a later stress-test once the `H=12` trend is understood.

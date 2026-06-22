# Branch Brief: CD-001 - LargeST2019 Baselines

- Branch ID: `CD-001`
- Stable session title: `[CD-001][W1][repro] LargeST2019 baselines`
- Type: `branch`
- Role: reproduction
- Based on snapshot: `snap-2026-06-22-001`
- Brief version: 1
- Created: 2026-06-22
- Parent: `[CD-MAIN][master] LargeST2019 virtual-node benchmark`

## Purpose Card

You are branch CD-001: LargeST2019 baselines.

Purpose: reproduce or prepare reproduction of PatchSTG, BiST, MAGE, and STID on LargeST 2019 `SD`, `GLA`, and `GBA`.

Not for: dynamic-threshold experiments, new model design, or changing baseline hyperparameters.

Input: master snapshot `snap-2026-06-22-001`; allowed repo context only.

Output: `completion-report.md`.

Return to master when: the SD/GLA/GBA baseline matrix is clear, including completed runs, queued runs, missing code/data blockers, and exact next commands.

## Why This Branch Exists

The project needs a clean baseline coordinate system before deciding whether a virtual-node/bipartite scalable ST forecasting idea is worth implementing.

## Allowed Context

- `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/LargeST`
- `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/PatchSTG`
- `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BiST`
- `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/baselines/STID`
- Server execution context for `183.174.228.180` and `183.174.228.178` if needed.
- Public official sources for PatchSTG, BiST, MAGE, and STID, only to verify code availability and official settings.

## Tasks

1. Audit local code availability:
   - PatchSTG: confirm local entrypoint and config for `SD`, `GLA`, `GBA`.
   - BiST: confirm local entrypoint and command for `SD`, `GLA`, `GBA`.
   - STID: confirm BasicTS configs for `SD`, `GLA`, `GBA`.
   - MAGE: confirm whether official code exists and whether it is already local. If not available, record as blocker; do not hand-write a reproduction.
2. Audit LargeST 2019 data availability for `SD`, `GLA`, `GBA`.
3. Produce smoke commands for `SD 2019`.
4. If safe and environment is ready, start SD smoke/full runs and queue GLA/GBA according to available GPUs. Do not alter baseline hyperparameters unless required for runtime feasibility.
5. Track metrics as MAE/RMSE/MAPE, plus training time, GPU memory, dataset version, and metric convention.

## Completion Criteria

- PatchSTG, BiST, and STID each have a clear status on `SD 2019`.
- `GLA 2019` and `GBA 2019` are either queued with commands or marked blocked with reason.
- MAGE status is explicit: official code available/local/missing.
- Report includes exact commands, config paths, and any deviations from paper/default settings.
- No unrelated repo files are modified.

## Expected Completion Report

Create `completion-report.md` with:

- Branch status: `DONE`, `DONE_WITH_CONCERNS`, or `BLOCKED`.
- Data availability table.
- Baseline availability table.
- Run/queue table for `SD`, `GLA`, `GBA`.
- Current metric table if any result has completed.
- Missing items and proposed next master decision.
- Suggested Merge Note, no more than 8 bullets.


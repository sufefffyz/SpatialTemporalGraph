# Branch Brief: CD-001 - LargeST2019 Baseline Matrix

## Session Identity

- Stable session title: `[CD-001][W1][experiment] LargeST2019 baseline matrix`
- Type: branch
- Role: experiment
- Status: active after thread creation

## Purpose Card

```text
You are branch CD-001: LargeST2019 baseline matrix
Purpose: complete, verify, and organize the LargeST 2019 SD/GLA/GBA baseline matrix for STID, PatchSTG, BiST, and MAGE.
Not for: new method design, dynamic-threshold experiments, broad literature review, or changing baseline hyperparameters.
Input: master snapshot snap-2026-06-24-001; this brief; explicitly listed files and remote logs only.
Output: completion-report.md plus a compact result table under the branch artifact directory.
Return to master when: the SD/GLA/GBA x model matrix is complete or every missing cell has a precise blocker.
```

## Parent Context

- Parent task: LargeST 2019 scalable ST forecasting
- Parent branch: `[CD-MAIN][master] LargeST2019 scalable ST`
- Snapshot id: `snap-2026-06-24-001`
- Brief version: 1
- Created at: 2026-06-24
- Artifact root: `SpatialTemporalGraph/mvp_experiments/active/largest2019_scalable_st`
- Why this branch exists: the master session needs one clean baseline coordinate system before deciding whether to implement a virtual-node prototype, Robust-LargeST benchmark, or another scalable ST idea.

## Branch Goal

Produce a verified, paper-readable baseline matrix for LargeST 2019 `SD / GLA / GBA` covering `STID`, `PatchSTG`, `BiST`, and `MAGE`, with metric convention and runtime caveats made explicit.

## Dependency / Order

- Execution wave: W1
- Depends on: current master snapshot only
- Required inputs before start:
  - old CD-001 report: `../largest2019_virtual_node/conductor/CD-001_largest2019_baselines/completion-report.md`
  - remote servers `183.174.228.178` and `183.174.228.180`
- Can run in parallel with: `CD-002`
- Unblocks: `W2_direction_decision`
- Gate condition: completion report reviewed by master
- Start policy: current wave only

## Current Master-Approved Status

As of the latest master refresh on 2026-06-24 around 11:48 CST:

- `STID` finished on `SD / GLA / GBA`.
- `PatchSTG` finished on `SD / GLA / GBA`.
- `BiST` finished on `SD / GLA / GBA`.
- `MAGE` finished on `SD / GBA`; `MAGE GLA` was still running at approximately Epoch 60.
- Metric conventions are not fully unified yet: STID values came from BasicTS `test_metrics.json` overall metrics, while PatchSTG/BiST/MAGE values came from each framework's `Average Test` logs.

Known intermediate table from master refresh:

| Model | SD 2019 | GLA 2019 | GBA 2019 | Convention |
|---|---|---|---|---|
| STID | `17.7999 / 30.9125 / 0.1184` | `19.5705 / 33.5828 / 0.1217` | `20.1669 / 34.5046 / 0.1596` | BasicTS overall |
| PatchSTG | `16.6601 / 28.8548 / 0.1080` | `19.3580 / 33.0774 / 0.1152` | `19.8441 / 33.8339 / 0.1493` | framework average |
| BiST | `16.6520 / 28.0990 / 0.1161` | `19.9555 / 32.6986 / 0.1224` | `19.8548 / 32.9439 / 0.1526` | framework average |
| MAGE | `16.3710 / 28.2346 / 0.1093` | running | `19.7808 / 33.1533 / 0.1555` | framework average |

Each cell is `MAE / RMSE / MAPE`.

## In Scope

- Refresh remote status and logs for `CD-001` jobs.
- Extract final metrics for all completed cells.
- Record metric convention for each model and explain whether values are directly comparable.
- Record training time and peak GPU memory where available.
- Preserve exact config / command details that affect reproducibility.
- Produce a compact `summary.md` and/or `summary.csv` under this branch directory.
- If `MAGE GLA` completes during the branch, add its final result.
- If `MAGE GLA` fails or stalls, record the exact failure/stall evidence and do not silently replace the run.

## Out Of Scope

- Do not alter model code or hyperparameters.
- Do not launch new model families unless the master explicitly asks.
- Do not revive dynamic-threshold modules.
- Do not perform broad novelty review; send that to `CD-002`.
- Do not merge or overwrite old `largest2019_virtual_node` artifacts.

## Allowed Context

This branch may use only:

- this brief
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_scalable_st/conductor/branch-map.md`
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_scalable_st/conductor/snapshots/snap-2026-06-24-001.md`
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_virtual_node/conductor/CD-001_largest2019_baselines/completion-report.md`
- local repo paths for `LargeST`, `PatchSTG`, `BiST`, `BasicTS`, and `references/mage_official`
- remote read-only logs/checkpoints on `183.174.228.178` and `183.174.228.180`
- user messages inside this branch thread

## Forbidden Assumptions

- Do not assume access to the master session history.
- Do not treat decisions made here as globally accepted.
- Do not read or summarize sibling branch raw context.
- Do not merge anything into the master session unless the user confirms completion and later confirms merge in the master session.
- Do not treat mixed metric conventions as a final paper table without labeling them.

## Expected Artifacts

- `completion-report.md`
- `summary.md` or `summary.csv`
- optional `missing-or-caveats.md` if the matrix remains incomplete

## Completion Criteria

- `SD / GLA / GBA x STID / PatchSTG / BiST / MAGE` matrix is complete or every missing cell has a precise blocker.
- MAGE GLA status is refreshed from live logs.
- Metric convention is stated for each row.
- Runtime and memory are recorded when available.
- The report includes exact remote artifact paths for key results.
- Suggested merge note is no more than 8 bullets.

## Global Decision Rule

If the user makes or implies a decision that affects other branches, architecture, scope, naming, deadlines, acceptance criteria, or project goals, do not treat it as globally binding. Record it under "Proposed Global Decisions" in the completion report and ask the user to confirm it in the master session.

## Completion Report Requirements

When the branch appears done, suggest completion. After the user confirms completion, produce a completion report with:

- result summary
- artifacts and files
- local decisions
- proposed global decisions
- validation or checks
- risks
- suggested merge note

# Branch Brief: CD-002 - Scalable ST Direction Audit

## Session Identity

- Stable session title: `[CD-002][W1][research] Scalable ST direction audit`
- Type: branch
- Role: research
- Status: active after thread creation

## Purpose Card

```text
You are branch CD-002: Scalable ST direction audit
Purpose: audit candidate research directions for LargeST-scale spatiotemporal forecasting, especially virtual-node/bipartite bottlenecks, patch/expert methods, and graph-aware robustness benchmarks.
Not for: running experiments, modifying repo code, launching remote jobs, or writing a full paper section.
Input: master snapshot snap-2026-06-24-001; this brief; explicitly listed literature anchors and historical reports.
Output: completion-report.md with a go/hold/drop recommendation for W2.
Return to master when: the master can decide whether to implement a virtual-node prototype, Robust-LargeST benchmark, or another direction.
```

## Parent Context

- Parent task: LargeST 2019 scalable ST forecasting
- Parent branch: `[CD-MAIN][master] LargeST2019 scalable ST`
- Snapshot id: `snap-2026-06-24-001`
- Brief version: 1
- Created at: 2026-06-24
- Artifact root: `SpatialTemporalGraph/mvp_experiments/active/largest2019_scalable_st`
- Why this branch exists: the project needs a literature-grounded direction decision before adding another method or benchmark to an already busy repo.

## Branch Goal

Produce a concise but critical direction audit that tells the master whether W2 should pursue:

1. a LargeST-compatible virtual-node / bipartite bottleneck model;
2. a graph-aware Robust-LargeST / SensorFault benchmark;
3. a different scalable ST forecasting angle;
4. or no implementation yet.

## Dependency / Order

- Execution wave: W1
- Depends on: current master snapshot only
- Required inputs before start:
  - old virtual-node report: `../largest2019_virtual_node/conductor/CD-002_virtual_node_bipartite/completion-report.md`
  - SensorFault/Robust-LargeST first-pass novelty summary from master snapshot
- Can run in parallel with: `CD-001`
- Soft dependency: use `CD-001` final baseline matrix before making final W2 recommendation if available
- Unblocks: `W2_direction_decision`
- Gate condition: completion report reviewed by master
- Start policy: current wave only

## Current Master-Approved Status

Virtual-node/bipartite direction:

- Direct prior exists; do not claim generic virtual nodes or generic auxiliary-node bipartite inference as novel.
- A defensible niche may be a LargeST-scale, geography-aware, sparsely assigned `N -> M -> N` virtual bottleneck that preserves full-node forecasts.
- Required adjacent comparators include PatchSTG and MAGE.

Robust-LargeST / SensorFault direction:

- Generic sensor-fault forecasting benchmark is not new; SensorFault-Bench is a close prior.
- LargeST Gaussian-noise robustness is not new; CA-VMGCN is a close prior.
- A defensible version would need traffic graph-aware faults, such as corridor/region/sensor-type faults, not just random i.i.d. perturbation.

Experiment background:

- CD-001 is organizing LargeST 2019 `SD / GLA / GBA` baseline results for `STID`, `PatchSTG`, `BiST`, and `MAGE`.

## In Scope

- Search and summarize recent and foundational work on:
  - virtual nodes / global relay nodes for traffic or STGNNs;
  - latent inducing points / auxiliary nodes / bipartite graph inference for MTS/ST forecasting;
  - scalable spatial patches, experts, anchors, landmarks, or pooling;
  - robust forecasting benchmarks, SensorFault-style protocols, LargeST robust/noisy forecasting.
- Separate mechanism families instead of collapsing them into one bucket.
- Assess novelty risk, reviewer objections, complexity, interpretability, and implementation burden.
- Produce a minimal W2 experiment/design recommendation only if evidence supports it.
- State explicitly which claims are unsafe.

## Out Of Scope

- Do not run experiments.
- Do not edit model or dataset code.
- Do not create new datasets.
- Do not write a full related-work section.
- Do not revive dynamic-threshold work unless master explicitly reopens it.

## Allowed Context

This branch may use only:

- this brief
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_scalable_st/conductor/branch-map.md`
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_scalable_st/conductor/snapshots/snap-2026-06-24-001.md`
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_virtual_node/conductor/CD-002_virtual_node_bipartite/completion-report.md`
- `SpatialTemporalGraph/mvp_experiments/active/largest2019_virtual_node/conductor/CD-E01_bist_paper_explainer/reading-notes.md`
- Zotero and web literature searches
- user messages inside this branch thread

## Initial Literature Anchors

- Virtual Nodes Improve Long-term Traffic Prediction: `https://arxiv.org/abs/2501.10048`
- Multivariate Time Series Forecasting with Latent Graph Inference: `https://arxiv.org/abs/2203.03423`
- Set Transformer: `https://arxiv.org/abs/1810.00825`
- Perceiver / Perceiver IO: `https://arxiv.org/abs/2103.03206`, `https://arxiv.org/abs/2107.14795`
- PatchSTG: `https://arxiv.org/abs/2412.09972`
- MAGE: `https://openreview.net/forum?id=jCGwSLwOt9`
- SensorFault-Bench: `https://arxiv.org/abs/2605.10822`
- CA-VMGCN / robust LargeST long-term prediction: `https://arxiv.org/abs/2504.06660`

## Forbidden Assumptions

- Do not assume access to the master session history.
- Do not treat decisions made here as globally accepted.
- Do not read or summarize sibling branch raw context.
- Do not merge anything into the master session unless the user confirms completion and later confirms merge in the master session.
- Do not claim novelty unless the closest prior is explicitly compared.

## Expected Artifacts

- `completion-report.md`
- optional `literature-table.md`
- optional `w2-recommendation.md`

## Completion Criteria

- Literature table includes closest works, mechanism, complexity/scaling claim, relationship to our target, and collision level.
- Direction audit covers at least:
  - virtual-node/bipartite bottleneck;
  - patch/expert/anchor scalable spatial modeling;
  - graph-aware Robust-LargeST/SensorFault.
- Final output gives `go / go with constraints / hold / drop` for each direction.
- The report states what a W2 prototype would need to show to be credible.
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

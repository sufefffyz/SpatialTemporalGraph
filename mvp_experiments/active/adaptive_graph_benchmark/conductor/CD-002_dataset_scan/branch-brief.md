# Branch Brief: CD-002 - Adaptive graph benchmark dataset scan

## Session Identity

- Stable session title: `[CD-002][W1][research] Adaptive Graph Dataset Scan`
- Type: branch
- Role: research
- Status: completion_suggested

## Purpose Card

```text
You are branch CD-002: Adaptive graph benchmark dataset scan
Purpose: identify easy-to-use spatiotemporal datasets with complete spatial metadata for internal adaptive-graph benchmarking.
Not for: claiming benchmark construction as a paper innovation, launching training jobs, or changing BasicTS configs.
Input: current SpatialTemporalGraph repo snapshot, approved master decision to make benchmark internal infrastructure, web/source checks, local dataset traces.
Output: dataset-candidate-scan.md
Return to master when: candidate datasets and priority tiers are ready for user review.
```

## Parent Context

- Parent task: adaptive graph benchmark and learned-graph mechanism study
- Parent branch: CD-MAIN
- Snapshot id: snap-2026-06-10-adaptive-graph-benchmark-w1
- Brief version: 1
- Created at: 2026-06-10
- Why this branch exists: the main line shifted away from dynamic-threshold-as-method toward studying mainstream adaptive graph construction mechanisms; a clean internal benchmark needs datasets with trustworthy coordinates and metadata.

## Branch Goal

Produce a compact shortlist of datasets that are easy to obtain, have coordinates or spatial metadata, and are suitable for adaptive graph learned-structure analysis under BasicTS-like protocols.

## Dependency / Order

- Execution wave: W1
- Depends on: user approval of new main line
- Required inputs before start: current repo inspection and public dataset availability checks
- Can run in parallel with: CD-003 benchmark design
- Unblocks: CD-003 benchmark design, CD-004 implementation
- Gate condition: user chooses first benchmark dataset set
- Start policy: current wave only

## In Scope

- Traffic, air-quality, and weather/spatial sensor datasets.
- Coordinate and metadata availability.
- Approximate scale and compute risk.
- Whether BasicTS already has scripts or configs.
- Recommended priority tiers.

## Out Of Scope

- Running model experiments.
- Downloading new datasets.
- Rebuilding data preprocessing.
- Treating dataset benchmark as the paper novelty.

## Allowed Context

This branch may use only:

- this brief
- `BasicTS/stgraph_ext/`
- `BasicTS/scripts/data_preparation/`
- `BasicTS/baselines/`
- `BasicTS/datasets/`
- `mvp_experiments/active/adaptive_threshold_dynamic_weight/` summaries when needed for continuity
- public dataset documentation or repositories

## Forbidden Assumptions

- Do not assume a dataset is usable just because a time series is public; coordinates must be verified or clearly marked uncertain.
- Do not assume PEMS04/07/08 coordinate provenance is reliable without an explicit node-id mapping.
- Do not promote LargeST GLA/GBA into the first wave because the user has marked them too expensive.
- Do not claim benchmark construction itself as an innovation.

## Expected Artifacts

- `dataset-candidate-scan.md`

## Completion Criteria

- Recommend a first-wave dataset set.
- List backup candidates and exclusions.
- Separate verified local availability from public-but-not-yet-local availability.
- Record risks and next checks before implementation.

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

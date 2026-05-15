# MVP Experiments

This directory is the stable home for small, idea-specific MVP experiments in
the SpatialTemporalGraph project.

The repository already contains several large upstream codebases and older
validation folders. New research ideas should start here instead of adding more
top-level directories.

## Directory Rule

Use one folder per idea:

```text
mvp_experiments/active/<idea_slug>/
```

Each MVP folder should keep the same basic shape:

```text
README.md              idea summary, scope, quick start
EXPERIMENT_PLAN.md     claim-driven run plan
EXPERIMENT_TRACKER.md  run registry and status
configs/               copied or generated BasicTS configs
scripts/               launch, prepare, summarize scripts
src/                   local prototype modules
results/               small summarized metrics and tables
outputs/               raw or bulky run outputs, usually gitignored later
notes/                 related work, decisions, scratch notes
```

## Current Registry

| Idea | Location | Status | Notes |
|---|---|---|---|
| Adaptive threshold + dynamic edge weights | `mvp_experiments/active/adaptive_threshold_dynamic_weight/` | active | Current MVP for budgeted physical sparse STGNN support and dynamic edge weights. |
| Delay-selective propagation | `mvp_experiments/active/delay_selective/` | active | Migrated from the former top-level `delay_selective/` entry. |
| Zero-aware traffic forecasting | `mvp_experiments/active/zero_aware_mvp/` | active | Migrated from the former top-level `zero_aware_mvp/` entry. |
| Decoupled spatiotemporal diagnostics | `mvp_experiments/active/decoupled_spatiotemporal_diagnostics/` | active | Post-hoc low/high-frequency, peak-window, distribution, and optional spatial residual diagnostics for saved SD predictions. |

The machine-readable registry is `mvp_experiments/registry.yml`.

## Canonical Paths

Use the canonical MVP locations for new commands and notes:

```text
mvp_experiments/active/delay_selective/
mvp_experiments/active/zero_aware_mvp/
mvp_experiments/active/decoupled_spatiotemporal_diagnostics/
```

The former top-level aliases have been removed to keep the project root focused.
When reproducing an old command, replace the historical top-level path with the
matching `mvp_experiments/active/...` path.

## When Adding a New MVP

1. Create a short lowercase slug, e.g. `my_new_idea`.
2. Create `mvp_experiments/active/my_new_idea/` from `TEMPLATE.md`.
3. Put all idea-specific scripts/configs under the MVP folder.
4. Reuse shared code from `BasicTS/`, `benchmark/`, or `scripts/` by import or wrapper script rather than copying large codebases.
5. Keep generated checkpoints and large artifacts out of source-controlled files; store only summaries and reproduction commands.
6. Update the registry table in this file.
7. Update `registry.yml`.

## Boundary

This folder is for MVP validation, not for polished reusable library code. If an
MVP becomes stable, promote the reusable parts into `BasicTS/stgraph_ext/`,
`benchmark/`, or another shared package, and leave this folder as the experiment
record.

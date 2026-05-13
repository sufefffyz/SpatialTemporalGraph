# Repository Layout

This repository is a research workspace for spatiotemporal graph forecasting.
It contains upstream or vendored project code, shared experiment utilities,
idea-specific MVPs, and local data/output folders. The goal of this layout is
to keep new research ideas organized without breaking older scripts.

## Top-Level Roles

| Path | Role | Git policy |
|---|---|---|
| `BasicTS/` | Main training framework and shared STGNN extensions. | Track reusable code, configs, tests, and docs. Do not track datasets/checkpoints/logs. |
| `benchmark/` | Shared graph baselines, data prep, and evaluation scripts. | Track reusable benchmark code and small docs. |
| `scripts/` | Repo-level automation such as remote run and note sync helpers. | Track scripts and example env files; keep real secrets ignored. |
| `mvp_experiments/` | Stable home for idea-specific MVP experiments. | Track plans, configs, scripts, prototype code, and small summaries. Ignore raw outputs. |
| `LargeST/`, `PatchSTG/`, `BiST/`, `urban-traffic-benchmark/`, `causalrivers/` | Vendored or upstream project code used for experiments and comparison. | Track source/docs needed for reproducibility; avoid generated data and outputs. |
| `data/`, `datasets/`, `product/`, `test-output/` | Local data, generated products, and bulky run artifacts. | Ignored by default. |

## MVP Experiment Policy

New research ideas should start under:

```text
mvp_experiments/active/<idea_slug>/
```

Each MVP should use this shape:

```text
README.md
EXPERIMENT_PLAN.md
EXPERIMENT_TRACKER.md
configs/
scripts/
src/
results/
outputs/
notes/
```

Only small, reviewable artifacts belong in Git:

- experiment plans and trackers
- configs and launch scripts
- prototype source code
- compact summaries such as `.md`, `.json`, or small `.csv` tables

The following should stay local or on the experiment server:

- checkpoints
- raw predictions
- raw `.npy/.npz/.pkl` data
- W&B/log directories
- bulky figures or generated products

## Promotion Rule

MVP code starts local to the MVP folder. Once a component becomes reusable, move
it into a shared location:

- `BasicTS/stgraph_ext/` for reusable BasicTS extensions
- `benchmark/graph_baselines/` for graph construction baselines
- `benchmark/eval/` for shared evaluation and analysis

The MVP folder should remain as the experiment record after promotion.

## Legacy Policy

Older MVP folders have been moved into `mvp_experiments/active/`. The old
top-level compatibility symlinks have been removed, so use the canonical paths:

```text
mvp_experiments/active/delay_selective/
mvp_experiments/active/zero_aware_mvp/
```

When updating old scripts or reproducing historical runs, refresh command paths
to the canonical `mvp_experiments/active/...` location.

# Adaptive Threshold + Dynamic Edge Weight MVP

This MVP validates the current idea:

> Learn a budgeted physical sparse support for STGNN traffic forecasting, then
> compute dynamic message-passing weights only on the active sparse edges.

The experiment scope is intentionally narrow:

- Use **LargeST family** datasets first: `SD`, then `GBA/GLA/CA`.
- Use **PEMS only when node coordinates are recoverable** from metadata or
  matched sensor files.
- Do not claim efficiency from dense masked attention. Dense implementations
  are only accuracy prototypes.

## Why This Folder Exists

`SpatialTemporalGraph` already has several idea-specific validation folders.
This folder is the new organized home for the adaptive-threshold MVP, keeping
all configs, scripts, notes, and results together.

## Folder Layout

```text
configs/               BasicTS configs for generated graph variants
scripts/               graph generation, launch, and summarization scripts
src/                   local prototype code for graph support/edge weights
results/               compact tables, JSON summaries, figures
outputs/               raw logs or intermediate outputs
notes/                 related work notes and design decisions
```

## First Run Order

1. Validate coordinates and graph construction on LargeST `SD/GBA/GLA/CA`.
2. Check which PEMS datasets can be mapped to node coordinates.
3. Run fixed sparse Pareto on LargeST-SD:
   - global distance threshold
   - per-node top-k
   - avg degree `{2, 4, 8, 12, 24}`
4. On the best fixed sparse support, compare:
   - fixed row-normalized weights
   - static learned masked weights
   - dynamic state-conditioned edge weights
5. Add learned adaptive threshold only after fixed sparse support and dynamic
   weights show a positive signal.
6. Prove sparse runtime on LargeST `CA/GBA/GLA` using edge-index/CSR style
   aggregation before making efficiency claims.

## Success Gate

Proceed beyond the first sweep only if:

- LargeST-SD with avg degree <= 8 is within about 1% MAE of the original graph,
  or clearly improves speed/memory at small accuracy cost.
- Dynamic edge weights improve the same fixed sparse support, especially on
  peak/off-peak or high-volatility slices.
- PEMS replication uses only datasets with validated coordinate mappings.

## Links

- Project-level related-work review:
  `/Users/richardo/Desktop/STproject/ADAPTIVE_THRESHOLD_GRAPH_2025_2026_RELATED_WORK.md`
- Earlier idea review:
  `/Users/richardo/Desktop/STproject/ADAPTIVE_THRESHOLD_GRAPH_IDEA_REVIEW.md`
- External planning artifact:
  `/Users/richardo/Desktop/STproject/refine-logs/adaptive-threshold-dynamic-weight/EXPERIMENT_PLAN.md`


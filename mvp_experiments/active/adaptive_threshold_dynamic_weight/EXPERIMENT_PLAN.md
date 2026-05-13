# Experiment Plan

## Scope

Run only on:

- LargeST family: `SD`, `GBA`, `GLA`, `CA`.
- PEMS datasets whose node IDs can be reliably mapped to coordinates.

Exclude datasets without coordinates or physical sensor metadata from main
tables, because they cannot support the distance/physical-threshold motivation.

## Claims

| Claim | Minimum evidence |
|---|---|
| Sparse physical support is enough | Gaussian OSRM global-threshold graphs can match the LargeST-SD original average degree and then trace a controlled degree/accuracy curve around it. |
| Adaptive support beats fixed sparsity | Learned threshold beats fixed global threshold and fixed top-k at equal edge budget. |
| Dynamic edge weights help | On the same fixed support, dynamic weights beat fixed/static learned weights. |
| Efficiency is real | Sparse edge-index/CSR implementation is faster or lower-memory on LargeST-CA/GBA/GLA. |

## Must-Run Blocks

### B0: Graph and Coordinate Diagnostics

- LargeST `SD/GBA/GLA/CA`: coordinate coverage, degree stats, density, isolated rows, row sums, K-hop coverage.
- PEMS: validate node ID to coordinate mapping before using any split.

### B1: Fixed Sparse Pareto on LargeST-SD

Compare under the same GraphWaveNet backbone:

- identity graph
- original graph
- OSRM Gaussian global threshold with target average degree ratios relative to
  the LargeST-SD original graph: `{0.25, 0.5, 1.0, 1.5, 2.0}`
- per-node top-k at matched edge budgets from the same target degrees
- adaptive-only GraphWaveNet reference

Metrics: MAE, RMSE, MAPE, WAPE, edge count, avg degree, train time, inference
throughput, GPU memory.

The LargeST-SD reference graph has 716 nodes and 17,319 off-diagonal directed
edges, so its reference average out-degree is:

```text
k_L = 17319 / 716 = 24.1885
```

For a target ratio beta, set:

```text
k_beta = beta * k_L
E_beta = round(N * k_beta)
```

Build scores from the OSRM distance matrix:

```text
S_ij = exp(-(d_ij^osrm / sigma)^2)
```

Then choose the global score quantile whose top `E_beta` off-diagonal entries
are retained. This avoids hand-picking kilometer thresholds while preserving a
clear comparison against the LargeST original sparsity.

### B2: Dynamic Weights on Fixed Support

Use the best support from B1 and compare:

- fixed row-normalized physical weights
- static learned masked weights
- dynamic state-conditioned weights on active edges

Keep the support identical so the dynamic-weight claim is isolated.

### B3: Learned Adaptive Threshold

At matched edge budgets, compare:

- fixed global threshold
- fixed per-node top-k
- learned per-node threshold / expected degree
- learned support + static weights
- learned support + dynamic weights

### B4: LargeST Sparse Runtime

On LargeST `CA/GBA/GLA`, compare dense masked attention against sparse
edge-index/CSR implementation using the same support where possible.

Do not claim efficiency unless this block is positive.

### B5: Coordinate-Validated PEMS Replication

Run the best method on one validated PEMS dataset. If coordinate mapping is not
reliable, mark the dataset as excluded rather than forcing it into the table.

## First Launch Order

1. `R001`: LargeST graph diagnostics.
2. `R002`: PEMS coordinate validation.
3. `R003`: one-batch LargeST-SD smoke run.
4. `R004/R005`: fixed sparse Pareto on LargeST-SD.
5. `R007/R008`: static vs dynamic weights on the best fixed support.
6. `R009/R010`: adaptive threshold and full two-stage operator.
7. `R011`: sparse runtime proof on LargeST-CA/GBA/GLA.

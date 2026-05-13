# Minimal BasicTS LargeST-15min Preprocessing Experiment Plan

**Date**: 2026-05-13

## Position

This plan is the smallest BasicTS-centered experiment package for the adaptive
threshold + dynamic edge weight idea.

The first scientific question is not whether a full learned graph module can
beat every baseline. The first question is stricter:

> Under the standard LargeST 15-minute BasicTS setting, after controlling input
> features, scaler choice, distance source, and edge budget, does a locally
> adaptive physical sparse graph provide a real accuracy/efficiency advantage
> over fixed global threshold and top-k graphs?

If this answer is negative, the full method should not be built yet.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: preprocessing is controlled | Avoid attributing gains from scaler or time-feature choices to graph construction | A preprocessing-only sweep under the fixed LargeST-15min protocol chooses one default, and all graph variants use it | B0, B1 |
| C2: local adaptive threshold is useful | This is the simplest version of the paper idea | At the same average degree, local-density threshold beats global distance threshold and top-k on SD | B2 |
| C3: road distance matters, or it does not | OSRM is expensive; it needs evidence | OSRM-distance graph beats straight-line graph at matched budget, or we drop OSRM from MVP | B2 |
| C4: dynamic weights add signal beyond support choice | Separates dynamic message weights from graph sparsification | On the same active edge set, dynamic weights improve static weights, especially in peak/high-volatility slices | B3 |
| C5: efficiency is plausible | The motivation includes sparse efficiency | Sparse edge count, runtime, and memory are reported; no efficiency claim if implementation is dense masked | B4 |

## Anti-Claims to Rule Out

- The gain is only caused by using a different scaler or input feature set.
- The gain is only caused by using more edges.
- The gain is only caused by switching from straight-line to OSRM distance.
- Dynamic weights are only learning time-of-day effects already present in input
  features.
- The sparse graph is accurate but implemented densely, so efficiency is not
  actually improved.

## Dataset Scope

### Primary MVP

- `LargeST-SD`, inside BasicTS.
- Use the standard LargeST 15-minute setting only.
- Keep the LargeST input/output horizon, split, and timestamp protocol fixed
  across all graph variants.
- Use one seed for screening, then 3 seeds only for finalists.

### Secondary sanity

- One coordinate-validated PEMS dataset only after SD is clean.
- Prefer `PEMS04F` if its distance matrix and time series are already validated.
- Do not use PEMS in the main claim if node-coordinate provenance is still
  ambiguous.

## BasicTS Backbone Scope

### Must use first

- `GWNet`: strong enough, already graph-sensitive, and already has SD and PEMS
  config variants in BasicTS.

### Use only after positive signal

- `DCRNN`: diffusion-graph confirmation.

### Negative/control baseline

- `STID` or identity/adaptive-only GWNet as graph-weak reference.

Do not start with many backbones. If GWNet shows no graph-side signal, adding
more backbones is likely wasted compute.

## Preprocessing Factors

The key discipline is to avoid a full Cartesian product. We test preprocessing
first, select one default, then test graph construction under that default.

### Temporal protocol

| ID | Variant | Purpose | Keep if |
|---|---|---|---|
| T0 | LargeST 15-minute standard protocol | Fixed task setting | Always fixed; do not compare against 5-minute data in this MVP |
| T1 | target-only input features | Checks whether dynamic weights just copy calendar/time features | Needed for dynamic-weight anti-claim |
| T2 | target + LargeST time features | Practical BasicTS setting | Default if it improves baseline and does not erase graph effects |

### Scaler preprocessing

| ID | Variant | Purpose | Keep if |
|---|---|---|---|
| S0 | BasicTS train-split scaler | Main no-leakage default | Always include |
| S1 | node/channel-wise scaler | Tests sensor heterogeneity handling | Use only if it changes conclusions materially |

Do not use full-series mean/std for method comparison. It is useful only as a
leakage diagnostic.

### Explicitly cut

- No 5-minute experiments in this MVP.
- No 5-to-15 minute resampling comparison.
- No temporal aggregation sweep. The sampling protocol is fixed to the
  LargeST-15min setting.

### Spatial preprocessing

| ID | Variant | Purpose | Keep if |
|---|---|---|---|
| G0 | original LargeST graph | Historical reference | Always include |
| G1 | straight-line global threshold | Cheap physical baseline | Helps decide whether OSRM is needed |
| G2 | OSRM Gaussian global threshold | Road-distance fixed threshold | Main fixed-threshold comparator |
| G3 | OSRM per-node top-k | Strong sparse heuristic | Main non-threshold comparator |
| G4 | OSRM local-density threshold | Minimal adaptive-threshold method | Core MVP method |

All graph variants must be matched by average degree, not by raw kilometer
threshold. Use the current LargeST-SD original graph as the reference:

```text
N = 716
E_L = 17319
k_L = E_L / N = 24.1885
```

For the first global-threshold sweep, build OSRM Gaussian scores:

```text
S_ij = exp(-(d_ij^osrm / sigma)^2)
```

Then select the global score quantile that keeps:

```text
E_beta = round(beta * E_L)
```

off-diagonal directed edges. The initial beta values are `{0.25, 0.5, 1.0,
1.5, 2.0}`. The `beta=1.0` graph is the main fair comparator because it
matches the original LargeST-SD average degree.

## Experiment Blocks

### B0: Data and Metric Sanity

- Claim tested: C1.
- Dataset: LargeST-SD under the 15-minute setting.
- Runs:
  - verify timestamps, split sizes, missing rate, feature channels.
  - verify train-only scaler statistics.
  - verify straight-line and OSRM distance matrix shape/order.
  - verify graph row sums, isolated nodes, average degree, connected components.
- Metrics:
  - coverage, NaN rate, isolated-node count, average degree, density.
- Success criterion:
  - node order is identical across data, metadata, distance matrix, and graph.
  - no graph variant has unexpected isolated rows at degree 4 or 8.
- Failure interpretation:
  - any forecasting run before this is not interpretable.

### B1: Preprocessing-Only Sweep

- Claim tested: C1.
- Backbone: GWNet.
- Graph fixed to the original LargeST 15-minute graph/reference.
- Dataset variants:
  - P0: T0 + S0 + T2.
  - P1: T0 + S0 + T1.
  - P2: T0 + S1 + T2.
- Metrics:
  - MAE, RMSE, MAPE, WAPE, per-horizon MAE, train sec/epoch.
- Success criterion:
  - select one default preprocessing setting before graph experiments.
  - if preprocessing changes MAE more than graph variants later, the paper must
    say the graph claim is secondary.
- Failure interpretation:
  - unstable or large preprocessing effects mean the current BasicTS pipeline is
    not a fair graph testbed yet.

### B2: Minimal Graph Construction Test

- Claims tested: C2, C3.
- Backbone: GWNet.
- Use the selected default from B1.
- Matched budgets:
  - beta `{0.25, 0.5, 1.0, 1.5, 2.0}` relative to LargeST-SD original average
    degree.
- Compared graph supports:
  - G0 original graph.
  - G1 straight global threshold.
  - G2 OSRM Gaussian global threshold.
  - G3 OSRM top-k.
  - G4 OSRM local-density threshold.
- Metrics:
  - prediction: MAE, RMSE, WAPE, per-horizon MAE.
  - graph: edge count, average degree, density, isolated nodes, K-hop coverage.
  - efficiency proxy: sec/epoch, inference throughput, peak GPU memory.
  - diagnostic: performance by local-density quartile.
- Success criterion:
  - G4 beats G2 and G3 at the same degree on MAE or WAPE, or gives similar MAE
    with clearer robustness in sparse/dense regions.
  - OSRM beats straight-line enough to justify its preprocessing cost; otherwise
    use straight-line for MVP.
- Failure interpretation:
  - if G3 top-k is best, the paper should pivot from threshold learning to
    budgeted sparse physical support.
  - if G1 and G2 are indistinguishable, OSRM should not be central.

### B3: Dynamic Weight Isolation

- Claim tested: C4.
- Run only after B2 chooses one support.
- Support fixed:
  - best degree-8 support from B2.
- Compared weights:
  - W0 fixed row-normalized distance weights.
  - W1 static learned masked edge weights.
  - W2 dynamic state-conditioned edge weights on the same edge set.
- Two input settings:
  - with time features.
  - target-only.
- Metrics:
  - MAE, WAPE, peak-period MAE, high-volatility-node MAE.
- Success criterion:
  - W2 improves W0/W1 on the same support.
  - improvement remains at least partly under target-only input.
- Failure interpretation:
  - if W2 only helps with time features, it may be duplicating temporal features,
    not learning meaningful message weights.

### B4: Sparse Efficiency Check

- Claim tested: C5.
- Compare only two systems:
  - original dense/masked implementation if that is what BasicTS currently uses.
  - edge-index/CSR sparse implementation or at least a sparse-support path.
- Dataset:
  - LargeST-SD for correctness.
  - one larger LargeST dataset, preferably GLA or GBA, for scaling.
- Metrics:
  - sec/epoch, inference samples/sec, peak GPU memory, edge count.
- Success criterion:
  - no paper efficiency claim unless runtime or memory improves in the sparse
    implementation.
- Failure interpretation:
  - if the method is algorithmically sparse but implemented densely, report only
    accuracy and graph-budget results.

### B5: Tiny Replication

- Run only after B1-B3 are positive.
- Dataset:
  - one coordinate-validated PEMS dataset.
- Systems:
  - selected preprocessing default.
  - original/reference graph.
  - best fixed sparse graph.
  - best local adaptive threshold graph.
- Purpose:
  - show the SD result is not a one-dataset artifact.

## Run Order

| Milestone | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|
| M0 sanity | B0 diagnostics + one 1-epoch GWNet run | no node-order or scaler issue | < 1 GPU-hour | silent data mismatch |
| M1 preprocessing | B1 P0-P2, one seed | choose one LargeST-15min preprocessing default | 3 short runs | scaler/time features dominate graph effect |
| M2 graph MVP | B2 degree 4/8, one seed | local-density threshold has a real signal | 8-9 runs | no graph variant beats original/top-k |
| M3 dynamic weights | B3 on one support | W2 beats W0/W1 on same support | 4 runs | dynamic module learns time confounds |
| M4 confirmation | finalists with 3 seeds | effect survives variance | 6-9 runs | signal disappears under seeds |
| M5 scaling | B4 + optional B5 | efficiency/replication claim allowed | 2-4 runs | sparse implementation not ready |

## First 9 Concrete Runs

| Run ID | Block | System | Dataset | Purpose | Priority |
|---|---|---|---|---|---|
| MBP001 | B0 | data/graph diagnostics | SD 15min | validate node order, scaler, graph stats | MUST |
| MBP002 | B0 | GWNet 1-epoch smoke | SD 15min, original graph | verify BasicTS training path | MUST |
| MBP003 | B1 | GWNet P0 LargeST-15min standard | SD 15min | baseline preprocessing | MUST |
| MBP004 | B1 | GWNet P1 target-only | SD 15min | test time-feature confound | MUST |
| MBP005 | B1 | GWNet P2 node-wise scaler | SD 15min | test scaler confound | MUST |
| MBP006 | B2 | GWNet straight global deg8 | SD 15min | cheap physical graph reference | MUST |
| MBP007 | B2 | GWNet OSRM global deg8 | SD 15min | road-distance reference | MUST |
| MBP008 | B2 | GWNet OSRM top-k deg8 | SD 15min | strong sparse heuristic | MUST |
| MBP009 | B2 | GWNet OSRM local-density deg8 | SD 15min | minimal adaptive threshold | MUST |

Degree 4 runs are second wave unless degree 8 produces a nontrivial graph-side
signal.

## Table and Figure Targets

- Table 1: preprocessing-only sweep, original graph fixed.
- Table 2: matched-budget graph construction on selected preprocessing.
- Table 3: dynamic weight isolation on the same support.
- Figure 1: MAE vs average degree Pareto curve.
- Figure 2: local-density quartile breakdown.
- Figure 3: edge budget vs runtime/memory, only if sparse implementation is real.

## Critical Stop Rules

- Stop graph-method development if B1 scaler/time-feature effects are larger
  than B2 graph effects and not controlled.
- Stop OSRM expansion if straight-line and OSRM distances yield indistinguishable
  results at matched budgets.
- Stop dynamic-weight development if W2 only helps when explicit time features
  are available.
- Stop efficiency claims if the BasicTS path still materializes dense N-by-N
  tensors.

## Implementation Notes

- Keep all runs inside BasicTS configs; do not create a parallel training stack.
- Generate graph variants as BasicTS datasets or config-level `adj_mx.pkl`
  alternatives so that only graph/preprocessing changes.
- Use W&B online for training logs, but archive only compact summary CSV/JSON in
  this MVP folder.
- Do not commit raw outputs, checkpoints, distance matrices, or prediction dumps.

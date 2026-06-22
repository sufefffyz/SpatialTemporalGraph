# Completion Report: CD-001 - Retrospective adaptive graph archive

## Result

This branch compressed the old adaptive-threshold / adaptive-graph work into a clean research-state summary. The strongest historical evidence is on SD: `GSP K64 + dynamic threshold` improves GWNet and DCRNN over their fixed/candidate baselines, with the best recorded SD checkpoint-style table reporting `GWNet + GSP K64 dynamic + addaptadj` at MAE `17.9169` and `DCRNN + GSP K64 dynamic` at MAE `18.0222`. The evidence is not uniform: METR-LA gains are small and backbone-dependent, PEMS-BAY lacks full fixed/original controls in the current table, LargeST GBA is weak, and KnowAir/CCAQ do not support the dynamic-threshold advantage. The old end-to-end dynamic-threshold story should no longer be treated as the main paper framing; the cleaner direction is a pretrained/static-or-bank graph constructor that exports sparse graphs and cuts gradients during downstream STGNN training.

## Artifacts / Files

- Branch brief: `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/mvp_experiments/active/adaptive_threshold_dynamic_weight/conductor/CD-001_retrospective_archive/branch-brief.md`
- This report: `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/mvp_experiments/active/adaptive_threshold_dynamic_weight/conductor/CD-001_retrospective_archive/completion-report.md`
- Main result snapshot: `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/mvp_experiments/active/adaptive_threshold_dynamic_weight/RESULTS_SNAPSHOT_20260605.md`
- Module discussion README: `/Users/richardo/Desktop/STproject/SpatialTemporalGraph/mvp_experiments/active/adaptive_threshold_dynamic_weight/MODULE_VARIANTS_README_20260608.md`
- Vault idea note: `/Users/richardo/Nutstore Files/我的坚果云/note/research-idea-vault/01_Ideas/Adaptive Distance Budget Graph.md`
- Vault experiment note: `/Users/richardo/Nutstore Files/我的坚果云/note/research-idea-vault/04_Experiments/Adaptive Graph Efficiency Benchmark.md`
- PeMS raw inventory: `/Users/richardo/Desktop/STproject/real-traffic-benchmark/outputs/raw_data_180_d3_d12_ml_hv_20260608/README_raw_data_180_ml_hv.md`

## Confirmed Findings

- SD OSRM beta sweep is useful as analysis, not final method: global threshold controls average degree but creates hub skew; node degree correlates differently with absolute and relative errors, motivating node-aware budgets.
- `GSP K64` is the most useful old candidate-pool evidence on SD. It uses train-signal roughness within an OSRM physical pool, not lag correlation.
- SD no-addapt node analysis: `GWNet + DynamicThreshold + GSP K64` has mean node MAE `18.3442` vs fixed GSP K64 `20.0766`, with `578` nodes better and `118` worse.
- SD addapt node analysis: `GWNet + DynamicThreshold + GSP K64 + addaptadj` has mean node MAE `17.5840` vs original adaptive `18.5390`, with `617` nodes better and `79` worse; this is diagnostic because addaptadj weakens a pure sparse-efficiency claim.
- K64 saturation is a real issue: `GSP K64 + dynamic + addaptadj` has mean active out-degree `59.43`, median `64`, and only small time-of-day degree range `2.66`; full-pair dynamic is more dynamic but too dense.
- Random degree initialization reduced saturation but did not improve accuracy; initialization radius/budget is a strong prior.
- Fixed sparse operators are promising only for fixed graphs so far: SparseDCRNN matched dense DCRNN closely, while SparseGraphWaveNet lost about `0.40` MAE in the beta=1 comparison. Dynamic hard masks have not yet delivered true sparse-training speedup.

## Weak / Tentative Evidence

- METR-LA: GWNet `degree-quantile probe full` is best in the recorded group at MAE `3.0540`, but the gain over fixed OSRM full is only about `0.018` MAE; DCRNN/STGCN do not show stable gains.
- PEMS-BAY: degree-quantile GWNet variants run and report MAE around `1.59`, but the current table lacks enough same-protocol fixed/original controls for a strong claim.
- LargeST GBA: `GWNet + OSRM K64 dynamic no-adaptive` is worse than STID; this does not support large-scale superiority.
- MTGNN: only OSRM K64 dynamic has an effective result; GSP/fullPair replacement attempts need rerun or log audit before use.
- FlowNet migration and PEMS04F results are useful background but should not drive the main adaptive-graph claim.

## Invalid / Do Not Claim

- Old KnowAir/CCAQ 3-channel runs are invalid for air-quality claims because they dropped official covariates.
- KnowAir/CCAQ MAGE-dim results do not support dynamic-threshold advantage; KNN/adaptive/fixed graphs are as good or better.
- Do not claim backbone-agnostic improvement: STGCN is unstable or negative in several settings.
- Do not claim end-to-end dynamic threshold already gives sparse training acceleration.
- Do not treat LargeST-flowonly results as the main strongest protocol; they are sanity/protocol checks and often weaker than older BasicTS settings.

## Current Method Reframe

The old framing was: learn a dynamic threshold module inside STGNN and train it end-to-end. The proposed new framing is: pretrain or estimate a task-aware sensor graph constructor from train-only labels/signals/metadata, export a sparse graph or graph bank, and use it as a frozen downstream graph so sparse graph convolution can be used. This better separates graph preprocessing from backbone training and avoids claiming that dense adaptive adjacency is efficient.

Recommended abstraction:

- Inputs: road/OSRM distance, geodesic distance, road metadata, train-only signal labels such as smoothness or lead-lag scores.
- Output: binary or weighted sparse graph, possibly one graph bank by time-of-day/regime.
- Downstream: GWNet/DCRNN/STGCN/MTGNN consume `stopgrad(A*)`; graph constructor receives no prediction-loss gradient during downstream training.

## PeMS Benchmark State

- The old `PEMS_BENCHMARK_INVENTORY_20260608.md` said raw D4-D12 station files were not found, but later raw scan corrected this: 180 has raw 2025 data under `/data/yuzhang_fei/PEMS` for D3/D4/D5/D6/D7/D8/D10/D11/D12; D9 is missing.
- Raw ML/HV usable counts include D7 `2787`, D8 `1592`, and D12 `1696`, so HOV-rich subsets exist.
- Most districts miss `2025-11-28` and `2025-11-29`; D10 also misses `2025-01-28`. Benchmark construction should avoid or repair these dates.
- Existing D3/D4 BasicTS datasets are trainable, but D3-D12 raw-to-BasicTS subset generation should be rebuilt from the raw `/data` source rather than relying on older graph-construction fallback archives.

## Dependency Outputs

- Unblocks:
  - CD-002: design pretrained graph-constructor labels and graph export protocol.
  - CD-003: implement train-only lag/lead-lag graph label builder and static exported graph variants.
  - CD-004: rebuild raw PeMS D3-D12 ML/HV benchmark subsets from `/data/yuzhang_fei/PEMS`.
- Outputs required by later branches:
  - Historical evidence boundaries and invalid-result list.
  - Method reframe from dynamic end-to-end module to frozen graph preprocessing.
  - PeMS raw source correction and missing-date warning.
- Gate status: report ready for master review; merge requires explicit user approval.

## Decisions Made Locally

- Treat old dynamic-threshold experiments as evidence and diagnostics, not as the final method definition.
- Mark air-quality dynamic-threshold results as negative/weak evidence, not main support.
- Mark GSP K64 as train-signal roughness, not lag correlation.
- Treat PeMS raw `/data/yuzhang_fei/PEMS` as the correct source for future D3-D12 benchmark work.

## Proposed Global Decisions

- Decision: Reposition the project from "end-to-end dynamic threshold plugin" to "pretrained task-aware sparse sensor graph constructor".
- Scope affected: method naming, experiment design, implementation roadmap, paper story, future result tables.
- Why it matters: it aligns with sparse operator acceleration, avoids dense adaptive-adjacency criticism, and separates graph construction from backbone training.
- Branches likely affected: future graph-label builder, pretrained constructor, PeMS benchmark, STGNN downstream evaluation.
- Recommended master-session action: approve this as the new main framing before launching implementation.

- Decision: Keep old `exp_tanh` dynamic threshold and `degree-quantile` runs as ablations/diagnostics, not the mainline.
- Scope affected: result reporting and paper claims.
- Why it matters: old runs are useful but not stable enough across datasets/backbones.
- Branches likely affected: graph-constructor experiments and paper tables.
- Recommended master-session action: merge only the compressed claims above, not the raw old tables.

## Validation / Checks

- Cross-read local result snapshot, module README, vault idea note, vault experiment note, input-feature audit, PeMS inventory, raw PeMS inventory, and selected diagnostic JSON.
- No experiments were rerun.
- No code or model files were modified.

## Risks

- Some recorded numbers come from snapshot tables rather than re-parsing every checkpoint; before paper use, run a separate experiment-audit branch to verify each metric against raw `test_metrics.json`.
- The new graph-pretraining framing is a proposed global decision from the latest master discussion; it still needs user approval before becoming canonical.
- PeMS raw benchmark construction still needs a dedicated implementation branch; the current report only records inventory and constraints.
- Existing vault notes may now be stale because they still emphasize dynamic-threshold/plugin framing.

## Suggested Merge Note

Old adaptive-threshold work should be treated as diagnostic evidence, not the final method. Strongest support is SD: GSP K64 + dynamic threshold improves GWNet/DCRNN, but degree saturates near K64 and cross-dataset/backbone gains are inconsistent. Air-quality results are weak/negative, LargeST GBA does not support superiority, and dynamic hard masks do not yet prove sparse training acceleration. The recommended new framing is a train-only pretrained task-aware sparse sensor graph constructor that exports a frozen graph or graph bank for downstream STGNNs. Future branches should verify raw metrics, design graph-constructor labels, and rebuild PeMS D3-D12 benchmarks from `/data/yuzhang_fei/PEMS`.

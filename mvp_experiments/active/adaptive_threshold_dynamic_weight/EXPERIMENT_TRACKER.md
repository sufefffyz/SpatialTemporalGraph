# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | LargeST graph diagnostics | original / global threshold / top-k | SD/GBA/GLA/CA | coordinate coverage, degree, density, isolated rows, row sums, K-hop coverage | MUST | IN_PROGRESS | Data availability check: `notes/data_availability_2026-05-11.md`; full diagnostics still pending. |
| R002 | M0 | PEMS coordinate validation | PEMS-BAY and PEMS03/04/07/08 if mappable | PEMS | coordinate coverage, matched-node rate | MUST | TODO | Exclude unreliable mappings. |
| R003 | M0 | One-batch smoke | GraphWaveNet original graph | LargeST-SD | forward/backward, finite loss | MUST | TODO | Baseline sanity. |
| R004 | M1 | Fixed sparse Pareto | OSRM Gaussian global threshold beta 0.25/0.5/1.0/1.5/2.0 of LargeST-SD original degree | LargeST-SD | MAE/RMSE/MAPE/WAPE, time, memory | MUST | IN_PROGRESS | Launched GWNet and DCRNN sweeps on 2026-05-13. Threshold is chosen by score quantile; beta 1.0 targets avg degree 24.1885. |
| R005 | M1 | Fixed sparse Pareto | OSRM top-k at matched average degrees | LargeST-SD | MAE/RMSE/MAPE/WAPE, time, memory | MUST | TODO | Compare with global Gaussian threshold at equal edge budget. |
| R006 | M1 | Accuracy reference | OSRM Gaussian global threshold + GraphWaveNet adaptive adjacency | LargeST-SD | MAE/RMSE/MAPE/WAPE | MUST | IN_PROGRESS | Added 5-beta GWNet `addaptadj=True` sweep on 2026-05-13 to compare fixed physical graph vs physical graph plus learned adaptive adjacency. |
| R007 | M2 | Static learned weights | MaskedGraphWaveNet on best fixed support | LargeST-SD | MAE/RMSE/WAPE, time, memory | MUST | TODO | Dense attention caveat. |
| R008 | M2 | Dynamic weights | dynamic sparse-edge weights on same support | LargeST-SD | MAE/RMSE/WAPE, peak/high-volatility MAE | MUST | TODO | Isolate edge-weight claim. |
| R009 | M3 | Adaptive threshold | learned per-node threshold at matched degree | LargeST-SD | MAE/RMSE/WAPE, edge budget, degree stats | MUST | TODO | Core novelty test. |
| R010 | M3 | Full method | adaptive support + dynamic weights | LargeST-SD | MAE/RMSE/WAPE, budget, time, memory | MUST | TODO | Only after R008/R009 promising. |
| R011 | M4 | Sparse runtime proof | dense masked vs edge-index/CSR | LargeST-CA/GBA/GLA | wall-clock, memory, throughput | MUST | IN_PROGRESS | Added `SparseGraphWaveNet` gather-sum MVP and nconv benchmark script; run on GPU before making efficiency claims. |
| R012 | M5 | Stability | top 2 methods, 3 seeds | LargeST-SD + one larger LargeST subset | mean/std metrics | MUST | TODO | Paper-grade evidence. |
| R013 | M5 | Qualitative diagnosis | learned degree/edge weights by region/time | LargeST-SD | degree distribution, peak/off-peak patterns | NICE | TODO | After positive quantitative signal. |
| R014 | M5 | PEMS replication | best method | coordinate-validated PEMS | MAE/RMSE/MAPE/WAPE | MUST | TODO | Main-table replication only if coordinates validated. |

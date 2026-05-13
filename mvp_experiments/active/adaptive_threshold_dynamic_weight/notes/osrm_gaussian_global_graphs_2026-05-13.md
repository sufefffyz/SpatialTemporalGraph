# OSRM Gaussian Global-Threshold Graphs

Date: 2026-05-13

## Decision

For the first fixed-threshold graph sweep, use a global score threshold after
Gaussian-kernel transformation of the precomputed OSRM distance matrix.

Given OSRM distance `d_ij`, compute:

```text
S_ij = exp(-(d_ij / sigma)^2)
```

Then retain the top `E_beta` off-diagonal directed scores, where:

```text
E_L = 17319
N = 716
k_L = E_L / N = 24.18854748603352
E_beta = round(beta * E_L)
```

This keeps the graph budget tied to the current LargeST-SD original graph
instead of hand-picking kilometer thresholds.

## Server Outputs

Generated on the server from the archived SD OSRM distance matrix:

```text
/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/graphs/SD/osrm_gaussian_global/
```

Inputs:

```text
/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/distance_matrices/SD/SD_osrm_shortest_distance_m.npy
/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/distance_matrices/SD/SD_node_ids.csv
/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/datasets/SD/adj_mx.pkl
```

Generated graph files:

| beta | edges | avg degree | threshold distance |
|---:|---:|---:|---:|
| 0.25 | 4,330 | 6.0475 | 3,887.2 m |
| 0.50 | 8,660 | 12.0950 | 5,318.6 m |
| 1.00 | 17,319 | 24.1885 | 7,200.8 m |
| 1.50 | 25,979 | 36.2835 | 8,649.8 m |
| 2.00 | 34,638 | 48.3771 | 9,900.6 m |

The beta 1.0 graph exactly matches the LargeST-SD original off-diagonal edge
count and average degree.

## Note

This is not the same as reproducing the LargeST construction formula with the
4 km geodesic candidate filter and `A_ij >= 0.01`. That formula gives an
estimated OSRM cutoff around 9.57 km under the current matrix. The graph set in
this note instead uses all-pairs OSRM Gaussian scores and chooses a global
quantile to match the target average degree.

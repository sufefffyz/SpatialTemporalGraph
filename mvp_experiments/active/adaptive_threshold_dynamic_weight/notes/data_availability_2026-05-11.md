# Data Availability Check: LargeST SD/GBA/GLA/CA

Date: 2026-05-11
Server repo: `/home/yuzhang_fei/code/SpatialTemporalGraph`

## Question

For the adaptive-threshold + dynamic message-weight experiment, check whether the SD/GBA/GLA/CA data already provide pairwise road-network distances.

## Short Answer

No complete all-pairs road-network distance matrix is currently available for SD/GBA/GLA/CA in the experiment-ready BasicTS/LargeST folders.

What is available is a sparse road-network adjacency/weight matrix:

- SD: `LargeST/data/sd/sd_rn_adj.npy`, also copied as `graphs/SD/adj_mx_largeST_original.pkl` and `BasicTS/datasets/SD_5min_full/adj_mx_largeST_original.pkl`
- GBA: `graphs/GBA/adj_mx_largeST_original.pkl`, `BasicTS/datasets/GBA_5min_full/adj_mx_largeST_original.pkl`
- GLA: `graphs/GLA/adj_mx_largeST_original.pkl`, `BasicTS/datasets/GLA_5min_full/adj_mx_largeST_original.pkl`
- CA: `/data/yuzhang_fei/LargeST/ca_rn_adj.npy` exists outside the repo, but no BasicTS-ready `CA_5min_full` directory exists yet

These matrices contain 0-1 sparse edge weights, not metric distances.

## Matrix Summary

| Dataset | Nodes | Available matrix | Nonzero off-diagonal density | Value range | Interpretation |
|---|---:|---|---:|---|---|
| SD | 716 | `sd_rn_adj.npy` / `adj_mx_largeST_original.pkl` | 0.033830 | 0.0100-1.0 | Sparse road-network adjacency weight |
| GBA | 2352 | `adj_mx_largeST_original.pkl` | 0.011076 | 0.0100-1.0 | Sparse road-network adjacency weight |
| GLA | 3834 | `adj_mx_largeST_original.pkl` | 0.006716 | 0.0100-1.0 | Sparse road-network adjacency weight |
| CA | 8600 | `/data/yuzhang_fei/LargeST/ca_rn_adj.npy` | 0.002723 | 0.0100-1.0 | Sparse road-network adjacency weight |

The LargeST notebooks derive SD/GBA/GLA road-network adjacency by slicing `ca_rn_adj.npy` with dataset-specific `ID2` indices.

## Metadata Availability

All four datasets have station metadata with coordinates and road attributes:

- `ID`
- `Lat`
- `Lng`
- `District`
- `County`
- `Fwy`
- `Lanes`
- `Type`
- `Direction`
- `ID2`

Available metadata paths:

- `BasicTS/datasets/SD_5min_full/meta.csv`
- `BasicTS/datasets/GBA_5min_full/meta.csv`
- `BasicTS/datasets/GLA_5min_full/meta.csv`
- `PatchSTG/data/CA/ca_meta.csv`

## Related Distance Artifacts

There is a separate PEMS D03 graph-construction artifact with local/OSRM distance matrices:

- `BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_osrm_dist_matrix.pkl`
- shape: `1439 x 1439`
- values look like meters, capped around 5000
- sparse/local candidate matrix, not full all-pairs

This artifact does not directly match the current LargeST SD 716-node dataset.

## Implication for the MVP

For the first adaptive-threshold experiment, the most direct path is:

1. Use the existing sparse road-network adjacency weight matrices as the fixed reference graph.
2. Use metadata coordinates to construct a controlled distance baseline, e.g. haversine/euclidean distance with adaptive top-k or target sparsity.
3. Treat exact road-network all-pairs shortest-path distance as a later extension, because it is not already available for SD/GBA/GLA/CA in experiment-ready form.
4. Exclude CA from the first BasicTS MVP unless a CA BasicTS dataset is generated; start with SD/GBA/GLA.


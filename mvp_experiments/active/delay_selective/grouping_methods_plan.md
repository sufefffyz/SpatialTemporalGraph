# Grouping Methods for Delay-Selective Validation

This note summarizes practical grouping choices for city-traffic-M/L and how
each choice relates to prior work found through Zotero semantic search.

## Zotero Sources Checked

| Paper | Zotero item | What matters here |
| --- | --- | --- |
| Rethinking Sensors Modeling: Hierarchical Information Enhanced Traffic Forecasting | `LU9N9XRT` | Builds regional nodes by merging sensors with high intra-region correlation, then adds global pattern nodes. Useful baseline, but similarity-based grouping is already prior art. |
| Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting | `YTJUWJPU` | Learns cluster assignments/pyramidal hierarchy from time-series relationships. Useful for a learned grouping baseline. |
| MillGNN: Learning Multi-Scale Lead-Lag Dependencies for Multi-Variate Time Series Forecasting | `SPQT62ST` | Direct collision risk for generic multi-scale group lead-lag modeling. Our grouping should be road-topology constrained and delay-selective, not only time-series similarity based. |
| MultiSPANS | `7NAWQS8P` | Uses road-network hierarchy through structural entropy optimization. Useful precedent for topology-derived hierarchy. |
| STDDE | `HCTF3T7P` | Explicit edge-level traffic delay. Our gap should be where not to learn edge delays and at what grouping scale delay is observable. |
| Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks | `TA6WEB6U` | city-traffic-M/L nodes are road segments and directed edges are permitted road-adjacency movements. Metadata gives natural grouping baselines. |
| DADiffNet | `FAZCBTWS` | Adaptive subgraphs plus delay-aware diffusion. Treat as collision risk; Zotero date is 08/2026, so verify bibliographic status before relying on it. |

## Main Conclusion

There are several existing ways to make groups. For this project, grouping
itself should not be claimed as the novelty. The defensible novelty is:

```text
Use road-topology-constrained groups to separate mostly synchronous local
relations from delayed inter-group propagation, then learn delay only where it
is observable and stable.
```

That means time-series clustering and generic lead-lag grouping should be
baselines, while the proposed path should combine road topology, metadata,
distance/direction, and delay-observability tests.

## Candidate Group Families

### 1. Metadata Groups

Use existing static attributes from city-traffic:

```text
region_id, category, edge_type, speed_mode, speed_limit
```

Recommended role: cheap baseline and sanity check.

Pros:
- No extra algorithm.
- Directly available in `spatial_node_features`.
- `region_id` is especially useful as a pre-existing regional grouping.

Cons:
- Road type groups are often disconnected and not physical propagation units.
- A `category=1.0` subgraph is not a proper group hierarchy by itself.

### 2. Road-Topology Community Groups

Build a weighted graph from the road adjacency, then partition it with a
community or graph-partitioning algorithm.

Practical algorithms:

```text
Louvain / Leiden community detection
Infomap for directed flow-like communities
METIS / KaHIP balanced graph partitioning
connected components after pruning weak edges
```

Suggested edge weight:

```text
w_ij =
  1.0 * road_adjacency
+ 0.5 * same_category
+ 0.5 * same_speed_mode
+ 0.2 * similar_speed_limit
+ distance_decay(centroid_distance)
```

Recommended role: first serious grouping method.

Pros:
- Matches the road-network nature of city-traffic-M/L.
- Gives contiguous or near-contiguous regions.
- Much less collision with MillGNN than pure time-series grouping.

Cons:
- Plain Louvain/Leiden does not guarantee balanced group sizes.
- Directed turn constraints may need simplification to an undirected graph for
  partitioning, then restored when building inter-group edges.

### 3. Corridor / Chain Groups

Group contiguous road segments that have similar road type, direction, speed
limit, and geometry bearing.

Approximate bearing from:

```text
x_coordinate_start, y_coordinate_start, x_coordinate_end, y_coordinate_end
```

Recommended role: physically interpretable grouping for delay propagation.

Pros:
- Strongly aligned with queue/wave propagation along roads.
- Helps make intra-group relations local and near-synchronous.

Cons:
- Harder in dense urban networks with intersections, turns, parallel roads,
  and short segments.
- Needs careful handling of turn movements and disconnected fragments.

### 4. Time-Series Similarity Groups

Cluster nodes by traffic profiles:

```text
daily mean profile correlation
Pearson/Spearman correlation on residualized speed
DTW distance on daily templates
spectral clustering on similarity graph
k-means on profile embeddings
```

Recommended role: important baseline, not the main novelty.

Pros:
- Directly related to HIEST and graph-based hierarchical forecasting.
- Can expose latent functional regions.

Cons:
- High collision risk with HIEST/MillGNN-style grouping.
- May group far-away roads with similar rush-hour profiles but no physical
  propagation path.

### 5. Delay-Observability Refined Groups

Start with topology/community/corridor groups, then refine them using the delay
audit results:

```text
1. Keep edges inside a group when lag 0 is dominant or non-zero lag is unstable.
2. Prefer inter-group edges when non-zero lag is high-confidence and stable.
3. Penalize group assignments that put many stable non-zero-delay edges inside
   the same synchronous group.
4. Penalize group assignments that create inter-group edges with no measurable
   delay signal.
```

Recommended role: best candidate for our proposed method.

This turns grouping into part of the research gap:

```text
Delay should be modeled only after checking whether the delay is observable at
the current sampling interval and spatial scale.
```

## Recommended First Experiment Order

1. `region_id` groups.
   Use this as the easiest metadata baseline.

2. Road-topology Louvain/Leiden groups.
   Symmetrize the directed road graph for partitioning, then recover directed
   inter-group edges for delay analysis.

3. Size-matched random groups.
   Negative control. If random grouping performs similarly, grouping has not
   captured meaningful traffic structure.

4. Time-series profile groups.
   Baseline against HIEST/MillGNN-like similarity grouping.

5. Delay-observability refined topology groups.
   Proposed route: intra-group synchronous, inter-group delay-selective.

## Group Quality Metrics

For each grouping, report:

```text
num_groups
group_size_mean / median / p90 / max
intra_edge_ratio
inter_group_edge_count
inter_group_degree_distribution
intra_group_near_zero_lag_ratio
intra_group_high_conf_nonzero_ratio
inter_group_high_conf_nonzero_ratio
best_lag_stability_across_days
best_lag_vs_distance_by_group_pair
```

The grouping supports the paper's claim if:

```text
intra-group edges are mostly zero-lag or unstable-delay,
inter-group edges contain a higher share of stable non-zero delay,
and this pattern is stronger than metadata-only, random, and pure time-series
similarity groupings.
```

## Claim Boundary

Avoid claiming:

```text
We propose group-level lead-lag modeling.
```

Safer claim:

```text
We study delay observability across road-network scales and use
topology-constrained groups to suppress spurious near-neighbor delays while
preserving delayed inter-group propagation.
```

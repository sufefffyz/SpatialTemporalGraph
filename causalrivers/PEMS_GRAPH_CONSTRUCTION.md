# PEMS Graph Construction Notes

This note turns the earlier feasibility discussion into a concrete workflow for building a PeMS graph that is compatible with the local `causalrivers` pipeline.

## Core idea

PeMS metadata does not contain a verified graph. That means we should not treat the metadata file as if it were the final adjacency. The safer workflow is:

1. Build a metadata-only skeleton graph.
2. Use external road-network information to validate and refine that skeleton.
3. Export the reviewed graph into the `causalrivers` product format.

This is close in spirit to the original CausalRivers pipeline:

- raw data normalization
- metadata unification
- external knowledge / map matching
- quality markers
- subgraph sampling for benchmarking

## New lightweight skeleton builder

This repository now includes:

- [`build_pems_metadata_skeleton.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/causalrivers/build_pems_metadata_skeleton.py)

It does **not** create a final `graph.p`. Instead it creates:

- `nodes.csv`
- `edges.csv`
- `suspicious_edges.csv`
- `manifest.json`

The default behavior is intentionally conservative:

- keep `ML` sensors only
- group by `Fwy` and `Dir`
- sort by `Abs_PM`
- connect adjacent sensors along the travel direction
- flag large postmile jumps or large geographic jumps for external review

This gives a reviewable backbone, not a claimed ground truth.

## Example

```bash
python SpatialTemporalGraph/causalrivers/build_pems_metadata_skeleton.py \
  --meta-dir data/PEMSD3/station_meta \
  --reference-date 2025-06-01 \
  --sensor-types ML \
  --output-dir SpatialTemporalGraph/causalrivers/pems_metadata_skeleton
```

## How to use the outputs

`nodes.csv`

- cleaned sensor metadata for the chosen snapshot
- good starting point for node attributes in a later graph export

`edges.csv`

- candidate directed edges from metadata ordering
- every edge is marked as `metadata_abs_pm_chain`

`suspicious_edges.csv`

- edges whose postmile jump or geodesic jump is large
- these should be validated against SHN / OSM / OSRM or removed

`manifest.json`

- records the snapshot date and thresholds used

## Recommended refinement stages

### Stage 1: Metadata-only skeleton

Use the new script to get a first-pass backbone.

### Stage 2: External road validation

Use external map sources to validate the candidate edges:

- Caltrans SHN
- OSM centerlines
- OSRM shortest-path checks

The existing exploratory scripts under [`BasicTS/PEMS/graph_construction`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction) are relevant here, especially:

- [`network_graph_pipeline2.6.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/network_graph_pipeline2.6.py)
- [`build_sensor_graph.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_sensor_graph.py)

### Stage 3: Quality markers

Keep explicit edge provenance:

- `quality_pm`: metadata ordering quality
- `quality_geo`: coordinate availability quality
- `quality_external`: external validation quality
- `edge_origin`: how the edge was created

This mirrors the design pattern used in the original rivers dataset.

### Stage 4: Export for causalrivers

Once the graph is reviewed:

1. create a directed graph whose node IDs match the time-series columns
2. save it as `graph.p`
3. save the aligned time series as `targets.npy` plus `unix_timestamps.npy`
4. sample subgraphs with the existing `causalrivers` tooling

## Important caveat

Even after road-network validation, this graph is still a **structural traffic graph**, not a causal ground-truth graph in the same sense as the rivers benchmark. It is still useful for:

- pipeline debugging
- benchmarking behavior on structured traffic systems
- comparing sampling strategies and baseline robustness

But the interpretation of scores should stay separate from the original CausalRivers leaderboard claims.

# PEMS Graph Script Review

This note records the current shortcomings of the `BasicTS/PEMS/graph_construction` pipeline before integrating anything into the local `causalrivers` workflow.

## Main findings

### 1. The checked-in runner uses a stale metadata snapshot

[`build_graph.sh`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_graph.sh) still points to `d03_text_meta_2024_11_27.txt`, while the available raw traffic sequence under `data/PEMSD3/station_raw` is from June 2025.

Implication:

- sensor additions / removals / relocations after 2024-11-27 are invisible to the default build
- even a correct matching algorithm will inherit avoidable temporal mismatch error

### 2. The pipeline treats metadata as a single static snapshot

[`network_graph_pipeline2.6.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/network_graph_pipeline2.6.py) loads one metadata file and uses it for the whole build.

Relevant lines:

- metadata loading: lines 200-277

Implication:

- no handling of station churn across time
- graph quality depends strongly on which snapshot was chosen

### 3. ML matching uses a very local nearest-edge heuristic

Relevant lines in [`network_graph_pipeline2.6.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/network_graph_pipeline2.6.py):

- ML matching: lines 1001-1133

Current behavior:

- choose same-`Fwy`, same-`Dir` candidate edges
- project the sensor point to every candidate edge geometry
- keep the nearest edge if the point-to-line distance is below `1e-4` degrees

Main risk:

- this is still a nearest-geometry assignment, not a route-consistent assignment
- in dense interchanges and parallel facilities, the closest line can still be the wrong line
- the degree-based threshold is fragile and not very interpretable

### 4. OR / FR ramp matching is node-nearest, not ramp-geometry-nearest

Relevant lines in [`network_graph_pipeline2.6.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/network_graph_pipeline2.6.py):

- ramp extraction and matching: lines 1139-1332

Current behavior:

- first collect ramp links adjacent to mainline nodes
- then match OR/FR sensors by nearest ramp node in the same `Fwy` and `Dir`

Main risk:

- nearest-node is much weaker than nearest-ramp-geometry
- interchange clusters can easily produce the wrong ramp assignment

### 5. FF sensors are explicitly left for manual work

Relevant lines in [`network_graph_pipeline2.6.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/network_graph_pipeline2.6.py):

- FF manual output: lines 1338-1383

Implication:

- the current automated pipeline is knowingly incomplete
- any realistic graph build plan must budget manual review time for FF from the start

### 6. Phase 6 converts incidence into a symmetric adjacency

Relevant lines in [`network_graph_pipeline2.6.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/network_graph_pipeline2.6.py):

- adjacency construction: lines 1407-1416

Current behavior:

- for each tail-head pair, both `adj[ti][hi]` and `adj[hi][ti]` are set to 1

Main risk:

- this destroys directionality at the adjacency level
- downstream users may believe they are getting a directed road graph when they are not

### 7. `build_sensor_graph.py` enumerates all geometric candidate pairs

Relevant lines in [`build_sensor_graph.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_sensor_graph.py):

- candidate pair generation: lines 343-372

Current behavior:

- all sensor pairs are enumerated
- only then are they filtered by geometric threshold

Main risk:

- complexity scales quadratically in sensor count
- adding OR/FR to ML quickly increases runtime and OSRM query volume

### 8. `build_sensor_graph.py` does not restrict candidate pairs to same freeway or direction

Relevant lines:

- same-fwy / same-dir are recorded but not enforced: lines 353-365

Main risk:

- sensors on nearby but different facilities can still enter the candidate set
- route-distance heuristics may then create spurious cross-facility sensor edges

### 9. The OSRM graph is masked by the local graph, so OSRM cannot fix local mistakes

Relevant lines in [`build_sensor_graph.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_sensor_graph.py):

- OSRM mask: lines 499-543

Current behavior:

- OSRM-directed edges are removed unless they already appear in the local raw edge set

Main risk:

- the stronger external routing check loses the ability to correct local graph mistakes
- this makes the two-stage design conservative in the wrong direction

### 10. Triangle pruning is too topology-agnostic

Relevant lines in [`build_sensor_graph.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_sensor_graph.py):

- transitive pruning: lines 546-600

Current behavior:

- whenever `a->b`, `b->c`, and `a->c` exist, remove the longest edge

Main risk:

- valid skip links or interchange-related edges can be deleted
- no check for sensor type, branch structure, or road semantics

### 11. Embedded sensor nodes are always labeled as `ML`

Relevant lines in [`build_sensor_graph.py`](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/BasicTS/PEMS/graph_construction/build_sensor_graph.py):

- node insertion: lines 262-271

Current behavior:

- `sensor_type="ML"` is hardcoded during graph embedding

Main risk:

- once OR / FR are included, the embedded graph node attributes become internally inconsistent

## What the lightweight metadata skeleton says

Using the new metadata-only skeleton builder on the `2025-04-26` snapshot for `ML` sensors produced:

- 880 nodes
- 840 candidate backbone edges
- 153 suspicious edges needing external validation

The suspicious edges are concentrated on:

- `80 E` / `80 W`
- `5 N` / `5 S`
- `99 N` / `99 S`
- `50 E` / `50 W`

That pattern is consistent with the expectation that dense, long, and interchange-heavy corridors dominate review effort.

## Rough manual review cost

Snapshot counts from `d03_text_meta_2025_04_26.txt`:

- `ML`: 881
- `OR`: 423
- `FR`: 276
- `FF`: 28

### Minimal engineering audit

Scope:

- review 153 suspicious ML backbone edges
- review all 28 FF sensors manually
- spot-check a sample of “non-suspicious” ML edges

Estimated cost:

- suspicious ML edges: `153 * 2-4 min` ≈ `5-10 person-hours`
- FF sensors: `28 * 8-15 min` ≈ `4-7 person-hours`
- spot checks / bookkeeping / reruns: `3-6 person-hours`

Total:

- about `12-23 person-hours`

### Safer benchmark-grade audit

Scope:

- the minimal audit above
- plus targeted review of a substantial OR/FR subset in dense interchanges
- plus correction reruns and diff checks

Estimated cost:

- minimal audit: `12-23 person-hours`
- OR/FR targeted review:
  - if reviewing 15-25% of 699 ramp sensors at `2-5 min` each
  - ≈ `4-15 person-hours`
- reruns and comparison passes: `6-12 person-hours`

Total:

- about `22-50 person-hours`

### Full publication-grade manual sweep

Scope:

- review every suspicious ML edge
- review every FF sensor
- review nearly all OR/FR assignments in major interchanges
- perform at least one full re-run after overrides

Estimated cost:

- about `40-80 person-hours`

This is the range I would use for planning if the goal is a graph that we are comfortable benchmarking against repeatedly.

## Practical conclusion

The current scripts are useful as candidate generators, but not yet trustworthy as a one-shot graph builder. The highest-leverage next step is not `causalrivers` integration. It is:

1. freeze a correct metadata snapshot policy
2. validate the ML backbone first
3. only then decide how much OR/FR/FF detail is worth carrying into the benchmark graph

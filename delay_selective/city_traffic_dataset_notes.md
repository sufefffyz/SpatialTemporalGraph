# City-Traffic Dataset Notes

Source paper from Zotero:

```text
Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks
Fedor Velikonivtsev, Oleg Platonov, Gleb Bazhenov, Liudmila Prokhorenkova
arXiv:2510.02278, DOI: 10.48550/arXiv.2510.02278
Zotero item: TA6WEB6U
```

## Paper Context

The paper introduces metropolis-scale road-segment traffic forecasting datasets. A node is an individual road segment, and a directed edge connects two road segments when they are incident and the movement is permitted by traffic rules. This matters for the delay-selective route because the graph is closer to a road-link graph than PeMS-style detector graphs.

For `city-traffic-M`, the paper reports:

| Field | Value |
| --- | ---: |
| Nodes | 53,530 |
| Edges | 121,236 |
| Timestamps | 35,449 |
| Train timestamps | 26,208 |
| Validation timestamps | 4,032 |
| Test timestamps | 5,209 |
| Granularity | 5 minutes |
| Period | July 1, 2024 to November 1, 2024 |
| Time zone | UTC+5 |

The target variables are speed and volume estimated from GPS traces. Speed can be missing when no GPS vehicle passes a segment in a 5-minute period; volume is zero rather than missing in such windows.

The paper lists 26 static road attributes:

```text
category, edge_type, speed_mode, speed_limit, region_id,
can_bind_to_reverse_edge, dismount_bike, has_masstransit_lane,
ends_with_crosswalk, ends_with_toll_post, is_in_poor_condition,
is_paved, is_restricted_for_trucks, is_toll, access_[0...5],
length, num_segments, x_coordinate_start, y_coordinate_start,
x_coordinate_end, y_coordinate_end
```

## Important Caveat For Current Local Files

The current local files:

```text
data/city_traffic_m_speed__category__1_0.npz
data/city_traffic_m_volume__category__1_0.npz
```

are category-specific subgraphs that were previously cut by road type. They should not be treated as the official full `city-traffic-M` graph. The profiler therefore checks observed node/edge counts against the paper's full-graph counts and writes warnings when `subgraph_*` metadata or `__category__` filenames are detected.

For the first server sweep, prefer the full files if they exist:

```text
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_speed.npz
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz
```

Then compare them against the `category=1.0` subgraphs to decide whether that cut is meaningful for the delay-validation route.

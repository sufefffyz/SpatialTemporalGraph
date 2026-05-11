# OSRM Runtime Check

Date: 2026-05-11
Server: `ssh -p 5102 yuzhang_fei@183.174.228.180`
Repo: `/home/yuzhang_fei/code/SpatialTemporalGraph`

## Result

Local OSRM can run on the server through Docker.

Available runtime:

- Docker: `28.1.1`
- Image: `ghcr.io/project-osrm/osrm-backend:latest`
- No host-level `osrm-routed`, `osrm-extract`, `osrm-partition`, or `osrm-customize` binaries were found.

## Smoke Test

Input map:

```text
BasicTS/PEMS/graph_construction/output_d03_2.7_full/d3_motorway.osm
```

The following Docker OSRM pipeline succeeded:

```bash
docker run --rm -v "$tmp:/data" ghcr.io/project-osrm/osrm-backend:latest \
  osrm-extract -p /opt/car.lua /data/d3_motorway.osm

docker run --rm -v "$tmp:/data" ghcr.io/project-osrm/osrm-backend:latest \
  osrm-partition /data/d3_motorway.osrm

docker run --rm -v "$tmp:/data" ghcr.io/project-osrm/osrm-backend:latest \
  osrm-customize /data/d3_motorway.osrm

docker run -d --rm --name stg_osrm_smoke -p 127.0.0.1:5010:5000 -v "$tmp:/data" \
  ghcr.io/project-osrm/osrm-backend:latest \
  osrm-routed --algorithm mld --max-table-size 1000000 /data/d3_motorway.osrm
```

Route API probe succeeded:

```text
GET /route/v1/driving/-121.48499004,38.41555904;-121.48825308,38.43130170?overview=false
distance = 1770.8 m
duration = 76.2 s
```

Table API probe succeeded:

```text
GET /table/v1/driving/...?...&annotations=distance
distances = [[0, 1770.8], [117759.7, 0]]
```

The asymmetric distance is expected because OSRM respects road direction.

## New Builder Script Test

Script:

```text
mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/build_distance_matrices.py
```

The script successfully generated both matrices for a 4-node D03 OSRM smoke test:

```bash
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
  mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/build_distance_matrices.py \
  --dataset D03_OSRM_SMOKE \
  --meta BasicTS/PEMS/graph_construction/output_d03_2.7_full/sensor_graph/phase7_osrm_sensor_info.csv \
  --output-dir /tmp/stg_osrm_script_smoke \
  --compute straight,osrm \
  --limit 4 \
  --osrm-url http://127.0.0.1:5011 \
  --osrm-block-size 2 \
  --overwrite
```

Generated:

- `D03_OSRM_SMOKE_straight_distance_m.npy`
- `D03_OSRM_SMOKE_osrm_shortest_distance_m.npy`
- `D03_OSRM_SMOKE_node_ids.csv`
- `D03_OSRM_SMOKE_distance_summary.json`
- `D03_OSRM_SMOKE_osrm_progress.jsonl`

## Implications

For SD/GBA/GLA/CA, we can use the same script once an OSRM map covering the target coordinates is prepared and served locally.

Recommended next step:

1. Use `/data/yuzhang_fei/LargeST/ca_rn_adj.npy` and CA metadata only as graph/reference data.
2. Prepare a California-wide OSRM dataset from `BasicTS/PEMS/graph_construction/california-latest.osm.pbf` or a verified larger OSM/PBF source.
3. Start OSRM with a large table limit.
4. First run SD all-pairs matrices, then GBA/GLA if runtime is acceptable.


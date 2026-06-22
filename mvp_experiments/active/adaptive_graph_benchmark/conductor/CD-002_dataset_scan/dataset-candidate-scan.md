# Dataset Candidate Scan: Adaptive Graph Internal Benchmark

Date: 2026-06-10

## Scope

The benchmark is internal infrastructure, not the paper innovation. It should be traffic-first and deliberately compact: the current goal is not to cover many datasets, but to build a clean testbed that can compare several mainstream adaptive graph construction types under controlled metadata and preprocessing.

LargeST GLA/GBA, Q_TRAFFIC, and broad urban-mobility datasets are intentionally deferred because they are too expensive or too semantically broad for the current iteration.

## Current Narrow Protocol

Use our own PeMS 2025 subsets as the main experimental field, with SD 15min as the continuity/control dataset. Do not expand into a large dataset zoo.

| Role | Dataset | Use now? | Reason |
|---|---|---|---|
| Main mechanism benchmark | Raw PeMS 2025 district subsets | yes | coordinate-complete, controllable node type/time subset, traffic-flow target, good for graph diagnostics |
| Continuity benchmark | SD 15min | yes | already used throughout the project; medium scale; keeps old findings comparable |
| Optional speed sanity | METR-LA or PEMS-BAY | optional later | standard speed benchmark, but not necessary for the first mechanism pass |
| Optional road-link extension | EXPY-TKY | defer | strong metadata, but 1843/2841 links and 10min protocol add extra preprocessing and compute |
| Urban mobility extension | NYCTAXI / TaxiBJ / Foursquare-TKY | defer | useful for generalization, but graph semantics differ from road-sensor/road-link forecasting |

The first pass should compare classic adaptive/spatial-dependency mechanisms, not our previous candidate-threshold idea and not a large dataset zoo.

| Mechanism type to cover | Representative model / variant | Why include |
|---|---|---|
| Fixed physical graph baseline | DCRNN or GWNet without adaptive adjacency | tells us how much the given road/sensor graph already explains |
| Dense node-embedding adaptive adjacency | GWNet `addaptadj=True` | canonical adaptive adjacency baseline; simple and widely reused |
| Node-adaptive graph convolution | AGCRN | represents node-embedding-conditioned graph convolution rather than only learned adjacency |
| Directed top-k graph learning | MTGNN graph constructor | represents learned sparse directed graph construction |
| Decoupled/dynamic graph learning | D2STGNN | optional second-stage model; useful but harder to isolate |
| Graph-free spatial identity embedding | STID | strong classic control: tests whether node/time identity embeddings already replace explicit graph learning |
| Transformer with spatio-temporal adaptive embeddings | STAEformer | optional strong control; tests whether adaptive embeddings plus attention dominate explicit graph design |

Do not include candidate-constrained GWNet in this line. That belongs to the previous adaptive-threshold/candidate-pool idea.

## Recommended First Wave

| Priority | Dataset | Domain | Scale / local evidence | Spatial metadata | Why use it |
|---|---|---|---|---|---|
| 1 | Raw PeMS 2025 D3-D12 subsets | traffic flow | server inventory shows D3/D4/D5/D6/D7/D8/D10/D11/D12 raw station data and 5min files | raw PeMS metadata has lat/lon/postmile/type; local scan already counted usable coordinates | best main field: coordinate-complete, controllable, traffic-flow target, good for mechanism analysis |
| 2 | SD 15min | traffic flow | local `BasicTS/datasets/SD`, shape `[35040, 716, 3]` | local `meta.csv` with coordinates | continuity/control dataset for comparison with previous project results |
| 3 | METR-LA or PEMS-BAY | traffic speed | BasicTS configs exist | DCRNN data release includes sensor locations | optional single speed sanity dataset only if needed |

## Second Wave / Optional

| Dataset | Domain | Why maybe useful | Main risk |
|---|---|---|---|
| PeMSD4 / PEMS04 | traffic flow / occupancy / speed | small mainstream flow dataset, common BasicTS node count is 307 | canonical ASTGCN release only has `distance.csv` and `pems04.npz`; no coordinates found |
| PeMSD8 / PEMS08 | traffic flow / occupancy / speed | smallest mainstream PeMS flow dataset, common BasicTS node count is 170 | canonical ASTGCN release only has `distance.csv` and `pems08.npz`; no coordinates found |
| PeMSD3 | traffic flow | LibCity lists 358 sensors and flow information | source points to STSGCN; coordinates not verified |
| PeMSD7 | traffic flow | LibCity lists 883 sensor stations | source points to STSGCN; coordinates not verified |
| Madrid M-DENSE | traffic intensity | LibCity lists 15min traffic intensity in cars/hour | `sp_data.csv` is a spatial-feature matrix without obvious lon/lat header; coordinate completeness not verified |
| PEMS04F | traffic flow | local BasicTS dataset exists, 307 nodes, 5min | coordinate and distance provenance must be rechecked; use only if it matches PeMSD4 node order |
| Rotterdam | traffic state | LibCity lists 208 links; small enough if data is accessible | repository link does not directly expose raw data/coordinates in the top-level file list |
| LOOP_SEATTLE | traffic speed | 323 freeway loop detector stations in Greater Seattle; closer to road-sensor graph than taxi/grid datasets | public repo listing exposes README/images only; need verify raw download and detector coordinates before use |
| SZ_TAXI | road traffic speed from taxi trajectories | small Shenzhen road-speed dataset used by T-GCN; has speed and adjacency files | T-GCN public data directory exposes `sz_speed.csv` and `sz_adj.csv`, but no coordinate file; use only after road/coordinate metadata is recovered |
| Q_TRAFFIC / BaiduTraffic | Beijing road-segment speed | 15,073 road segments, 15min sampled speed, road network sub-dataset, query auxiliary data | very large and preprocessing-heavy; check raw package accessibility and whether graph metadata maps cleanly to forecasting samples |
| Weather2K | weather | public multi-station weather dataset with station-like spatial structure | not yet integrated; need verify download format and target variables |
| Bike/CitiBike style station-demand data | bike demand | station coordinates usually complete and download is easy | not traffic sensors; graph semantics differ from road traffic propagation |

## Urban Mobility Branch / Optional

These can enter the internal benchmark if we explicitly evaluate whether the graph learner generalizes beyond road sensors. They should be reported separately from freeway sensor / road-link traffic.

| Dataset | Domain | Why maybe useful | Main risk |
|---|---|---|---|
| NYCTAXI / NYCTAXI_DYNA / NYCTAXI_OD / NYCTAXI_GRID | taxi trajectory, regional inflow/outflow, OD, or grid demand | large and easy to motivate as urban mobility; coordinates or grid/region geometry can support graph construction | graph semantics are region/OD demand, not physical sensor message passing |
| NYC_TOD | NYC grid-based inflow/outflow | CSTN-style grid demand benchmark | same grid-demand semantic gap as NYCTAXI_GRID |
| TaxiBJ | Beijing taxi crowd flow with meteorology/holiday covariates | classic crowd-flow benchmark, useful if comparing grid-based urban flow | grid cells, not sensors or road links |
| Foursquare-TKY | Tokyo check-in / POI mobility | Tokyo-related urban mobility dataset | check-in/POI semantics; not the Tokyo expressway sensor dataset |

## Not Recommended For First Wave

| Dataset | Reason |
|---|---|
| LargeST GLA/GBA | too large for rapid iteration right now |
| PEMS03/04/07/08 standard BasicTS-only raw folders | acceptable for training but not enough for graph diagnosis unless `.geo` / station-id / coordinate mapping is verified; current evidence suggests coordinates are not included in the common releases |
| BeijingAirQuality in BasicTS | local script has no graph file and likely too few stations for learned-graph mechanism claims |
| Generic Weather / Electricity LTSF datasets | no meaningful sensor coordinate graph in the default BasicTS form |
| CCAQ / China City AQI | useful later, but not needed if KnowAir is already the single non-traffic sanity dataset |
| Seattle map-matching GPS dataset | single GPS trajectory for map matching, not a traffic forecasting sensor benchmark |

## PeMS Coordinate Audit

Current evidence supports the user's concern: the common PeMSD3/4/7/8 benchmark releases should not be treated as coordinate-complete.

| Dataset | Evidence | Coordinate status |
|---|---|---|
| PeMSD4 / PEMS04 | ASTGCN GitHub data directory contains only `distance.csv` and `pems04.npz` | no coordinates found |
| PeMSD8 / PEMS08 | ASTGCN GitHub data directory contains only `distance.csv` and `pems08.npz` | no coordinates found |
| PeMSD3 | LibCity points to `Davidham3/STSGCN`; BasicTS adjacency script expects `PEMS03.csv` and optional `PEMS03.txt` ID mapping | no coordinates verified |
| PeMSD7 | LibCity points to `Davidham3/STSGCN`; BasicTS adjacency script expects `PEMS07.csv` and optional `PEMS07.txt` ID mapping | no coordinates verified |
| LibCity atomic format | `.geo` format supports `geo_id,type,coordinates`, but raw-data documentation does not prove PeMSD3/4/7/8 packages include real coordinates | format support only, not dataset evidence |

Conclusion: do not use standard PeMSD3/4/7/8 as main coordinate-diagnostic datasets unless a verified `.geo` or station metadata mapping is obtained. For traffic-flow + coordinates, use raw PeMS station data or SD first.

## Data Selection Logic

The first benchmark should cover mechanisms, not breadth:

1. main coordinate-complete traffic flow: raw PeMS 2025 D3-D12 subsets, likely starting from one or two manageable districts;
2. continuity traffic flow: SD 15min;
3. optional single speed control: either METR-LA or PEMS-BAY, not both in the first pass.

This keeps compute under control while still allowing comparison across adaptive graph construction types.

## Immediate Checks Before CD-003

- Decide whether raw PeMS 2025 subsets can be used as an internal benchmark even though they are not the standard PeMSD3/4/7/8 paper datasets.
- For raw PeMS 2025, choose one or two manageable districts and fix node filters, dates, missing-day policy, and graph metadata fields.
- Keep standard PeMSD3/4/7/8 out of coordinate-based learned-graph diagnosis unless `.geo` or official station coordinates are recovered.
- Verify METR-LA and PEMS-BAY raw folders or download scripts are available in the current server environment; mark them as speed controls.
- Copy or preserve sensor-location files alongside BasicTS dataset dirs rather than relying only on `adj_mx.pkl`.
- For KnowAir, keep the MAGE-aligned feature protocol if used for prediction; use `city.txt` and altitude only for graph diagnostics.
- For every dataset, record node order, coordinate order, target channel, time feature policy, train split, scaler, and missing-value policy before training.

## Suggested First Benchmark Matrix

| Dataset | Models / graph types | Graph outputs to save | Diagnostic axes |
|---|---|---|---|
| Raw PeMS 2025 district subset | DCRNN/GWNet fixed, GWNet adaptive, AGCRN, MTGNN, STID; add STAEformer if config is clean | learned graph or embedding snapshots where available | distance locality, directionality, density, freeway/postmile/type alignment, graph-vs-identity comparison |
| SD 15min | same compact model set if configs are already clean | learned graph or embedding snapshots where available | continuity with old SD findings; distance locality, density, node-error relation |
| METR-LA or PEMS-BAY | only a minimal sanity set if needed, e.g. GWNet adaptive + STID/STAEformer | learned graph or embedding snapshots where available | optional speed-control sanity |

## Sources / Evidence

- Local repo: `BasicTS/stgraph_ext/README.md` already documents learned-graph snapshots for AGCRN, GTS, MTGNN, GWNet, and D2STGNN.
- Local repo: `BasicTS/stgraph_ext/adjacency.py` already contains extraction functions for learned graph snapshots.
- Local repo: BasicTS already has configs for PeMSD3/4/7/8 across AGCRN, GWNet, MTGNN, D2STGNN, GTS, DCRNN, STGCN, and other baselines.
- Local repo: `BasicTS/scripts/data_preparation/PEMS04` and `PEMS08` build adjacency from CSV distance files, but do not by themselves prove coordinate provenance.
- External check: ASTGCN PEMS04 and PEMS08 data directories expose only `distance.csv` and `pems04/pems08.npz`, with no coordinate file.
- External check: LibCity raw-data docs list SZ_TAXI as Shenzhen taxi trajectories with road adjacency and speed, LOOP_SEATTLE as 323 loop detector stations, NYCTAXI as taxi GPS/region/OD/grid datasets, and a separate Seattle map-matching GPS dataset.
- External check: T-GCN public `data/` directory exposes `sz_adj.csv` and `sz_speed.csv`, but no coordinate file.
- External check: Seattle-Loop-Data public repository top-level listing exposes README/images only; raw detector coordinate availability is not yet verified.
- External check: MegaCRN release lists EXPY-TKY with 1843 links used in the paper and an EXPY-TKY* superset with all 2841 Tokyo expressway links; the dataset directory contains monthly speed CSVs, `adj01.npy`, `adjdis.npy`, link index files, and link start/end lon/lat metadata.
- External check: BaiduTraffic/Q_TRAFFIC README describes 15,073 Beijing road segments, 15min speed after smoothing/resampling, and a road network sub-dataset with start/end nodes plus road attributes.
- Local repo: `BasicTS/scripts/data_preparation/PEMS03` and `PEMS07` similarly build adjacency from `PEMS03/07.csv` plus optional ID mapping, not coordinates.
- Local repo: `BasicTS/scripts/data_preparation/METR-LA` and `PEMS-BAY` already follow standard BasicTS traffic-speed preparation.
- Local repo: `BasicTS/scripts/data_preparation/AirQuality/generate_training_data.py` verifies KnowAir city coordinates and MAGE-style feature modes.
- Local data check: `KnowAir.npy` is `(11688, 184, 18)` and `city.txt` has 184 longitude/latitude rows.
- LibCity documentation lists PeMSD3/4/7/8, METR-LA, PEMS-BAY, and M-DENSE as traffic datasets, and defines `.geo` files as the place where geographic coordinates are stored, but this is a format definition rather than proof that PeMSD3/4/7/8 include coordinates.
- Local raw PeMS inventory: `real-traffic-benchmark/outputs/raw_data_180_d3_d12_ml_hv_20260608/README_raw_data_180_ml_hv.md` reports D3/D4/D5/D6/D7/D8/D10/D11/D12 raw station data with usable ML/HV nodes after dropping missing coordinates.

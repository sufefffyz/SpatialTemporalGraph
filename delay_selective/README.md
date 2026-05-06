# Delay-Selective Quick Validation

This folder is the workspace for delay-selective traffic forecasting validation.

## City-Traffic Dataset Profiling

The dataset paper is summarized in:

```text
delay_selective/city_traffic_dataset_notes.md
```

Grouping options and the Zotero-backed literature synthesis are summarized in:

```text
delay_selective/grouping_methods_plan.md
```

The important caution is that the current local files:

```text
data/city_traffic_m_speed__category__1_0.npz
data/city_traffic_m_volume__category__1_0.npz
```

are category-specific subgraphs, not the canonical full `city-traffic-M` graph from the paper. The paper reports the full `city-traffic-M` graph as 53,530 road-segment nodes and 121,236 directed road-adjacency edges, and `city-traffic-L` as 94,009 nodes and 164,424 edges. Both use 35,449 timestamps at 5-minute granularity.

On the server, run:

```bash
bash delay_selective/run_city_traffic_profile.sh
```

The script automatically looks for full files first:

```text
data/city_traffic_m_speed.npz
data/city_traffic_m_volume.npz
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_speed.npz
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_m_volume.npz
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_l_speed.npz
/data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_l_volume.npz
```

and also profiles the `category=1.0` subgraphs when present. Outputs are written to:

```text
delay_selective/outputs/dataset_profile/
```

Main outputs:

```text
all_dataset_profile_summary.json
dataset_comparison.csv
<dataset>/summary.json
<dataset>/target_histogram.csv
<dataset>/degree_distribution.csv
<dataset>/edge_distance_distribution.csv
<dataset>/missingness_summary.csv
<dataset>/numeric_spatial_feature_summary.csv
<dataset>/low_cardinality_spatial_feature_counts.csv
```

The profiler records paper-count checks and warnings whenever a file looks like a road-type subgraph, so later delay experiments do not accidentally claim full road-network evidence from a derived subset.

To generate cleaner figures from these profiling outputs:

```bash
bash delay_selective/plot_city_traffic_profile_figures.sh
```

Default figure outputs:

```text
delay_selective/figures/dataset_profile/city_traffic_target_distributions.pdf
delay_selective/figures/dataset_profile/city_traffic_target_distributions.png
delay_selective/figures/dataset_profile/city_traffic_degree_distributions.pdf
delay_selective/figures/dataset_profile/city_traffic_degree_distributions.png
delay_selective/figures/dataset_profile/city_traffic_target_distribution_summary.csv
delay_selective/figures/dataset_profile/city_traffic_degree_summary.csv
```

## Quick Delay Audit

The first server-side sanity experiment is a lightweight **delay observability audit** on the existing road-segment speed dataset:

```text
data/city_traffic_m_speed__category__1_0.npz
product/traffic_city_traffic_m_speed__category__1_0/targets.npy
```

The goal is not to train a model yet. The goal is to quickly test the core claim:

```text
Many near-neighbor road-segment edges have zero or unstable delay,
while a smaller subset of longer-range or inter-segment relations shows identifiable non-zero lag.
```

## Server Command

From the repository root:

```bash
bash delay_selective/run_quick_delay_audit.sh
```

Outputs are written to:

```text
delay_selective/outputs/quick_delay_audit/
```

## What The Script Computes

- lagged correlation for directed graph edges
- best lag per edge, in 5-minute steps by default
- improvement over zero-lag correlation
- near-zero-delay ratio
- non-zero high-confidence delay ratio
- distance-bin summaries using road-segment centroid distance
- optional plots if `matplotlib` is installed

## Main Outputs

```text
edge_delay_scores.csv
distance_bin_summary.csv
summary.json
best_lag_hist.png              optional
best_lag_vs_distance.png       optional
mean_corr_by_lag.png           optional
```

## Interpretation

Evidence supporting the research direction:

```text
1. A large fraction of short-distance edges select lag 0.
2. Non-zero lag edges have clear correlation improvement over lag 0.
3. Non-zero lag ratio increases with edge distance or in selected regimes.
4. All-edge delay would be noisy because many edges do not have stable non-zero delay.
```

This validates the premise for a later model:

```text
intra-group synchronous aggregation
inter-group delay-selective asynchronous propagation
```

## Full Speed Delay Audit

For the first full-road-network audit, run the speed targets on the canonical city-L and city-M graphs:

```bash
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/graph_ml/bin/python \
bash delay_selective/run_full_delay_audit.sh
```

Default outputs:

```text
delay_selective/outputs/full_delay_audit/city_traffic_l_speed_full_train/
delay_selective/outputs/full_delay_audit/city_traffic_m_speed_full_train/
```

The script samples up to 6,000 eligible directed edges per graph after a coverage filter, uses the first 10,000 train timestamps, and computes lagged correlations for 0 to 12 five-minute lags. City-L uses `min_pair_coverage=0.80`; city-M uses `0.50` because the speed target has much heavier missingness.

## LargeST 5-Minute Delay Audit

For SD/GLA/GBA 5-minute delay-effect validation, first build the GLA/GBA BasicTS
bundles from the official LargeST CA raw files:

```bash
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash delay_selective/build_largest_5min_basicts.sh --datasets GBA GLA --overwrite
```

The builder follows the LargeST notebooks:

```text
GBA = CA District 4
GLA = CA Districts 7, 8, and 12
SD  = CA District 11
```

Then run the 5-minute delay audit:

```bash
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
python delay_selective/run_largest_5min_delay_audit.py
```

Default outputs:

```text
delay_selective/outputs/largest_5min_delay_audit/all_summary.csv
delay_selective/outputs/largest_5min_delay_audit/all_distance_bin_summary.csv
delay_selective/outputs/largest_5min_delay_audit/<DATASET>/edge_delay_scores.csv
delay_selective/outputs/largest_5min_delay_audit/<DATASET>/summary.csv
delay_selective/outputs/largest_5min_delay_audit/<DATASET>/distance_bin_summary.csv
delay_selective/outputs/largest_5min_delay_audit/<DATASET>/best_lag_hist.png
delay_selective/outputs/largest_5min_delay_audit/<DATASET>/delay_effect_ratios.png
```

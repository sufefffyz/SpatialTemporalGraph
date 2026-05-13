# Delay-Selective Quick Validation

This folder is the workspace for delay-selective traffic forecasting validation.

## City-Traffic Dataset Profiling

The dataset paper is summarized in:

```text
mvp_experiments/active/delay_selective/city_traffic_dataset_notes.md
```

Grouping options and the Zotero-backed literature synthesis are summarized in:

```text
mvp_experiments/active/delay_selective/grouping_methods_plan.md
```

The important caution is that the current local files:

```text
data/city_traffic_m_speed__category__1_0.npz
data/city_traffic_m_volume__category__1_0.npz
```

are category-specific subgraphs, not the canonical full `city-traffic-M` graph from the paper. The paper reports the full `city-traffic-M` graph as 53,530 road-segment nodes and 121,236 directed road-adjacency edges, and `city-traffic-L` as 94,009 nodes and 164,424 edges. Both use 35,449 timestamps at 5-minute granularity.

On the server, run:

```bash
bash mvp_experiments/active/delay_selective/run_city_traffic_profile.sh
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
mvp_experiments/active/delay_selective/outputs/dataset_profile/
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
bash mvp_experiments/active/delay_selective/plot_city_traffic_profile_figures.sh
```

Default figure outputs:

```text
mvp_experiments/active/delay_selective/figures/dataset_profile/city_traffic_target_distributions.pdf
mvp_experiments/active/delay_selective/figures/dataset_profile/city_traffic_target_distributions.png
mvp_experiments/active/delay_selective/figures/dataset_profile/city_traffic_degree_distributions.pdf
mvp_experiments/active/delay_selective/figures/dataset_profile/city_traffic_degree_distributions.png
mvp_experiments/active/delay_selective/figures/dataset_profile/city_traffic_target_distribution_summary.csv
mvp_experiments/active/delay_selective/figures/dataset_profile/city_traffic_degree_summary.csv
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
bash mvp_experiments/active/delay_selective/run_quick_delay_audit.sh
```

Outputs are written to:

```text
mvp_experiments/active/delay_selective/outputs/quick_delay_audit/
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
bash mvp_experiments/active/delay_selective/run_full_delay_audit.sh
```

Default outputs:

```text
mvp_experiments/active/delay_selective/outputs/full_delay_audit/city_traffic_l_speed_full_train/
mvp_experiments/active/delay_selective/outputs/full_delay_audit/city_traffic_m_speed_full_train/
```

The script samples up to 6,000 eligible directed edges per graph after a coverage filter, uses the first 10,000 train timestamps, and computes lagged correlations for 0 to 12 five-minute lags. City-L uses `min_pair_coverage=0.80`; city-M uses `0.50` because the speed target has much heavier missingness.

## LargeST 5-Minute Delay Audit

For SD/GLA/GBA 5-minute delay-effect validation, first build the GLA/GBA BasicTS
bundles from the official LargeST CA raw files:

```bash
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/build_largest_5min_basicts.sh --datasets GBA GLA --overwrite
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
python mvp_experiments/active/delay_selective/run_largest_5min_delay_audit.py
```

By default this audits only the LargeST built-in graph edges:

```text
distthre = adj_mx_largeST_original.pkl / adj_mx.pkl
```

Do not mix in `physical_dir` unless explicitly needed for a separate SD-only
comparison.

Default outputs:

```text
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/all_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/all_distance_bin_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/<DATASET>/edge_delay_scores.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/<DATASET>/summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/<DATASET>/distance_bin_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/<DATASET>/best_lag_hist.png
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_audit/<DATASET>/delay_effect_ratios.png
```

For the stricter delay-selective audit, use the multi-method script:

```bash
bash mvp_experiments/active/delay_selective/run_largest_5min_delay_multimethod_audit.sh
```

This uses `corr >= 0.8` by default and compares:

```text
mcc_5min_resid          current 5-minute lagged Pearson MCC after residualization
stdde_spline_fft_mcc    STDDE-style smoothed MCC: natural cubic spline to 1 minute, then FFT cross-correlation
lift_fft_abs            LIFT-style normalized-window FFT cross-correlation using absolute lead-lag score
```

It audits the first month and a one-week daily breakdown by default. Outputs:

```text
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_multimethod_audit/all_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_multimethod_audit/all_distance_bin_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_multimethod_audit/<DATASET>/edge_delay_scores.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_multimethod_audit/<DATASET>/summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_delay_multimethod_audit/<DATASET>/distance_bin_summary.csv
```

The shell wrapper also generates visual diagnostics:

```bash
python mvp_experiments/active/delay_selective/plot_largest_5min_delay_multimethod_audit.py
```

Default figures:

```text
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/month_raw_best_delay_hist_log_count.pdf
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/month_effective_delay_hist_log_count.pdf
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/month_delay_acceptance_ratios.pdf
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/week_daily_high_conf_nonzero_ratio.pdf
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/month_distance_bin_high_conf_heatmap_mcc.pdf
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/month_raw_best_delay_by_distance_bin_log_count.pdf
mvp_experiments/active/delay_selective/figures/largest_5min_delay_multimethod_audit/month_effective_delay_by_distance_bin_log_count.pdf
```

The delay histogram y-axis is log-count by default so the large zero-lag bar does not hide the non-zero delay tail. Edges whose best lag confidence is filtered by `corr < 0.8` are written as `NaN` in `effective_lag_*` and plotted as a separate `Filtered` bar instead of being merged into lag 0. If `corr >= 0.8`, the effective lag keeps the best-score lag even when the improvement over zero lag is below the strict high-confidence cutoff; those edges are separately marked by `low_improvement_nonzero_delay`. The distance-bin heatmaps use log-count colors for the same reason.

## Interpretable Group-Level Delay Audit

The first interpretable regional-delay pipeline is:

```text
1. Partition the local LargeST graph into topology-contiguous groups.
2. Aggregate node signals into group-level time series.
3. Audit only directed inter-group edges for delayed propagation.
```

Run it on the server with:

```bash
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/run_largest_5min_interpretable_group_delay.sh
```

Defaults are intentionally conservative:

```text
grouping = topology_bfs
target_group_size = 32
pooling = median
group_signal_mode = pooled
graph = distthre
windows = month + week_daily
methods = mcc_5min_resid + stdde_spline_fft_mcc + lift_fft_abs
```

This keeps the original graph as a local synchronous graph, then tests whether
coarser directed group-to-group edges contain more observable non-zero delay.

Main outputs:

```text
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/all_group_delay_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/all_group_delay_distance_bin_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/all_group_delay_weekly_stability.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/group_assignments.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/group_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/inter_group_edges.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/group_delay_edges.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/group_delay_summary.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/group_delay_weekly_stability.csv
mvp_experiments/active/delay_selective/outputs/largest_5min_interpretable_group_delay/<DATASET>/group_signal_<WINDOW>_<LABEL>.npy
```

The key comparison against the raw edge audit is:

```text
inter-group high_conf_nonzero_ratio
vs.
original graph high_conf_nonzero_ratio
```

If the group-level ratio and weekly stability increase, the data supports
`local synchronous aggregation + sparse regional delayed propagation`. If it
does not increase, the current graph and 5-minute signals probably do not expose
the cross-region delay effect clearly enough.

Additional grouping baselines:

```bash
# PatchSTG-style KDTree spatial patches.
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/run_largest_5min_interpretable_group_delay.sh \
  --grouping patchstg_kdtree \
  --target-group-size 32 \
  --output-dir mvp_experiments/active/delay_selective/outputs/largest_5min_group_delay_patchstg_kdtree

# Coordinate k-means spatial clusters.
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/run_largest_5min_interpretable_group_delay.sh \
  --grouping coordinate_kmeans \
  --target-group-size 32 \
  --output-dir mvp_experiments/active/delay_selective/outputs/largest_5min_group_delay_coordinate_kmeans

# Size-matched random negative control.
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/run_largest_5min_interpretable_group_delay.sh \
  --grouping random_size_matched \
  --target-group-size 32 \
  --output-dir mvp_experiments/active/delay_selective/outputs/largest_5min_group_delay_random
```

Additional group signal modes:

```bash
# Preserve the top-k principal component time series inside each group.
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/run_largest_5min_interpretable_group_delay.sh \
  --grouping patchstg_kdtree \
  --group-signal-mode pca \
  --group-components 3 \
  --output-dir mvp_experiments/active/delay_selective/outputs/largest_5min_group_delay_patchstg_pca

# Brute-force sampled node-node delay across connected groups.
PYTHON_BIN=/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python \
bash mvp_experiments/active/delay_selective/run_largest_5min_interpretable_group_delay.sh \
  --grouping patchstg_kdtree \
  --group-signal-mode node_pair \
  --max-node-pairs-per-group-edge 256 \
  --output-dir mvp_experiments/active/delay_selective/outputs/largest_5min_group_delay_patchstg_node_pair
```

`patchstg_kdtree` follows PatchSTG's spatial data-management idea: recursively
split irregularly distributed sensors by alternating longitude/latitude axes to
produce balanced, non-overlapping spatial patches. This is a geometry-balanced
baseline, not a road-propagation claim by itself.

`pca` and `node_pair` are intended to test whether median/mean pooling destroys
the delayed signal:

```text
pooled      one aggregate time series per group
pca         top-k component time series per group
node_pair   sampled original node-node pairs across connected groups
```

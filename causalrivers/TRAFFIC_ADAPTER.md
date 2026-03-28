# Traffic Adapter

This repository now includes a helper script for adapting an Urban Traffic Benchmark `.npz` dataset so it can be run through the local `causalrivers` benchmark code.

## macOS environment

The original upstream environment file is Linux-oriented. A smaller macOS-friendly conda file is now included:

```bash
conda env create -f causal_rivers_macos.yml
conda activate causalrivers-macos
```

## Why an adapter is needed

`causalrivers` expects two things:

- a time series source referenced by `data_path`
- a pickle referenced by `label_path` that contains a list of NetworkX subgraphs used as labels

The traffic dataset at the repository root uses a different format: a single `.npz` file with `targets`, `unix_timestamps`, `edges`, and feature arrays.

## What the adapter does

`prepare_urban_traffic_for_causalrivers.py` converts the traffic `.npz` into:

- `product/traffic_<dataset>/graph.p`
- `product/traffic_<dataset>/targets.npy`
- `product/traffic_<dataset>/unix_timestamps.npy`
- `product/traffic_<dataset>/node_ids.npy`
- sampled label pickles under `datasets/traffic_<dataset>/.../<dataset>.p`

The benchmark loader in `tools/tools.py` was also extended so `data_path` can point to:

- a CSV file
- an NPZ file
- a directory containing `targets.npy` and `unix_timestamps.npy`

The directory format is preferred for large traffic datasets because it can memory-map the target matrix instead of forcing a full CSV conversion.

## Important limitation

This only makes the pipeline runnable.

The Urban Traffic Benchmark file does not provide a verified causal ground-truth graph in the same sense as the original river benchmark. The adapter therefore uses the road-topology edges as pseudo labels. That means:

- the code can run
- metrics can be computed
- the scores are not directly comparable to the original CausalRivers leaderboard

## Recommended first run

Use a tiny smoke-test sample set first, because the traffic graph is much larger than the river graphs.

```bash
python prepare_urban_traffic_for_causalrivers.py \
  --input-npz ../data/city_traffic_m_speed__category__1_0.npz \
  --strategies debug_set \
  --n-vars 3 \
  --max-samples 5
```

This now writes into the local `causalrivers/product/` and `causalrivers/datasets/` directories even if you launch the script from the repository root.

Then run:

```bash
python benchmark.py \
  label_path=/absolute/path/to/causalrivers/datasets/traffic_city_traffic_m_speed__category__1_0/debug_set_3/city_traffic_m_speed__category__1_0.p \
  data_path=/absolute/path/to/causalrivers/product/traffic_city_traffic_m_speed__category__1_0 \
  method=var \
  data_preprocess.normalize=False \
  data_preprocess.resolution=5min
```

Both the adapter and benchmark now show progress bars for sample assembly and per-sample benchmarking on macOS when `tqdm` is installed through `causal_rivers_macos.yml`.

## Larger 10k traffic run

To prepare a larger `10k` traffic label set without overwriting the earlier smoke-test labels:

```bash
python prepare_urban_traffic_for_causalrivers.py \
  --input-npz ../data/city_traffic_m_speed__category__1_0.npz \
  --strategies debug_set \
  --n-vars 3 \
  --max-samples 10000 \
  --label-tag 10k
```

This writes labels to:

```text
datasets/traffic_city_traffic_m_speed__category__1_0/debug_set_3_10k/city_traffic_m_speed__category__1_0.p
```

To benchmark several baselines in one run:

```bash
python benchmark.py --config-name benchmark_traffic_multi
```

The multi-baseline preset currently runs:

- `var`
- `corr`
- `lagcorr`

The benchmark now streams samples one at a time, which is much more stable for `10k` traffic subsets than loading every sample into memory first.

For the traffic speed data, the preset now treats remaining `NaN` values as `0` after resampling and interpolation. This is intended for the common case where a road segment has no flow and therefore no measured speed.

To use a larger server more effectively, you can parallelize at the sample level:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmark.py --config-name benchmark_traffic_multi \
  n_jobs=16 chunk_size=128 \
  score_n_jobs=16 score_chunk_size=512
```

Notes:

- `n_jobs` controls how many workers process sample chunks in parallel.
- `chunk_size` controls how many graph samples are grouped into one worker task.
- `score_n_jobs` controls how many workers are used during the scoring stage.
- `score_chunk_size` controls the scoring-task chunk size.
- On Linux servers this uses process workers.
- In restricted local environments where process workers are unavailable, the benchmark falls back to thread workers so you can still smoke-test the pipeline.

If you want the older CausalRivers behavior that eagerly loads all required time series once and then benchmarks everything from memory, use:

```bash
python benchmark.py --config-name benchmark_traffic_multi load_mode=eager n_jobs=1
```

`load_mode=eager` is the closest match to the original one-shot benchmark flow. For the large traffic dataset, `load_mode=streaming` is still the safer default.

## Strategy notes

- `debug_set` is the safest option for a first validation run.
- `random` can work, but exhaustive connected-subgraph enumeration on large traffic graphs can become expensive.
- `root_cause` is usually not applicable because road graphs are not DAGs.
- `close` and `disjoint` only work when coordinate features are present.

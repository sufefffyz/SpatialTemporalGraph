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
- sampled label pickles under `datasets/traffic_<dataset>/.../east.p`

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
  label_path=/absolute/path/to/causalrivers/datasets/traffic_city_traffic_m_speed__category__1_0/debug_set_3/east.p \
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
datasets/traffic_city_traffic_m_speed__category__1_0/debug_set_3_10k/east.p
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

## Strategy notes

- `debug_set` is the safest option for a first validation run.
- `random` can work, but exhaustive connected-subgraph enumeration on large traffic graphs can become expensive.
- `root_cause` is usually not applicable because road graphs are not DAGs.
- `close` and `disjoint` only work when coordinate features are present.

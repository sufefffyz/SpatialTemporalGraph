# STGraph BasicTS Extensions

This extension layer adapts two datasets to BasicTS with minimal impact on the upstream framework:

- `TRAFFIC_VOLUME_5MIN`
- `RIVERS_EAST_GERMANY_15MIN`

Both preparation scripts now also support optional temporal aggregation, so you can generate
coarser-resolution forecasting datasets such as `15MIN`, `30MIN`, `1H`, `6H`, `12H`, and `24H`.

It adds:

- explicit-split forecasting dataset support via `split_indices.npz`
- explicit-train-split z-score scaling
- periodic learned-graph snapshots under `learned_graphs/`
- 8 ready-to-run configs for `AGCRN`, `GTS`, `MTGNN`, and `GWNet`

## Prepare datasets

From the `BasicTS` root:

```bash
python stgraph_ext/scripts/prepare_traffic_volume_dataset.py \
  --input-npz ../data/city_traffic_m_volume__category__1_0.npz
```

```bash
python stgraph_ext/scripts/prepare_rivers_east_dataset.py \
  --input-csv ../causalrivers/product/rivers_ts_east_germany.csv \
  --graph-pickle ../causalrivers/product/rivers_east_germany.p
```

The traffic split is copied from UTB's explicit split arrays. The rivers split is created chronologically as `70/10/20`.

To generate coarser-resolution datasets:

```bash
python stgraph_ext/scripts/prepare_traffic_volume_dataset.py \
  --input-npz ../data/city_traffic_m_volume__category__1_0.npz \
  --resolution 15min
```

```bash
python stgraph_ext/scripts/prepare_rivers_east_dataset.py \
  --input-csv ../causalrivers/product/rivers_ts_east_germany.csv \
  --graph-pickle ../causalrivers/product/rivers_east_germany.p \
  --resolution 1h
```

If `--dataset-name` is omitted, the output name is inferred from the requested resolution, e.g.
`TRAFFIC_VOLUME_15MIN` or `RIVERS_EAST_GERMANY_1H`.

- traffic aggregation keeps UTB's explicit split semantics by mapping each aggregated bucket to
  `train / val / test`; mixed buckets that cross split boundaries are dropped to avoid leakage
- rivers aggregation is applied first, and the `70/10/20` chronological split is rebuilt on the
  aggregated series

## Train models

```bash
python experiments/train.py -c stgraph_ext/configs/AGCRN_TRAFFIC_VOLUME_5MIN.py -g 0
python experiments/train.py -c stgraph_ext/configs/GTS_TRAFFIC_VOLUME_5MIN.py -g 0
python experiments/train.py -c stgraph_ext/configs/MTGNN_TRAFFIC_VOLUME_5MIN.py -g 0
python experiments/train.py -c stgraph_ext/configs/GWNET_TRAFFIC_VOLUME_5MIN.py -g 0
```

```bash
python experiments/train.py -c stgraph_ext/configs/AGCRN_RIVERS_EAST_GERMANY_15MIN.py -g 0
python experiments/train.py -c stgraph_ext/configs/GTS_RIVERS_EAST_GERMANY_15MIN.py -g 0
python experiments/train.py -c stgraph_ext/configs/MTGNN_RIVERS_EAST_GERMANY_15MIN.py -g 0
python experiments/train.py -c stgraph_ext/configs/GWNET_RIVERS_EAST_GERMANY_15MIN.py -g 0
```

You can override the epoch count for any config with:

```bash
STGRAPH_NUM_EPOCHS=10 python experiments/train.py -c stgraph_ext/configs/AGCRN_TRAFFIC_VOLUME_5MIN.py -g 0
```

To train on an aggregated dataset without creating another config file, override the dataset name:

```bash
STGRAPH_DATASET_NAME=TRAFFIC_VOLUME_15MIN \
python experiments/train.py -c stgraph_ext/configs/AGCRN_TRAFFIC_VOLUME_5MIN.py -g 0
```

```bash
STGRAPH_DATASET_NAME=RIVERS_EAST_GERMANY_1H \
python experiments/train.py -c stgraph_ext/configs/GWNET_RIVERS_EAST_GERMANY_15MIN.py -g 0
```

## Learned graph snapshots

For supported adaptive-graph models, snapshots are written to:

```text
checkpoints/<Model>/<Dataset>_<epochs>_<input_len>_<output_len>/learned_graphs/
```

Files follow this pattern:

- `epoch_005.npz`
- `epoch_010.npz`
- `final.npz`

Each `.npz` stores at least:

- `adj`
- `epoch`
- `model_name`
- `dataset_name`

## Notes

- This extension intentionally avoids modifying `basicts/` and the model bodies.
- `GTS` can become memory-intensive on very large graphs because it learns dense pairwise structure.

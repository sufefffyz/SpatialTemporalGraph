# STGraph BasicTS Extensions

This extension layer adapts two datasets to BasicTS with minimal impact on the upstream framework:

- `TRAFFIC_VOLUME_5MIN`
- `RIVERS_EAST_GERMANY_15MIN`

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

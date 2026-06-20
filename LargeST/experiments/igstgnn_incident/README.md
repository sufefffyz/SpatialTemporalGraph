# IGSTGNN Incident Datasets in the LargeST Framework

This adapter runs LargeST-framework baselines on the three IGSTGNN released
incident datasets:

- `Alameda`
- `Contra_Costa`
- `Orange`

The datasets remain IGSTGNN datasets. This is not the LargeST SD/GBA/GLA
benchmark. The adapter only lets LargeST baseline implementations consume the
released IGSTGNN `incident_train.npy`, `incident_val.npy`, and
`incident_test.npy` split files.

## Data Contract

Each dataset directory is expected to contain:

- `incident_train.npy`
- `incident_val.npy`
- `incident_test.npy`
- `incident_stats.npz`
- `adj_matrix.npy`

The object-array samples are stacked at runtime:

- `x = sample["x_data"][..., :input_dim]`
- `y = sample["y_data"][..., :1]`

The split order is preserved. No reshuffling, resplitting, or renormalization is
performed. `incident_stats.npz` is reused for inverse scaling.

## Models

- `DSTAGNN`: follows the LargeST script convention and uses `input_dim=1`.
- `D2STGNN`: uses `input_dim=3`, with flow as `num_feat=1` and the time features
  used as temporal embeddings.
- `BiST`: uses `input_dim=3`. A seed argument is added for reproducible runs.

Run smoke tests:

```bash
RUN_MODE=smoke GPU=2 bash experiments/igstgnn_incident/run_igstgnn_incident_baselines.sh
```

Run full experiments:

```bash
RUN_MODE=full GPU=2 bash experiments/igstgnn_incident/run_igstgnn_incident_baselines.sh
```

The full-run batch sizes are project defaults for these IGSTGNN datasets because
the papers do not provide LargeST-baseline hyperparameters for Alameda,
Contra_Costa, or Orange.


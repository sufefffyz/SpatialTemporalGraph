# OSRM Gaussian Global BasicTS Runs

Date: 2026-05-13

## Scope

Run the five precomputed OSRM Gaussian global-threshold SD graph variants on:

- GraphWaveNet fixed graph (`addaptadj=False`)
- DCRNN fixed diffusion graph

Common settings:

- `BASICTS_SEED=2023`
- `NUM_EPOCHS=100`
- `BATCH_SIZE=64`
- `EARLY_STOPPING_PATIENCE=30`
- `WANDB_MODE=online`
- `WANDB_PROJECT=adaptive_threshold_dynamic_weight`

## Server Launch

Server:

```text
ssh -p 5102 yuzhang_fei@183.174.228.180
```

Screens:

```text
osrmg_gwnet_20260513
osrmg_dcrnn_20260513
```

Each screen runs the five beta variants sequentially on GPU 1:

```text
beta0p25 -> beta0p50 -> beta1p00 -> beta1p50 -> beta2p00
```

## Logs

```text
BasicTS/logs/adaptive_threshold_dynamic_weight/osrm_gaussian_global/osrm_gaussian_global_20260513/gwnet/
BasicTS/logs/adaptive_threshold_dynamic_weight/osrm_gaussian_global/osrm_gaussian_global_20260513/dcrnn/
```

## Initial Verification

Both first runs (`beta0p25`) loaded the staged datasets and entered the training
loop successfully. GPU 1 was active with about 21 GB memory used after launch.

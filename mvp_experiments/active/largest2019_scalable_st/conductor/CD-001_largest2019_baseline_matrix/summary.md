# CD-001 LargeST2019 Baseline Matrix Summary

Generated: 2026-06-24 12:10 CST

Branch: `[CD-001][W1][experiment] LargeST2019 baseline matrix`

Status: `MATRIX_COMPLETE_EXCEPT_MAGE_GLA_RUNNING`

`completion-report.md` has not been written yet. The branch brief requires the completion report only after user confirmation.

## Metric Convention

Each metric cell is `MAE / RMSE / MAPE`.

| Model | Convention |
|---|---|
| STID | BasicTS `test_metrics.json` `overall` metrics from the best-val checkpoint. |
| PatchSTG | Framework `TEST MODE` average from `PatchSTG/log/cd001_patchstg_*.log`. |
| BiST | Framework `Average Test` row from `LargeST/experiments/BiST/*/logs/cd001_bist_*.log`. |
| MAGE | Framework `Average Test` row from `references/mage_official/experiments/mage/*/*_fix1.outer.log`. |

These rows target the same LargeST 2019 SD/GLA/GBA split and 12-step forecasting task, but the metric conventions remain mixed. Treat this as a labeled reproduction coordinate table, not an unlabeled final paper table.

## Result Matrix

| Model | SD 2019 | GLA 2019 | GBA 2019 |
|---|---:|---:|---:|
| STID | `17.7999 / 30.9125 / 0.1184` | `19.5705 / 33.5828 / 0.1217` | `20.1669 / 34.5046 / 0.1596` |
| PatchSTG | `16.6601 / 28.8548 / 0.1080` | `19.3580 / 33.0774 / 0.1152` | `19.8441 / 33.8339 / 0.1493` |
| BiST | `16.6520 / 28.0990 / 0.1161` | `19.9555 / 32.6986 / 0.1224` | `19.8548 / 32.9439 / 0.1526` |
| MAGE | `16.3710 / 28.2346 / 0.1093` | `RUNNING: no final Average Test yet` | `19.7808 / 33.1533 / 0.1555` |

## Runtime And Memory

| Model | Dataset | Runtime Evidence | Peak GPU Memory |
|---|---|---:|---:|
| STID | SD | `1201s` profiler duration | `1015 MB` |
| STID | GLA | `12365s` profiler duration | `17527 MB` |
| STID | GBA | `11585s` profiler duration | `16359 MB` |
| PatchSTG | SD | `4310.8s` summed 50 logged epoch times | not logged |
| PatchSTG | GLA | `19462.1s` summed 50 logged epoch times | not logged |
| PatchSTG | GBA | `27172.5s` summed 50 logged epoch times | not logged |
| BiST | SD | `1964s` wall time, early stop epoch 53 | not logged |
| BiST | GLA | `29410s` wall time, early stop epoch 120 | not logged |
| BiST | GBA | `7499s` wall time, early stop epoch 57 | not logged |
| MAGE | SD | `8019s` wall time, `8004.36s` log Time Cost, early stop epoch 175 | not logged |
| MAGE | GLA | still running; latest complete epoch 61; `75031.88s` train+val through epoch 61 | live current memory `28638 MB` at 2026-06-24 12:05 CST; peak not logged |
| MAGE | GBA | `50351s` wall time, `50298.64s` log Time Cost, early stop epoch 138 | not logged |

## Primary Artifacts

| Model | Dataset | Server | Artifact |
|---|---|---|---|
| STID | SD | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/checkpoints/STID/SD_100_12_12_cd001_stid_sd_2019_s2023_retry/3370c6b0105fff7ae63fb65dc0b510c9/test_metrics.json` |
| STID | GLA | `183.174.228.180` | `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/checkpoints/STID/GLA_100_12_12/f98dce2691ccacf6ab61944541898d20/test_metrics.json` |
| STID | GBA | `183.174.228.180` | `/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS/checkpoints/STID/GBA_100_12_12/8512bfd98d9d64469b1cbf9f14168755/test_metrics.json` |
| PatchSTG | SD | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG/log/cd001_patchstg_sd_full_g1.log` |
| PatchSTG | GLA | `183.174.228.180` | `/home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG/log/cd001_patchstg_gla_full_g0.log` |
| PatchSTG | GBA | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG/log/cd001_patchstg_gba_full_g0.log` |
| BiST | SD | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/LargeST/experiments/BiST/SD/logs/cd001_bist_sd_full_g1.log` |
| BiST | GLA | `183.174.228.180` | `/home/yuzhang_fei/code/SpatialTemporalGraph/LargeST/experiments/BiST/GLA/logs/cd001_bist_gla_full_g0.log` |
| BiST | GBA | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/LargeST/experiments/BiST/GBA/logs/cd001_bist_gba_full_g0.log` |
| MAGE | SD | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/references/mage_official/experiments/mage/sd/cd001_mage_sd_full_g1_fix1.outer.log` |
| MAGE | GLA | `183.174.228.180` | `/home/yuzhang_fei/code/SpatialTemporalGraph/references/mage_official/experiments/mage/gla/cd001_mage_gla_full_g1_fix1.outer.log` |
| MAGE | GBA | `183.174.228.178` | `/home/yuzhang_fei/code/SpatialTemporalGraph/references/mage_official/experiments/mage/gba/cd001_mage_gba_full_g0_fix1.outer.log` |

## Reproducibility Notes

- STID used BasicTS configs `baselines/STID/{SD,GLA,GBA}.py`, seed `2023`, profile tags `STID_SD`, `STID_GLA`, and `STID_GBA`.
- PatchSTG used `./config/CD001_SD_g1.conf`, `./config/CD001_GLA_g0.conf`, and `./config/CD001_GBA_g0.conf`; logged `train_ratio: 0.6`; logged `recur_times` are `9` for SD and `11` for GLA/GBA; padded nodes are `1024` for SD and `4096` for GLA/GBA.
- BiST used `model_name='bist'`, seed `2025`, `bs=64`, `seq_len=12`, `horizon=12`, `max_epochs=500`, `patience=20`, `kernel_size=3`; dataset cores were `8` for SD, `32` for GLA, and `24` for GBA.
- MAGE official reference copy is at commit `f1fdd27` under `references/mage_official`; runs used seed `3028`, `bs=64`, `max_epochs=300`, `patience=30`, `recur_num=16`, `topk=4`, `lrate=0.01`, `wdecay=0.0001`, `dropout=0.1`.
- MAGE had a local evaluation-stage compatibility patch in `src/engines/MAGE_enine.py`: handle outputs without `topk_indices`, use `self._args.recur_num`/`self._device` for expert-load accumulation, and remove a hard-coded `np.save('/data/mjm1/...')` plus `sys.exit(0)`. No MAGE hyperparameters or model architecture were changed by this branch pass.

## Live State Refresh

- `183.174.228.178` at 2026-06-24 12:05 CST: no CD-001 screens. GPU0 had unrelated activity; GPU1 was idle.
- `183.174.228.180` at 2026-06-24 12:05 CST: screen `472029.cd001_mage_gla_full_g1_fix1` was detached and active.
- MAGE GLA process `PID 472035` was `/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/mage/main.py --device cuda:0 --dataset gla --years 2019 --model_name mage`, elapsed `20:55:18`, using `28638 MB` at the check.
- Latest MAGE GLA completed epoch in the log was Epoch 61 at 2026-06-24 12:03:28 CST, with validation MAE/RMSE/MAPE `17.2290 / 28.7417 / 0.1208`; no final `Average Test` row exists yet.

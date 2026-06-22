# CD-001 Completion Report: LargeST2019 Baselines

Status: DONE_WITH_CONCERNS
Timestamp: 2026-06-22 16:25:32 CST
Branch: CD-001 LargeST2019 baselines

## Scope

Prepared and audited reproduction status for PatchSTG, BiST, MAGE, and STID on LargeST 2019 SD, GLA, and GBA. No baseline hyperparameters were intentionally changed. No repo files outside this branch artifact directory were modified.

## Data Availability

| Location | SD 2019 | GLA 2019 | GBA 2019 | Notes |
| --- | --- | --- | --- | --- |
| Local LargeST | BLOCKED | BLOCKED | BLOCKED | `LargeST/data/{sd,gla,gba}` has notebooks/scripts only; no generated `2019/his.npz` locally. |
| Local PatchSTG | BLOCKED | BLOCKED | BLOCKED | Only metadata CSVs exist under `PatchSTG/data`; missing `flow*.npz` and `adj.npy`. |
| Local BiST | BLOCKED | BLOCKED | BLOCKED | Only data-generation docs/scripts; missing generated `{sd,gla,gba}/2019/{his.npz,idx_*.npy}`. |
| Local BasicTS | READY | BLOCKED | BLOCKED | `BasicTS/datasets/SD/{data.dat,adj_mx.pkl,meta.csv,desc.json}` exists; local `GLA` and `GBA` exact dataset dirs absent. |
| 183.174.228.180 LargeST | READY | BLOCKED | BLOCKED | Has `/home/yuzhang_fei/code/SpatialTemporalGraph/LargeST/data/sd/2019/{his.npz,idx_*.npy}` only. |
| 183.174.228.180 BasicTS | READY | READY | READY | Exact `BasicTS/datasets/{SD,GLA,GBA}` all have descriptors and required data/graph files. GPUs busy at audit time. |
| 183.174.228.178 LargeST | READY | BLOCKED | BLOCKED | Has SD 2019 generated data only; no GLA/GBA `2019/his.npz`. |
| 183.174.228.178 BasicTS | READY | BLOCKED | CONCERN | Exact `SD` is complete. Exact `GBA` has descriptor/meta but `adj_mx.pkl` check failed; `GBA_5min_full` exists but is a different dataset name/frequency. Exact `GLA` absent. |

## Baseline Availability

| Baseline | Local code status | SD runnable status | GLA/GBA status | Notes |
| --- | --- | --- | --- | --- |
| PatchSTG | READY | DATA_BLOCKED | DATA_BLOCKED | Entrypoint exists: `PatchSTG/main.py` and `LargeST/experiments/patchstg/main.py`; configs exist for `SD/GLA/GBA`. Missing `flow*.npz` and `adj.npy` data. |
| BiST | CONCERN | CODE_BLOCKED | DATA_AND_CODE_BLOCKED | Preserved upstream `BiST/experiments/main.py` exists; integrated `LargeST/experiments/bist/main.py` exists. Remote preflight showed integrated path recurses on `get_config`; preserved upstream path failed with `ModuleNotFoundError: No module named 'src'` when run from repo root on 178. |
| STID | READY | LAUNCH_ATTEMPT_FAILED_ENV | GLA/GBA WAIT_FOR_SAFE_SERVER | BasicTS configs exist: `baselines/STID/{SD,GLA,GBA}.py`. Remote env interpreter exists, but wrapper launch failed because `python` is not on PATH inside screen. |
| MAGE | BLOCKED | BLOCKED | BLOCKED | No local MAGE checkout found under allowed context. Quick online search did not confirm an official LargeST/MAGE baseline repository; do not hand-write MAGE. Needs exact paper/title or official repo confirmation. |

## Commands

### PatchSTG SD 2019

Preflight:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG
test -f ./data/SD/flowsd.npz
test -f ./data/SD/adj.npy
test -f ./data/SD/sd_meta.csv
```

Full run after data is present:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python main.py --config ./config/SD.conf
```

GLA/GBA queue templates after `flowgla.npz`/`flowgba.npz` and `adj.npy` are present:

```bash
screen -dmS cd001_patchstg_gla_full_g0 bash -lc 'cd /home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG && /home/yuzhang_fei/miniconda3/envs/STGraph/bin/python main.py --config ./config/GLA.conf > log/cd001_gla_full.log 2>&1'
screen -dmS cd001_patchstg_gba_full_g1 bash -lc 'cd /home/yuzhang_fei/code/SpatialTemporalGraph/PatchSTG && /home/yuzhang_fei/miniconda3/envs/STGraph/bin/python main.py --config ./config/GBA.conf > log/cd001_gba_full.log 2>&1'
```

### BiST SD 2019

Preflight:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/LargeST
test -f ./data/sd/2019/his.npz
test -f ./data/sd/2019/idx_train.npy
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/bist/main.py --help
```

Expected full command after fixing the `get_config` recursion in the integrated entrypoint or using a confirmed upstream path:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/LargeST
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/bist/main.py --dataset SD --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 8 --seq_len 12 --horizon 12 --years 2019
```

GLA/GBA queue templates after data and entrypoint are fixed:

```bash
screen -dmS cd001_bist_gla_full_g0 bash -lc 'cd /home/yuzhang_fei/code/SpatialTemporalGraph/LargeST && /home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/bist/main.py --dataset GLA --kernel_size 3 --device cuda:0 --bs 64 --model_name bist --core 32 --seq_len 12 --horizon 12 --years 2019 > experiments/BiST/GLA/cd001_full.log 2>&1'
screen -dmS cd001_bist_gba_full_g1 bash -lc 'cd /home/yuzhang_fei/code/SpatialTemporalGraph/LargeST && /home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/bist/main.py --dataset GBA --kernel_size 3 --device cuda:1 --bs 64 --model_name bist --core 24 --seq_len 12 --horizon 12 --years 2019 > experiments/BiST/GBA/cd001_full.log 2>&1'
```

### STID SD 2019

Failed launch command actually run:

```bash
ssh -p 5102 yuzhang_fei@183.174.228.178 'cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS && mkdir -p logs/cd001_largest2019_baselines && screen -dmS cd001_stid_sd_full_g0 bash -lc "cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS && WANDB_MODE=offline BASICTS_RUN_TAG=cd001_stid_sd_2019_s2023 BASICTS_SEED=2023 bash scripts/run_largest_baseline.sh STID SD 0 > logs/cd001_largest2019_baselines/stid_sd_full_g0.log 2>&1"'
```

Result: screen `cd001_stid_sd_full_g0` exited immediately. Profile summary:

```json
{
  "config_path": "baselines/STID/SD.py",
  "gpus": "0",
  "seed": 2023,
  "tag": "STID_SD",
  "duration_seconds": 0,
  "peak_gpu_memory_mb": 0,
  "exit_code": 127,
  "test_metrics_path": null
}
```

Exact blocker:

```text
scripts/run_profiled_training.sh: line 62: python: command not found
```

Recommended corrected SD full command:

```bash
ssh -p 5102 yuzhang_fei@183.174.228.178 'cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS && screen -dmS cd001_stid_sd_full_g0_retry bash -lc "cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS && export PATH=/home/yuzhang_fei/miniconda3/envs/STGraph/bin:$PATH && WANDB_MODE=offline BASICTS_RUN_TAG=cd001_stid_sd_2019_s2023_retry BASICTS_SEED=2023 bash scripts/run_largest_baseline.sh STID SD 0 > logs/cd001_largest2019_baselines/stid_sd_full_g0_retry.log 2>&1"'
```

Alternative direct command avoiding the wrapper:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS
WANDB_MODE=offline BASICTS_RUN_TAG=cd001_stid_sd_2019_s2023 BASICTS_SEED=2023 /home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py -c baselines/STID/SD.py -g 0
```

GLA/GBA queue templates for 180 after current GPU jobs finish:

```bash
screen -dmS cd001_stid_gla_full_wait_g0 bash -lc 'while true; do mem=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [ "$mem" -lt 500 ] && break; sleep 300; done; cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS && export PATH=/home/yuzhang_fei/miniconda3/envs/STGraph/bin:$PATH && WANDB_MODE=offline BASICTS_RUN_TAG=cd001_stid_gla_2019_s2023 BASICTS_SEED=2023 bash scripts/run_largest_baseline.sh STID GLA 0 > logs/cd001_largest2019_baselines/stid_gla_full_g0.log 2>&1'
screen -dmS cd001_stid_gba_full_wait_g1 bash -lc 'while true; do mem=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [ "$mem" -lt 500 ] && break; sleep 300; done; cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS && export PATH=/home/yuzhang_fei/miniconda3/envs/STGraph/bin:$PATH && WANDB_MODE=offline BASICTS_RUN_TAG=cd001_stid_gba_2019_s2023 BASICTS_SEED=2023 bash scripts/run_largest_baseline.sh STID GBA 1 > logs/cd001_largest2019_baselines/stid_gba_full_g1.log 2>&1'
```

## Remote GPU / Job Snapshot

| Server | Snapshot | Status | Decision |
| --- | --- | --- | --- |
| 183.174.228.180 | 2026-06-22 16:22 CST | GPU0/GPU1 both about 38 GB used and 94-95% util; active screens include `stemdist_table3_gla1p_allmetrics_g0/g1`, `cd006_block_random_node_window_stid_sd_g0`, and old adaptive-graph screens. | Did not launch or kill anything. Suitable for queued GLA/GBA only after user approval. |
| 183.174.228.178 | 2026-06-22 16:22 CST | No screens, both A100s idle at 17 MiB. | Attempted STID SD full on GPU0; screen exited immediately due PATH issue. No running CD-001 job remains. |

## Run / Queue Table

| Baseline | Dataset | Server | Action | Screen | Status | Metric convention |
| --- | --- | --- | --- | --- | --- | --- |
| STID | SD 2019 | 183.174.228.178 | Launched full run via wrapper | `cd001_stid_sd_full_g0` | FAILED immediately, exit 127 (`python` not found) | BasicTS overall if completed; no metrics produced. |
| STID | GLA 2019 | 183.174.228.180 | Not launched | none | READY data, GPUs busy | BasicTS overall. |
| STID | GBA 2019 | 183.174.228.180 | Not launched | none | READY data, GPUs busy | BasicTS overall. |
| PatchSTG | SD/GLA/GBA | none | Not launched | none | DATA_BLOCKED | Original/PatchSTG average if completed. |
| BiST | SD | none | Not launched | none | CODE_BLOCKED | Original/BiST average if completed. |
| MAGE | SD/GLA/GBA | none | Not launched | none | OFFICIAL_CODE_BLOCKED | Not applicable. |

## Metrics

No CD-001 baseline metrics completed in this turn.

Metric convention to use once results exist:

| Baseline | Convention to record |
| --- | --- |
| PatchSTG | Original/PatchSTG reported per horizon plus average from its `metric(pred, label)` logging. |
| BiST | Original/BiST engine metrics from its experiment logs; record whether per-horizon or aggregate after run. |
| STID | BasicTS overall metrics from `test_metrics.json`; if horizon-specific values are extracted later, label them separately from BasicTS overall. |
| MAGE | Unknown until official code/protocol is confirmed. |

## Missing Items

- PatchSTG processed data: `flowsd.npz`, `flowgla.npz`, `flowgba.npz`, and `adj.npy` under the corresponding `PatchSTG/data/*` folders.
- LargeST/BiST processed GLA/GBA 2019 data on both remote servers.
- BiST launchable entrypoint: integrated LargeST path has `get_config` recursion; preserved upstream path fails import resolution on 178.
- MAGE official source confirmation. No local code found; online quick search did not establish an official repo.
- STID rerun needs only environment PATH fix or direct interpreter command.

## Proposed Next Master Decision

Approve one of:

1. Relaunch STID SD on 178 using the corrected PATH/direct-python command, then queue STID GLA/GBA on 180 after current jobs finish.
2. First fix/sync BiST entrypoint and prepare LargeST GLA/GBA 2019 generated data, then launch all baselines together.
3. Keep CD-001 as preparation-only until MAGE official code status is resolved.

## Suggested Merge Note

- CD-001 audited PatchSTG, BiST, STID, and MAGE for LargeST 2019 SD/GLA/GBA.
- PatchSTG code/configs exist but processed `flow*.npz` and `adj.npy` are missing locally/remotely.
- BiST code exists but remote entrypoints are not currently launchable: integrated path recurses; upstream path fails `src` import.
- STID BasicTS configs exist for SD/GLA/GBA; exact remote data exists on 180, while 178 has exact SD and partial GBA/no GLA.
- One STID SD full launch was attempted on 178 as `cd001_stid_sd_full_g0`; it exited immediately with `python: command not found` in wrapper.
- No CD-001 jobs are currently running; no metrics completed.
- MAGE remains blocked: no local official code found and quick online search did not confirm an official repo.
- Next safe action is rerun STID SD with PATH/direct interpreter, then queue STID GLA/GBA on 180 after existing GPU jobs finish.

## Master Follow-up Update

Timestamp: 2026-06-22 16:29 CST

After this report returned, master applied the recommended PATH fix and relaunched STID SD on 178.

| Baseline | Dataset | Server | Screen | Status | Notes |
|---|---|---|---|---|---|
| STID | SD 2019 | 183.174.228.178 | `cd001_stid_sd_full_g0_retry` | RUNNING | Epoch 1 completed; train time about 10.2s/epoch; estimated finish around 2026-06-22 16:48 CST. |
| STID | GLA 2019 | 183.174.228.180 | `cd001_stid_gla_full_wait_g0` | QUEUED_WAITING | Polls GPU0 every 300s and starts only when memory used is below 500 MiB. |
| STID | GBA 2019 | 183.174.228.180 | `cd001_stid_gba_full_wait_g1` | QUEUED_WAITING | Polls GPU1 every 300s and starts only when memory used is below 500 MiB. |

The original `python: command not found` blocker is resolved for the relaunched and queued STID jobs by exporting `/home/yuzhang_fei/miniconda3/envs/STGraph/bin` into `PATH`.

## Master Follow-up Update 2

Timestamp: 2026-06-22 16:32 CST

| Baseline | Dataset | Server | Screen | Status | Notes |
|---|---|---|---|---|---|
| STID | SD 2019 | 183.174.228.178 | `cd001_stid_sd_full_g0_retry` | RUNNING | Around Epoch 18/100; GPU0 uses about 1015 MiB; train time about 10s/epoch; latest best val MAE 16.2996 at Epoch 17; estimated finish remains around 2026-06-22 16:48 CST. |
| STID | GLA 2019 | 183.174.228.180 | `cd001_stid_gla_full_wait_g0` | QUEUED_WAITING | GPU0 still busy at about 38 GiB used and 95% util; wait screen has not started training. |
| STID | GBA 2019 | 183.174.228.180 | `cd001_stid_gba_full_wait_g1` | QUEUED_WAITING | GPU1 still busy at about 38 GiB used and 100% util; wait screen has not started training. |

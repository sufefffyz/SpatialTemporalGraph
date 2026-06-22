# Remote Automation

This repository currently has three sibling projects and no unified runner yet, as noted in [PROJECT_LAYOUT.md](/Users/richardo/Desktop/STproject/SpatialTemporalGraph/PROJECT_LAYOUT.md). The scripts in this directory add a thin automation layer without changing any existing training code.

## What this gives you

- Local development stays in the Git repository on your laptop.
- Remote runs still happen on your Linux server.
- You can choose between:
  - manual remote pull + run
  - local push + remote trigger over SSH

## Files

- `scripts/remote_pull_and_run.sh`
  - Runs on the remote server.
  - Pulls a branch, switches into a subproject, records logs, and launches a command.
- `scripts/push_and_trigger_remote.sh`
  - Runs locally.
  - Pushes the current branch to GitHub and triggers the remote script over SSH.
- `scripts/remote.example.env`
  - Template for local defaults such as server host and remote repo path.

## Setup

1. Clone this repository on the remote server in a stable path.
2. Make sure the remote clone points to the same GitHub `origin`.
3. Copy `scripts/remote.example.env` to `.remote-run.env` in the repository root on your local machine and fill in the values.
4. Mark the scripts as executable:

```bash
chmod +x scripts/remote_pull_and_run.sh scripts/push_and_trigger_remote.sh
```

## Manual remote usage

This is the safest first step if you still want to confirm each run by hand on the server:

```bash
cd /absolute/path/to/SpatialTemporalGraph
bash scripts/remote_pull_and_run.sh \
  --project urban-traffic-benchmark \
  --conda-env graph_ml \
  --ref 0.5.8 \
  --mode bg \
  -- bash reproduce_experiments_models_dataset_path.sh
```

This will:

- `git fetch` + `git pull --ff-only`
- switch into `urban-traffic-benchmark/`
- save metadata and stdout under `~/.spatial-temporal-graph-runs/`
- start the run in the background

## Local push + remote trigger

Once `.remote-run.env` is set, you can launch the same workflow from your laptop:

```bash
cd /Users/richardo/Desktop/STproject/SpatialTemporalGraph
bash scripts/push_and_trigger_remote.sh \
  --project urban-traffic-benchmark \
  --conda-env graph_ml \
  --mode bg \
  -- bash reproduce_experiments_models_dataset_path.sh
```

For a one-off command:

```bash
bash scripts/push_and_trigger_remote.sh \
  --project urban-traffic-benchmark \
  --conda-env graph_ml \
  --mode bg \
  -- python run_single_experiment.py \
     --name debug_run \
     --dataset city_traffic_m_speed \
     --metric MAE \
     --prediction_horizon 12 \
     --direct_lookback_num_steps 48 \
     --device cuda:0
```

## Recommended workflow

1. Let Codex modify code locally in this repository.
2. Review the diff locally.
3. Commit the change.
4. Run `bash scripts/push_and_trigger_remote.sh ...`.
5. Watch logs on the remote server.

## Important behavior

- The local trigger script refuses to run if your local worktree has uncommitted changes, because the remote server can only see committed code.
- The remote script refuses to run if the remote repo is dirty, unless you pass `--allow-dirty`.
- The remote repo path should be an absolute path.
- The scripts do not assume anything about the subproject internals; they only `cd` into the selected subproject and execute your command there.

## Next automation step

If you want this to become fully unattended later, the clean next upgrade is to install a self-hosted GitHub Actions runner on the remote server. Then a push, tag, or manual workflow dispatch can launch jobs directly on the GPU machine without requiring an SSH trigger from your laptop.

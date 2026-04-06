#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/run_largest_dataset_queue.sh <STID|BigST> <gpu> <num_runs> <min_free_mb> <dataset1> [dataset2 ...]" >&2
  exit 1
fi

BASELINE="$1"
GPU="$2"
NUM_RUNS="$3"
MIN_FREE_MB="$4"
shift 4
DATASETS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  echo "gpu must be a non-negative integer, got: $GPU" >&2
  exit 1
fi

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -le 0 ]]; then
  echo "num_runs must be a positive integer, got: $NUM_RUNS" >&2
  exit 1
fi

if ! [[ "$MIN_FREE_MB" =~ ^[0-9]+$ ]] || [[ "$MIN_FREE_MB" -le 0 ]]; then
  echo "min_free_mb must be a positive integer, got: $MIN_FREE_MB" >&2
  exit 1
fi

if ! [[ "$CHECK_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$CHECK_INTERVAL" -le 0 ]]; then
  echo "CHECK_INTERVAL must be a positive integer, got: $CHECK_INTERVAL" >&2
  exit 1
fi

query_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' '
}

timestamp() {
  date "+%F %T"
}

echo "Queue started at $(timestamp)"
echo "Baseline        : $BASELINE"
echo "GPU             : $GPU"
echo "NUM_RUNS        : $NUM_RUNS"
echo "MIN_FREE_MB     : $MIN_FREE_MB"
echo "CHECK_INTERVAL  : $CHECK_INTERVAL"
echo "Datasets        : ${DATASETS[*]}"

for dataset in "${DATASETS[@]}"; do
  while true; do
    free_mb="$(query_free_mb)"
    if [[ -z "$free_mb" ]]; then
      echo "$(timestamp) Failed to query GPU $GPU free memory, retry in ${CHECK_INTERVAL}s"
      sleep "$CHECK_INTERVAL"
      continue
    fi

    if [[ "$free_mb" -ge "$MIN_FREE_MB" ]]; then
      echo "$(timestamp) GPU $GPU free=${free_mb}MB >= ${MIN_FREE_MB}MB, starting ${dataset}"
      break
    fi

    echo "$(timestamp) GPU $GPU free=${free_mb}MB < ${MIN_FREE_MB}MB, waiting for ${dataset}"
    sleep "$CHECK_INTERVAL"
  done

  bash scripts/run_largest_multiseed.sh "$BASELINE" "$dataset" "$GPU" "$NUM_RUNS"
  echo "$(timestamp) Finished ${BASELINE} on ${dataset}"
done

echo "Queue finished at $(timestamp)"

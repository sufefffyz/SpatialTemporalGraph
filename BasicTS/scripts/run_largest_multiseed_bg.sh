#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_largest_multiseed_bg.sh <STID|BigST> <CA|GBA|GLA|SD> [gpus] [num_runs]" >&2
  exit 1
fi

BASELINE="$1"
DATASET="$2"
GPUS="${3:-0}"
NUM_RUNS="${4:-${NUM_RUNS:-3}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${MULTISEED_BG_LOG_DIR:-$REPO_ROOT/profile_logs/bg_logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="${MULTISEED_BG_LOG_FILE:-$LOG_DIR/${BASELINE}_${DATASET}_multiseed_${NUM_RUNS}_${TIMESTAMP}.log}"
PID_FILE="${LOG_FILE%.log}.pid"

ENV_ARGS=()

for var_name in \
  SEEDS \
  SEED_FILE \
  SHARED_SEED_DIR \
  GENERATOR_SEED \
  SEED_LOW \
  SEED_HIGH \
  PROFILE_OUTPUT_DIR \
  BASICTS_SEED
do
  if [[ -n "${!var_name:-}" ]]; then
    ENV_ARGS+=("${var_name}=${!var_name}")
  fi
done

nohup env "${ENV_ARGS[@]}" \
  bash scripts/run_largest_multiseed.sh "$BASELINE" "$DATASET" "$GPUS" "$NUM_RUNS" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started background multi-seed run."
echo "PID      : $PID"
echo "PID file : $PID_FILE"
echo "Log file : $LOG_FILE"
echo "Follow   : tail -f $LOG_FILE"

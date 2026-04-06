#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_profiled_training.sh <config_path> [gpus] [tag]" >&2
  exit 1
fi

CFG_PATH="$1"
GPUS="${2:-0}"
TAG="${3:-$(basename "${CFG_PATH%.py}")_$(date +%Y%m%d-%H%M%S)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-$REPO_ROOT/profile_logs/$TAG}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/train.log"
SUMMARY_FILE="$OUTPUT_DIR/profile_summary.json"
PEAK_FILE="$OUTPUT_DIR/peak_gpu_memory_mb.txt"
echo "0" > "$PEAK_FILE"

monitor_gpu_memory() {
  local target_pid="$1"
  local gpu_list="$2"
  local peak_file="$3"
  local peak=0

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "" > "$peak_file"
    return 0
  fi

  local normalized_gpus="${gpu_list//,/ }"
  while kill -0 "$target_pid" 2>/dev/null; do
    for gpu in $normalized_gpus; do
      local mem
      mem="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
      if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > peak )); then
        peak="$mem"
        echo "$peak" > "$peak_file"
      fi
    done
    sleep 5
  done
}

echo "Config      : $CFG_PATH"
echo "GPUs        : $GPUS"
echo "Output dir  : $OUTPUT_DIR"
echo "Log file    : $LOG_FILE"

START_TS="$(date +%s)"
START_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python experiments/train.py -c "$CFG_PATH" -g "$GPUS" >"$LOG_FILE" 2>&1 &
TRAIN_PID=$!
monitor_gpu_memory "$TRAIN_PID" "$GPUS" "$PEAK_FILE" &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
EXIT_CODE=$?
set -e

wait "$MONITOR_PID" 2>/dev/null || true

END_TS="$(date +%s)"
END_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
DURATION="$((END_TS - START_TS))"
PEAK_GPU_MEM="$(cat "$PEAK_FILE" 2>/dev/null || true)"

if [[ -z "$PEAK_GPU_MEM" ]]; then
  PEAK_GPU_MEM_JSON="null"
else
  PEAK_GPU_MEM_JSON="$PEAK_GPU_MEM"
fi

cat >"$SUMMARY_FILE" <<EOF
{
  "config_path": "$CFG_PATH",
  "gpus": "$GPUS",
  "tag": "$TAG",
  "start_time_utc": "$START_ISO",
  "end_time_utc": "$END_ISO",
  "duration_seconds": $DURATION,
  "peak_gpu_memory_mb": $PEAK_GPU_MEM_JSON,
  "exit_code": $EXIT_CODE,
  "log_file": "$LOG_FILE"
}
EOF

echo "Profile summary saved to $SUMMARY_FILE"
cat "$SUMMARY_FILE"

exit "$EXIT_CODE"

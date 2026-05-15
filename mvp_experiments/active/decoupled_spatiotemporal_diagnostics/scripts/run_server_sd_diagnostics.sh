#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MVP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MVP_ROOT}/../../.." && pwd)"

DATASET_NAME="${DATASET_NAME:-SD_5min_full}"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${MVP_ROOT}/results/server_sd_${TIMESTAMP}}"
MAX_RUNS="${MAX_RUNS:-12}"
MOVING_WINDOW="${MOVING_WINDOW:-12}"
PEAK_Q="${PEAK_Q:-0.90}"

SEARCH_ROOTS=()
if [ "$#" -gt 0 ]; then
  SEARCH_ROOTS=("$@")
else
  SEARCH_ROOTS=(
    "${REPO_ROOT}/BasicTS/checkpoints"
  )
fi

cd "${REPO_ROOT}"

SEARCH_ARGS=()
for ROOT in "${SEARCH_ROOTS[@]}"; do
  SEARCH_ARGS+=(--search-root "${ROOT}")
done

python "${SCRIPT_DIR}/run_frequency_diagnostics.py" \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --moving-window "${MOVING_WINDOW}" \
  --peak-q "${PEAK_Q}" \
  --horizons 1 3 6 12 \
  --max-runs "${MAX_RUNS}" \
  --include "${DATASET_NAME}" \
  "${SEARCH_ARGS[@]}"

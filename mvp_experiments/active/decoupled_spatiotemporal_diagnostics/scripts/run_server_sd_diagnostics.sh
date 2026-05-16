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
FFT_CUTOFF_PERIOD="${FFT_CUTOFF_PERIOD:-${MOVING_WINDOW}}"
DECOMP_METHODS="${DECOMP_METHODS:-moving_average fft_lowpass}"
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
read -r -a DECOMP_METHOD_ARGS <<< "${DECOMP_METHODS}"

python "${SCRIPT_DIR}/run_frequency_diagnostics.py" \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --moving-window "${MOVING_WINDOW}" \
  --fft-cutoff-period "${FFT_CUTOFF_PERIOD}" \
  --decomp-methods "${DECOMP_METHOD_ARGS[@]}" \
  --peak-q "${PEAK_Q}" \
  --horizons 1 3 6 12 \
  --max-runs "${MAX_RUNS}" \
  --include "${DATASET_NAME}" \
  "${SEARCH_ARGS[@]}"

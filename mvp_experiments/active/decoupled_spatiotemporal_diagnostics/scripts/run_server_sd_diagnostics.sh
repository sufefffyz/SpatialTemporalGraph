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
CONDITION_HIGH_Q="${CONDITION_HIGH_Q:-0.75}"
CONDITION_RAMP_Q="${CONDITION_RAMP_Q:-0.90}"
JOINT_ST_CONDITIONS="${JOINT_ST_CONDITIONS:-high_volume peak ramp}"
JOINT_ST_WORKERS="${JOINT_ST_WORKERS:-1}"
HORIZONS="${HORIZONS:-1 2 3 4 5 6 7 8 9 10 11 12}"
INCLUDE_TOKENS="${INCLUDE_TOKENS:-${DATASET_NAME}_}"
ALIGNMENT_MAX_SHIFT="${ALIGNMENT_MAX_SHIFT:-3}"
ALIGNMENT_TIME_WINDOWS="${ALIGNMENT_TIME_WINDOWS:-0 1}"
ALIGNMENT_HOP_KS="${ALIGNMENT_HOP_KS:-0 1}"
ADJ_PATH="${ADJ_PATH:-}"
ALIGNMENT_DIRECTED="${ALIGNMENT_DIRECTED:-0}"
ALIGNMENT_ONLY="${ALIGNMENT_ONLY:-0}"

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
read -r -a HORIZON_ARGS <<< "${HORIZONS}"
read -r -a INCLUDE_TOKEN_ARGS <<< "${INCLUDE_TOKENS}"
read -r -a ALIGNMENT_TIME_WINDOW_ARGS <<< "${ALIGNMENT_TIME_WINDOWS}"
read -r -a ALIGNMENT_HOP_K_ARGS <<< "${ALIGNMENT_HOP_KS}"
read -r -a JOINT_ST_CONDITION_ARGS <<< "${JOINT_ST_CONDITIONS}"

INCLUDE_ARGS=()
for TOKEN in "${INCLUDE_TOKEN_ARGS[@]}"; do
  INCLUDE_ARGS+=(--include "${TOKEN}")
done

ADJ_ARGS=()
if [ -n "${ADJ_PATH}" ]; then
  ADJ_ARGS+=(--adj-path "${ADJ_PATH}")
fi
if [ "${ALIGNMENT_DIRECTED}" = "1" ] || [ "${ALIGNMENT_DIRECTED}" = "true" ]; then
  ADJ_ARGS+=(--alignment-directed)
fi
if [ "${ALIGNMENT_ONLY}" = "1" ] || [ "${ALIGNMENT_ONLY}" = "true" ]; then
  ADJ_ARGS+=(--alignment-only)
fi

python "${SCRIPT_DIR}/run_frequency_diagnostics.py" \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --moving-window "${MOVING_WINDOW}" \
  --fft-cutoff-period "${FFT_CUTOFF_PERIOD}" \
  --decomp-methods "${DECOMP_METHOD_ARGS[@]}" \
  --peak-q "${PEAK_Q}" \
  --condition-high-q "${CONDITION_HIGH_Q}" \
  --condition-ramp-q "${CONDITION_RAMP_Q}" \
  --joint-st-conditions "${JOINT_ST_CONDITION_ARGS[@]}" \
  --joint-st-workers "${JOINT_ST_WORKERS}" \
  --horizons "${HORIZON_ARGS[@]}" \
  --alignment-max-shift "${ALIGNMENT_MAX_SHIFT}" \
  --alignment-time-windows "${ALIGNMENT_TIME_WINDOW_ARGS[@]}" \
  --alignment-hop-ks "${ALIGNMENT_HOP_K_ARGS[@]}" \
  --max-runs "${MAX_RUNS}" \
  "${INCLUDE_ARGS[@]}" \
  "${ADJ_ARGS[@]}" \
  "${SEARCH_ARGS[@]}"

python "${SCRIPT_DIR}/plot_decoupled_diagnostics.py" \
  --input-dir "${OUTPUT_DIR}" \
  --plot-format png

if [[ "${ALIGNMENT_ONLY}" != "1" && "${ALIGNMENT_ONLY}" != "true" && " ${DECOMP_METHODS} " == *" moving_average "* && " ${DECOMP_METHODS} " == *" fft_lowpass "* ]]; then
  python "${SCRIPT_DIR}/summarize_decomposition_rankings.py" \
    --summary-csv "${OUTPUT_DIR}/decomposition_average_summary.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --baseline-method moving_average \
    --comparison-method fft_lowpass
fi

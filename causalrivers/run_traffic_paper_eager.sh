#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-benchmark_traffic_paper_baselines}"
DATASET_NAME="${DATASET_NAME:-city_traffic_m_volume__category__1_0}"
LABEL_TAG="${LABEL_TAG:-paperlike}"
N_VARS="${N_VARS:-3}"
SCORE_N_JOBS="${SCORE_N_JOBS:-32}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  STRATEGIES=("$@")
else
  STRATEGIES=(random confounder close root_cause)
fi

echo "Config      : $CONFIG_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Label tag   : $LABEL_TAG"
echo "n_vars      : $N_VARS"
echo "Strategies  : ${STRATEGIES[*]}"
echo "Score n_jobs: $SCORE_N_JOBS"
echo "Score chunk : $SCORE_CHUNK_SIZE"
echo

for strategy in "${STRATEGIES[@]}"; do
  label_path="datasets/traffic_${DATASET_NAME}/${strategy}_${N_VARS}_${LABEL_TAG}/${DATASET_NAME}.p"
  log_path="${LOG_DIR}/${strategy}_${N_VARS}_${LABEL_TAG}_eager.log"

  if [ ! -f "$label_path" ]; then
    echo "Missing label file: $label_path" >&2
    echo "Skip strategy: $strategy" >&2
    echo >&2
    continue
  fi

  echo "Running strategy: $strategy"
  echo "Label path      : $label_path"
  echo "Log             : $log_path"

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" benchmark.py --config-name "$CONFIG_NAME" \
    load_mode=eager \
    n_jobs=1 \
    score_n_jobs="$SCORE_N_JOBS" \
    score_chunk_size="$SCORE_CHUNK_SIZE" \
    label_path="$label_path" \
    2>&1 | tee "$log_path"

  echo
done

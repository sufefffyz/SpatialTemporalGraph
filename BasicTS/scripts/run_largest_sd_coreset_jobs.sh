#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: bash scripts/run_largest_sd_coreset_jobs.sh <gpu> <stid|gwnet> [ratio] [seed]}"
MODEL="${2:?usage: bash scripts/run_largest_sd_coreset_jobs.sh <gpu> <stid|gwnet> [ratio] [seed]}"
RATIO="${3:-0.1}"
SEED="${4:-2023}"
RUN_TS="${RUN_TS:-$(date '+%Y%m%d_%H%M%S')}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASICTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASICTS_ROOT}"

case "${MODEL}" in
  stid)
    FULL_CONFIG="baselines/STID/SD.py"
    CORESET_CONFIG="baselines/STID/SD_coreset.py"
    ;;
  gwnet)
    FULL_CONFIG="baselines/GWNet/SD_largest_aligned.py"
    CORESET_CONFIG="baselines/GWNet/SD_largest_aligned_coreset.py"
    ;;
  *)
    echo "Unsupported model: ${MODEL}. Use stid or gwnet." >&2
    exit 2
    ;;
esac

LOG_DIR="logs/st_coreset_largest_sd/${RUN_TS}/${MODEL}"
mkdir -p "${LOG_DIR}"

run_train() {
  local label="$1"
  local config="$2"
  shift 2
  echo "[$(date '+%F %T')] START ${MODEL}/${label} on GPU ${GPU}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export WANDB_RUN_GROUP="st_coreset_largest_sd_${RUN_TS}_${MODEL}"
    "$@" "${PYTHON_BIN}" experiments/train.py -c "${config}" -g "${GPU}"
  ) 2>&1 | tee "${LOG_DIR}/${label}.log"
  echo "[$(date '+%F %T')] END ${MODEL}/${label}"
}

STRATEGIES=(
  random
  temporal
  kcenter
  temporal_kcenter
  temporal_kcenter_difficulty
)

for STRATEGY in "${STRATEGIES[@]}"; do
  TAG="stcoreset_${RUN_TS}_${MODEL}_${STRATEGY}_r${RATIO}"
  run_train "${STRATEGY}_r${RATIO}" "${CORESET_CONFIG}" \
    env \
      BASICTS_RUN_TAG="${TAG}" \
      BASICTS_SEED="${SEED}" \
      CORESET_STRATEGY="${STRATEGY}" \
      CORESET_RATIO="${RATIO}" \
      CORESET_SEED="${SEED}" \
      CORESET_TEMPORAL_PERIOD=96
done

run_train "full" "${FULL_CONFIG}" \
  env \
    BASICTS_RUN_TAG="stcoreset_${RUN_TS}_${MODEL}_full" \
    BASICTS_SEED="${SEED}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <MODEL: STID|CycleNet> <GPU_ID>" >&2
  exit 2
fi

MODEL="$1"
GPU_ID="$2"
REPO_ROOT="${REPO_ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph}"
RUN_TAG="${FAST_LTSF_RUN_TAG:-fast_ltsf_full_representative}"
LOG_DIR="${REPO_ROOT}/mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/outputs/logs/fast_ltsf_representative_full"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}/BasicTS"

export WANDB_MODE="${WANDB_MODE:-offline}"
export BASICTS_SEED="${BASICTS_SEED:-42}"
export FAST_LTSF_NUM_EPOCHS="${FAST_LTSF_NUM_EPOCHS:-50}"
export FAST_LTSF_PATIENCE="${FAST_LTSF_PATIENCE:-30}"
export FAST_LTSF_BATCH_SIZE="${FAST_LTSF_BATCH_SIZE:-64}"
export FAST_LTSF_RUN_TAG="${RUN_TAG}"

DATASETS=(SD GBA GLA)
HORIZONS=(48 96 192 672)

for DATASET in "${DATASETS[@]}"; do
  for HORIZON in "${HORIZONS[@]}"; do
    CONFIG="baselines/FaSTLTSF/generated/${MODEL}_${DATASET}_L96_H${HORIZON}_fast_ltsf_full_representative.py"
    LOG="${LOG_DIR}/${MODEL}_${DATASET}_L96_H${HORIZON}.log"
    echo "[$(date '+%F %T')] START ${MODEL} ${DATASET} L96 H${HORIZON}" | tee -a "${LOG}"
    python experiments/train.py -c "${CONFIG}" -g "${GPU_ID}" 2>&1 | tee -a "${LOG}"
    echo "[$(date '+%F %T')] END ${MODEL} ${DATASET} L96 H${HORIZON}" | tee -a "${LOG}"
  done
done

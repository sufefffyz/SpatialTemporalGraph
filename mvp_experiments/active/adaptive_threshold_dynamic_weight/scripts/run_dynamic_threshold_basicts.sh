#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
MODE="${2:-hard}"
GPU="${3:-1}"
RUN_TAG="${4:-dynamic_threshold_20260519}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_dynamic_weight}"

if [[ "${MODEL}" != "gwnet" && "${MODEL}" != "dcrnn" && "${MODEL}" != "stgcn" && "${MODEL}" != "flownet_soft" ]]; then
  echo "Usage: $0 {gwnet|dcrnn|stgcn|flownet_soft} {hard|soft} [gpu_id] [run_tag]" >&2
  exit 2
fi
if [[ "${MODE}" != "hard" && "${MODE}" != "soft" ]]; then
  echo "MODE must be hard or soft." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/${RUN_TAG}/${MODE}"
mkdir -p "${LOG_ROOT}"

case "${MODEL}" in
  gwnet)
    CFG="baselines/GWNet/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdGraphWaveNet"
    ;;
  dcrnn)
    CFG="baselines/DCRNN/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdDCRNN"
    ;;
  stgcn)
    CFG="baselines/STGCN/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdSTGCN"
    ;;
  flownet_soft)
    CFG="baselines/FlowNet/SD_osrm_soft_dense.py"
    MODEL_NAME="BasicTSFlowNet"
    MODE="soft"
    ;;
esac

cd "${BASICTS_DIR}"
export BASICTS_DATA_NAME="${BASICTS_DATA_NAME:-SD}"
export BASICTS_RUN_TAG="${RUN_TAG}"
export BASICTS_SEED="${BASICTS_SEED:-2023}"
export DYNAMIC_GRAPH_MODE="${MODE}"
export WANDB_PROJECT="${PROJECT}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="sd_dynamic_threshold_${MODE}_${MODEL}"
export WANDB_TAGS="adaptive-threshold,sd,dynamic-threshold,${MODE},${MODEL}"
export WANDB_NAME="${MODEL_NAME}_${BASICTS_DATA_NAME}_${MODE}_${RUN_TAG}"

log_file="${LOG_ROOT}/${MODEL}_${MODE}.log"
{
  echo "model,mode,dataset,gpu,config,log"
  echo "${MODEL},${MODE},${BASICTS_DATA_NAME},${GPU},${CFG},${log_file}"
} > "${LOG_ROOT}/manifest_${MODEL}_${MODE}.csv"

echo "[$(date '+%F %T')] Starting ${MODEL} ${MODE} on ${BASICTS_DATA_NAME} GPU ${GPU}" | tee -a "${log_file}"
python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
echo "[$(date '+%F %T')] Finished ${MODEL} ${MODE}" | tee -a "${log_file}"

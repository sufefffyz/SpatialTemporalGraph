#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-0}"
RUN_TAG="${3:-effres_k64_20260526}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_topk_prior}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-40}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-12000}"

if [[ "${MODEL}" != "gwnet" && "${MODEL}" != "dcrnn" && "${MODEL}" != "dynamic_gwnet" && "${MODEL}" != "dynamic_dcrnn" && "${MODEL}" != "dynamic_stgcn" ]]; then
  echo "Usage: $0 {gwnet|dcrnn|dynamic_gwnet|dynamic_dcrnn|dynamic_stgcn} [gpu_id] [run_tag]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
DATASET_NAME="SD_EFFRESTOPK_K064"
GRAPH_TAG="effresK64"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/effective_resistance/${RUN_TAG}/${MODEL}"
mkdir -p "${LOG_ROOT}"

case "${MODEL}" in
  gwnet)
    CFG="baselines/GWNet/SD_topk_prior.py"
    MODEL_NAME="GraphWaveNet"
    DATA_NAME="${DATASET_NAME}"
    EXTRA_GROUP="fixed"
    ;;
  dcrnn)
    CFG="baselines/DCRNN/SD_osrm_gaussian_global.py"
    MODEL_NAME="DCRNN"
    DATA_NAME="${DATASET_NAME}"
    EXTRA_GROUP="fixed"
    ;;
  dynamic_gwnet)
    CFG="baselines/GWNet/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdGraphWaveNet"
    DATA_NAME="SD"
    EXTRA_GROUP="dynamic"
    ;;
  dynamic_dcrnn)
    CFG="baselines/DCRNN/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdDCRNN"
    DATA_NAME="SD"
    EXTRA_GROUP="dynamic"
    ;;
  dynamic_stgcn)
    CFG="baselines/STGCN/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdSTGCN"
    DATA_NAME="SD"
    EXTRA_GROUP="dynamic"
    ;;
esac

wait_for_gpu_idle() {
  if [[ "${WAIT_FOR_GPU_IDLE}" != "1" ]]; then
    return
  fi
  while true; do
    util="$(nvidia-smi --id="${GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    mem_used="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ -n "${util}" && -n "${mem_used}" && "${util}" -le "${GPU_UTIL_LIMIT}" && "${mem_used}" -le "${GPU_MEM_LIMIT_MB}" ]]; then
      echo "[$(date '+%F %T')] GPU ${GPU} util ${util}% <= ${GPU_UTIL_LIMIT}% and mem ${mem_used}MiB <= ${GPU_MEM_LIMIT_MB}MiB, starting."
      return
    fi
    echo "[$(date '+%F %T')] Waiting for GPU ${GPU}: util=${util}% / mem=${mem_used}MiB; limits util<=${GPU_UTIL_LIMIT}%, mem<=${GPU_MEM_LIMIT_MB}MiB"
    sleep 300
  done
}

log_file="${LOG_ROOT}/${MODEL}_${GRAPH_TAG}.log"
{
  echo "model,graph,dataset,gpu,config,log"
  echo "${MODEL},${GRAPH_TAG},${DATA_NAME},${GPU},${CFG},${log_file}"
} > "${LOG_ROOT}/manifest_${MODEL}_${GRAPH_TAG}.csv"

cd "${BASICTS_DIR}"
if [[ ! -e "datasets/${DATASET_NAME}/adj_mx.pkl" ]]; then
  echo "Missing datasets/${DATASET_NAME}/adj_mx.pkl. Build the effective-resistance graph first." >&2
  exit 1
fi

wait_for_gpu_idle

export BASICTS_DATA_NAME="${DATA_NAME}"
export BASICTS_GRAPH_TAG="${GRAPH_TAG}"
export BASICTS_RUN_TAG="${RUN_TAG}"
export BASICTS_SEED="${BASICTS_SEED:-2023}"
export BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS:-100}"
export WANDB_PROJECT="${PROJECT}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="sd_effective_resistance_${MODEL}_${EXTRA_GROUP}"
export WANDB_TAGS="adaptive-threshold,sd,effective-resistance,k64,${MODEL},${EXTRA_GROUP}"
export WANDB_NAME="${MODEL_NAME}_SD_${GRAPH_TAG}_${EXTRA_GROUP}_${RUN_TAG}"

if [[ "${EXTRA_GROUP}" == "dynamic" ]]; then
  export DYNAMIC_GRAPH_MODE="hard"
  export DYNAMIC_GRAPH_WEIGHT_MODE="${DYNAMIC_GRAPH_WEIGHT_MODE:-binary}"
  export DYNAMIC_GRAPH_TARGET_AVG_DEGREE="${DYNAMIC_GRAPH_TARGET_AVG_DEGREE:-64}"
  export DYNAMIC_GRAPH_CANDIDATE_ADJ="datasets/${DATASET_NAME}/adj_mx.pkl"
  export DYNAMIC_GWNET_ADDAPTADJ="${DYNAMIC_GWNET_ADDAPTADJ:-0}"
fi

echo "[$(date '+%F %T')] Starting ${MODEL} ${GRAPH_TAG} on GPU ${GPU}" | tee -a "${log_file}"
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
echo "[$(date '+%F %T')] Finished ${MODEL} ${GRAPH_TAG}" | tee -a "${log_file}"

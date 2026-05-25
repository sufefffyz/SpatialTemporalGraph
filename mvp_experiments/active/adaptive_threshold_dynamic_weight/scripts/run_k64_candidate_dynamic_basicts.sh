#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
GRAPH="${2:-}"
GPU="${3:-0}"
RUN_TAG="${4:-dynamic_threshold_k64_candidate_20260525}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_topk_prior}"

if [[ "${MODEL}" != "gwnet" && "${MODEL}" != "dcrnn" ]]; then
  echo "Usage: $0 {gwnet|dcrnn} {osrmK64|gspK64} [gpu_id] [run_tag]" >&2
  exit 2
fi
case "${GRAPH}" in
  osrmK64)
    CANDIDATE_DATASET="SD_OSRMTOPK_K064"
    ;;
  gspK64)
    CANDIDATE_DATASET="SD_GSPTOPK_K064"
    ;;
  *)
    echo "GRAPH must be osrmK64 or gspK64." >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/dynamic_threshold/${RUN_TAG}/k64_candidate/${MODEL}"
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
esac

log_file="${LOG_ROOT}/${MODEL}_${GRAPH}.log"
{
  echo "model,graph,dataset,candidate_adj,gpu,config,log"
  echo "${MODEL},${GRAPH},SD,datasets/${CANDIDATE_DATASET}/adj_mx.pkl,${GPU},${CFG},${log_file}"
} > "${LOG_ROOT}/manifest_${MODEL}_${GRAPH}.csv"

cd "${BASICTS_DIR}"
export BASICTS_DATA_NAME="SD"
export BASICTS_GRAPH_TAG="${GRAPH}"
export BASICTS_RUN_TAG="${RUN_TAG}"
export BASICTS_SEED="${BASICTS_SEED:-2023}"
export BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS:-100}"
export DYNAMIC_GRAPH_MODE="hard"
export DYNAMIC_GRAPH_WEIGHT_MODE="${DYNAMIC_GRAPH_WEIGHT_MODE:-binary}"
export DYNAMIC_GRAPH_TARGET_AVG_DEGREE="${DYNAMIC_GRAPH_TARGET_AVG_DEGREE:-64}"
export DYNAMIC_GRAPH_CANDIDATE_ADJ="datasets/${CANDIDATE_DATASET}/adj_mx.pkl"
export DYNAMIC_GWNET_ADDAPTADJ="${DYNAMIC_GWNET_ADDAPTADJ:-0}"
export WANDB_PROJECT="${PROJECT}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="sd_dynamic_threshold_k64_candidate_${MODEL}"
export WANDB_TAGS="adaptive-threshold,sd,dynamic-threshold,k64-candidate,${MODEL},${GRAPH}"
export WANDB_NAME="${MODEL_NAME}_SD_${GRAPH}_dynamic_k64_candidate_${RUN_TAG}"

echo "[$(date '+%F %T')] Starting ${MODEL} dynamic K64 candidate ${GRAPH} on GPU ${GPU}" | tee -a "${log_file}"
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
echo "[$(date '+%F %T')] Finished ${MODEL} dynamic K64 candidate ${GRAPH}" | tee -a "${log_file}"

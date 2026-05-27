#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
GRAPH="${2:-}"
GPU="${3:-0}"
RUN_TAG="${4:-largest_aligned_priority_$(date '+%Y%m%d_%H%M%S')}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_largest_aligned}"

if [[ "${MODEL}" != "gwnet" && "${MODEL}" != "dcrnn" && "${MODEL}" != "stgcn" ]]; then
  echo "Usage: $0 {gwnet|dcrnn|stgcn} {original|osrmK64|gspK64|osrmK64Addapt|gspK64Addapt|dynFull|dynOsrmK64|dynGspK64|dynFullAddapt|dynOsrmK64Addapt|dynGspK64Addapt} [gpu_id] [run_tag]" >&2
  exit 2
fi

ADDAPT="0"
CANDIDATE_DATASET=""
case "${GRAPH}" in
  original)
    DATASET_NAME="SD"
    GRAPH_TAG="original"
    GRAPH_KIND="original"
    DYNAMIC="0"
    ;;
  osrmK64)
    DATASET_NAME="SD_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="topk_prior"
    DYNAMIC="0"
    ;;
  osrmK64Addapt)
    DATASET_NAME="SD_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="topk_prior"
    DYNAMIC="0"
    ADDAPT="1"
    ;;
  gspK64)
    DATASET_NAME="SD_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="topk_prior"
    DYNAMIC="0"
    ;;
  gspK64Addapt)
    DATASET_NAME="SD_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="topk_prior"
    DYNAMIC="0"
    ADDAPT="1"
    ;;
  dynOsrmK64)
    DATASET_NAME="SD"
    CANDIDATE_DATASET="SD_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ;;
  dynFull)
    DATASET_NAME="SD"
    GRAPH_TAG="fullPair"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ;;
  dynGspK64)
    DATASET_NAME="SD"
    CANDIDATE_DATASET="SD_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ;;
  dynFullAddapt)
    DATASET_NAME="SD"
    GRAPH_TAG="fullPair"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ADDAPT="1"
    ;;
  dynOsrmK64Addapt)
    DATASET_NAME="SD"
    CANDIDATE_DATASET="SD_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ADDAPT="1"
    ;;
  dynGspK64Addapt)
    DATASET_NAME="SD"
    CANDIDATE_DATASET="SD_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ADDAPT="1"
    ;;
  *)
    echo "GRAPH must be one of: original, osrmK64, gspK64, osrmK64Addapt, gspK64Addapt, dynFull, dynOsrmK64, dynGspK64, dynFullAddapt, dynOsrmK64Addapt, dynGspK64Addapt" >&2
    exit 2
    ;;
esac

if [[ "${ADDAPT}" == "1" && "${MODEL}" != "gwnet" ]]; then
  echo "Addaptive adjacency variants are only valid for GWNet, got MODEL=${MODEL} GRAPH=${GRAPH}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/largest_aligned/${RUN_TAG}/${MODEL}"
mkdir -p "${LOG_ROOT}"

case "${MODEL}:${GRAPH_KIND}" in
  gwnet:original)
    CFG="baselines/GWNet/SD_largest_aligned.py"
    MODEL_NAME="GraphWaveNet"
    ;;
  gwnet:topk_prior)
    CFG="baselines/GWNet/SD_topk_prior_largest_aligned.py"
    MODEL_NAME="GraphWaveNet"
    ;;
  gwnet:dynamic_threshold)
    CFG="baselines/GWNet/SD_dynamic_threshold_largest_aligned.py"
    MODEL_NAME="DynamicThresholdGraphWaveNet"
    ;;
  dcrnn:original)
    CFG="baselines/DCRNN/SD_largest_aligned.py"
    MODEL_NAME="DCRNN"
    ;;
  dcrnn:topk_prior)
    CFG="baselines/DCRNN/SD_topk_prior_largest_aligned.py"
    MODEL_NAME="DCRNN"
    ;;
  dcrnn:dynamic_threshold)
    CFG="baselines/DCRNN/SD_dynamic_threshold_largest_aligned.py"
    MODEL_NAME="DynamicThresholdDCRNN"
    ;;
  stgcn:original)
    CFG="baselines/STGCN/SD_largest_aligned.py"
    MODEL_NAME="STGCNChebGraphConv"
    ;;
  stgcn:topk_prior)
    CFG="baselines/STGCN/SD_topk_prior_largest_aligned.py"
    MODEL_NAME="STGCNChebGraphConv"
    ;;
  stgcn:dynamic_threshold)
    CFG="baselines/STGCN/SD_dynamic_threshold_largest_aligned.py"
    MODEL_NAME="DynamicThresholdSTGCN"
    ;;
  *)
    echo "Unsupported MODEL/GRAPH_KIND combination: ${MODEL}:${GRAPH_KIND}" >&2
    exit 2
    ;;
esac

log_file="${LOG_ROOT}/${MODEL}_${GRAPH}.log"
{
  echo "model,graph,dataset,candidate_adj,gpu,config,run_tag,log"
  if [[ "${DYNAMIC}" == "1" && -n "${CANDIDATE_DATASET}" ]]; then
    echo "${MODEL},${GRAPH},${DATASET_NAME},datasets/${CANDIDATE_DATASET}/adj_mx.pkl,${GPU},${CFG},${RUN_TAG},${log_file}"
  else
    echo "${MODEL},${GRAPH},${DATASET_NAME},,${GPU},${CFG},${RUN_TAG},${log_file}"
  fi
} > "${LOG_ROOT}/manifest_${MODEL}_${GRAPH}.csv"

cd "${BASICTS_DIR}"
export BASICTS_DATA_NAME="${DATASET_NAME}"
export BASICTS_GRAPH_TAG="${GRAPH_TAG}"
export BASICTS_RUN_TAG="${RUN_TAG}"
export BASICTS_SEED="${BASICTS_SEED:-2023}"
export BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS:-100}"
export BASICTS_PATIENCE="${BASICTS_PATIENCE:-30}"
export BASICTS_BATCH_SIZE="${BASICTS_BATCH_SIZE:-64}"
export WANDB_PROJECT="${PROJECT}"
export WANDB_MODE="${WANDB_MODE:-online}"
ADDAPT_TAG="no-addaptadj"
if [[ "${ADDAPT}" == "1" ]]; then
  ADDAPT_TAG="addaptadj"
fi
CANDIDATE_TAG="full-pair"
if [[ -n "${CANDIDATE_DATASET}" ]]; then
  CANDIDATE_TAG="candidate-k64"
fi
export WANDB_RUN_GROUP="sd_largest_aligned_${MODEL}_${GRAPH_KIND}_${CANDIDATE_TAG}_${ADDAPT_TAG}"
export WANDB_TAGS="adaptive-threshold,sd,largest-aligned,${MODEL},${GRAPH_KIND},${GRAPH_TAG},${CANDIDATE_TAG},${ADDAPT_TAG}"
export WANDB_NAME="${MODEL_NAME}_${DATASET_NAME}_${GRAPH}_${GRAPH_KIND}_${RUN_TAG}"

if [[ "${MODEL}" == "gwnet" ]]; then
  export GWNET_ADDAPTADJ="${ADDAPT}"
fi

if [[ "${DYNAMIC}" == "1" ]]; then
  export DYNAMIC_GRAPH_MODE="hard"
  export DYNAMIC_GRAPH_WEIGHT_MODE="${DYNAMIC_GRAPH_WEIGHT_MODE:-binary}"
  export DYNAMIC_GRAPH_TARGET_AVG_DEGREE="${DYNAMIC_GRAPH_TARGET_AVG_DEGREE:-64}"
  if [[ -n "${CANDIDATE_DATASET}" ]]; then
    export DYNAMIC_GRAPH_CANDIDATE_ADJ="datasets/${CANDIDATE_DATASET}/adj_mx.pkl"
  else
    unset DYNAMIC_GRAPH_CANDIDATE_ADJ
  fi
  export DYNAMIC_GWNET_ADDAPTADJ="${ADDAPT}"
fi

echo "[$(date '+%F %T')] Starting ${MODEL} ${GRAPH} (${GRAPH_KIND}) on GPU ${GPU}" | tee -a "${log_file}"
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
echo "[$(date '+%F %T')] Finished ${MODEL} ${GRAPH} (${GRAPH_KIND})" | tee -a "${log_file}"

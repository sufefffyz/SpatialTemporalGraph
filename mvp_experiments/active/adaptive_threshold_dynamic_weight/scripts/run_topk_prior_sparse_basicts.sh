#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-0}"
RUN_TAG="${3:-topk_prior_sparse_20260525}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_topk_prior}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-0}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-40}"

if [[ "${MODEL}" != "sparsegwnet" && "${MODEL}" != "sparsedcrnn" ]]; then
  echo "Usage: $0 {sparsegwnet|sparsedcrnn} [gpu_id] [run_tag]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/topk_prior/${RUN_TAG}/${MODEL}"
mkdir -p "${LOG_ROOT}"

case "${MODEL}" in
  sparsegwnet)
    CFG="baselines/GWNet/SD_topk_prior_sparse.py"
    MODEL_NAME="SparseGraphWaveNet"
    ;;
  sparsedcrnn)
    CFG="baselines/DCRNN/SD_topk_prior_sparse.py"
    MODEL_NAME="SparseDCRNN"
    ;;
esac

DATASETS=(
  "osrmK32:SD_OSRMTOPK_K032"
  "gspK32:SD_GSPTOPK_K032"
  "osrmK64:SD_OSRMTOPK_K064"
  "gspK64:SD_GSPTOPK_K064"
)

wait_for_gpu_idle() {
  if [[ "${WAIT_FOR_GPU_IDLE}" != "1" ]]; then
    return
  fi
  while true; do
    util="$(nvidia-smi --id="${GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ -n "${util}" && "${util}" -le "${GPU_UTIL_LIMIT}" ]]; then
      echo "[$(date '+%F %T')] GPU ${GPU} util ${util}% <= ${GPU_UTIL_LIMIT}%, starting."
      return
    fi
    echo "[$(date '+%F %T')] Waiting for GPU ${GPU}: util=${util}% > ${GPU_UTIL_LIMIT}%"
    sleep 300
  done
}

{
  echo "model,graph_tag,dataset,gpu,config,log"
  for item in "${DATASETS[@]}"; do
    graph_tag="${item%%:*}"
    dataset_name="${item##*:}"
    log_file="${LOG_ROOT}/${MODEL}_${graph_tag}.log"
    echo "${MODEL},${graph_tag},${dataset_name},${GPU},${CFG},${log_file}"
  done
} > "${LOG_ROOT}/manifest.csv"

cd "${BASICTS_DIR}"
wait_for_gpu_idle

for item in "${DATASETS[@]}"; do
  graph_tag="${item%%:*}"
  dataset_name="${item##*:}"
  log_file="${LOG_ROOT}/${MODEL}_${graph_tag}.log"
  export BASICTS_DATA_NAME="${dataset_name}"
  export BASICTS_GRAPH_TAG="${graph_tag}"
  export BASICTS_RUN_TAG="${RUN_TAG}"
  export BASICTS_SEED="${BASICTS_SEED:-2023}"
  export BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS:-100}"
  export WANDB_PROJECT="${PROJECT}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_RUN_GROUP="sd_topk_prior_${MODEL}_fixed"
  export WANDB_TAGS="adaptive-threshold,sd,topk-prior,${MODEL},${graph_tag}"
  export WANDB_NAME="${MODEL_NAME}_${dataset_name}_${graph_tag}_${RUN_TAG}"

  echo "[$(date '+%F %T')] Starting ${MODEL} ${graph_tag} (${dataset_name}) on GPU ${GPU}" | tee -a "${log_file}"
  /home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] Finished ${MODEL} ${graph_tag} (${dataset_name})" | tee -a "${log_file}"
done

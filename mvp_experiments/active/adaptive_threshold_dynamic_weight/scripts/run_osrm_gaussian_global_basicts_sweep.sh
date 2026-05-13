#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-1}"
RUN_TAG="${3:-osrm_gaussian_global_20260513}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_dynamic_weight}"

if [[ "${MODEL}" != "gwnet" && "${MODEL}" != "dcrnn" ]]; then
  echo "Usage: $0 {gwnet|dcrnn} [gpu_id] [run_tag]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/osrm_gaussian_global/${RUN_TAG}/${MODEL}"
mkdir -p "${LOG_ROOT}"

case "${MODEL}" in
  gwnet)
    CFG="baselines/GWNet/SD_osrm_gaussian_global.py"
    MODEL_NAME="GraphWaveNet"
    ;;
  dcrnn)
    CFG="baselines/DCRNN/SD_osrm_gaussian_global.py"
    MODEL_NAME="DCRNN"
    ;;
esac

DATASETS=(
  "beta0p25:SD_OSRMGG_B025"
  "beta0p50:SD_OSRMGG_B050"
  "beta1p00:SD_OSRMGG_B100"
  "beta1p50:SD_OSRMGG_B150"
  "beta2p00:SD_OSRMGG_B200"
)

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
for item in "${DATASETS[@]}"; do
  graph_tag="${item%%:*}"
  dataset_name="${item##*:}"
  log_file="${LOG_ROOT}/${MODEL}_${graph_tag}.log"
  export BASICTS_DATA_NAME="${dataset_name}"
  export BASICTS_GRAPH_TAG="${graph_tag}"
  export BASICTS_RUN_TAG="${RUN_TAG}"
  export BASICTS_SEED="${BASICTS_SEED:-2023}"
  export WANDB_PROJECT="${PROJECT}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_RUN_GROUP="sd_osrm_gaussian_global_${MODEL}"
  export WANDB_TAGS="adaptive-threshold,sd,osrm-gaussian-global,${MODEL},${graph_tag}"
  export WANDB_NAME="${MODEL_NAME}_${dataset_name}_${graph_tag}_${RUN_TAG}"

  echo "[$(date '+%F %T')] Starting ${MODEL} ${graph_tag} (${dataset_name}) on GPU ${GPU}" | tee -a "${log_file}"
  python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] Finished ${MODEL} ${graph_tag} (${dataset_name})" | tee -a "${log_file}"
done

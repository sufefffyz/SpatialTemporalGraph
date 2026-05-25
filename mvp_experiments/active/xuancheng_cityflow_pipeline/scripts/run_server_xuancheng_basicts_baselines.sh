#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph}"
BASICTS_DIR="${BASICTS_DIR:-${REPO_ROOT}/BasicTS}"
GPU="${GPU:-0}"
MODELS="${MODELS:-gwnet,stgcn}"
DATASETS="${DATASETS:-XCHENG_10S_FLOW,XCHENG_10S_STOCK,XCHENG_5MIN_FLOW,XCHENG_5MIN_STOCK}"
RUN_TAG="${RUN_TAG:-xuancheng_2x2_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${BASICTS_DIR}/logs/xuancheng_cityflow/${RUN_TAG}}"
BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS:-50}"
BASICTS_BATCH_SIZE="${BASICTS_BATCH_SIZE:-16}"
WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "${LOG_ROOT}"
cd "${BASICTS_DIR}"

echo "[info] Xuancheng BasicTS baseline queue"
echo "[info] basicts_dir=${BASICTS_DIR}"
echo "[info] datasets=${DATASETS}"
echo "[info] models=${MODELS}"
echo "[info] gpu=${GPU} epochs=${BASICTS_NUM_EPOCHS} batch=${BASICTS_BATCH_SIZE}"
echo "[info] log_root=${LOG_ROOT}"

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"

MANIFEST="${LOG_ROOT}/manifest.csv"
echo "model,dataset,gpu,epochs,batch_size,config,log" > "${MANIFEST}"

for DATASET in "${DATASET_LIST[@]}"; do
  for MODEL in "${MODEL_LIST[@]}"; do
    case "${MODEL}" in
      gwnet)
        CFG="baselines/GWNet/XCHENG.py"
        ;;
      sparsegwnet)
        CFG="baselines/GWNet/XCHENG_SPARSE.py"
        ;;
      stgcn)
        CFG="baselines/STGCN/XCHENG.py"
        ;;
      *)
        echo "[error] unsupported model: ${MODEL}" >&2
        exit 2
        ;;
    esac

    LOG_FILE="${LOG_ROOT}/${DATASET}_${MODEL}.log"
    echo "${MODEL},${DATASET},${GPU},${BASICTS_NUM_EPOCHS},${BASICTS_BATCH_SIZE},${CFG},${LOG_FILE}" >> "${MANIFEST}"
    echo "[$(date '+%F %T')] start model=${MODEL} dataset=${DATASET}" | tee -a "${LOG_FILE}"
    BASICTS_DATA_NAME="${DATASET}" \
    BASICTS_RUN_TAG="${RUN_TAG}" \
    BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS}" \
    BASICTS_BATCH_SIZE="${BASICTS_BATCH_SIZE}" \
    WANDB_MODE="${WANDB_MODE}" \
      python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${LOG_FILE}"
    echo "[$(date '+%F %T')] done model=${MODEL} dataset=${DATASET}" | tee -a "${LOG_FILE}"
  done
done

echo "[done] all queued Xuancheng baselines finished"

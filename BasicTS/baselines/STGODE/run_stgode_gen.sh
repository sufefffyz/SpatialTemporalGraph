#!/usr/bin/env bash

set -e

SCRIPT="baselines/STGODE/generate_matrices.py"   # 你的 python 脚本名
LOG_DIR="logs/stgode_gen"
DATASETS=(PEMS04 PEMS08 PEMS03 PEMS07)

mkdir -p "${LOG_DIR}"

echo "Starting STGODE matrix generation..."
echo "Logs saved to ${LOG_DIR}"

for DATASET in "${DATASETS[@]}"; do
  LOG_FILE="${LOG_DIR}/${DATASET}.log"

  echo "Launching ${DATASET}, log -> ${LOG_FILE}"

  # 后台运行 + 独立日志
  nohup python "${SCRIPT}" \
      --dataset "${DATASET}" \
      > "${LOG_FILE}" 2>&1 &

done

echo "All jobs launched in background."
echo "Use: tail -f logs/stgode_gen/<DATASET>.log to check progress"

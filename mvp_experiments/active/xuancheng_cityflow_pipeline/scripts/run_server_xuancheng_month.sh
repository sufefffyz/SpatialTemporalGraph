#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/road_agg_30d_fixed_time}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/road_agg_30d_fixed_time_$(date +%Y%m%d_%H%M%S)}"
DURATION="${DURATION:-86400}"
BUCKET_SECONDS="${BUCKET_SECONDS:-60}"
THREAD_NUM="${THREAD_NUM:-1}"
TL_MODE="${TL_MODE:-fixed_time}"
FILTER_FLOW="${FILTER_FLOW:-1}"
DOWNLOAD_FIRST="${DOWNLOAD_FIRST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DEFAULT_CITYFLOW_PYTHON="/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_CITYFLOW_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_CITYFLOW_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DATES=(
  2023-04-01
  2023-04-02
  2023-04-03
  2023-04-04
  2023-04-05
  2023-04-06
  2023-04-07
  2023-04-08
  2023-04-09
  2023-04-10
  2023-04-11
  2023-04-12
  2023-04-13
  2023-04-14
  2023-04-15
  2023-04-16
  2023-04-17
  2023-04-18
  2023-04-19
  2023-04-20
  2023-04-21
  2023-04-22
  2023-04-23
  2023-04-24
  2023-04-25
  2023-04-26
  2023-04-27
  2023-04-28
  2023-04-29
  2023-04-30
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] data_root=${DATA_ROOT}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] log_dir=${LOG_DIR}"
echo "[info] duration=${DURATION} bucket=${BUCKET_SECONDS} tl_mode=${TL_MODE} filter_flow=${FILTER_FLOW}"
echo "[info] python=${PYTHON_BIN}"

if [[ "${DOWNLOAD_FIRST}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/download_xuancheng_figshare.py" \
    --data-root "${DATA_ROOT}" \
    --all-days \
    --retries "${DOWNLOAD_RETRIES}"
fi

for DATE in "${DATES[@]}"; do
  NPZ_PATH="${OUTPUT_DIR}/xuancheng_${DATE}_road_agg_${BUCKET_SECONDS}s.npz"
  CSV_PATH="${OUTPUT_DIR}/xuancheng_${DATE}_road_agg_${BUCKET_SECONDS}s.csv.gz"
  LOG_PATH="${LOG_DIR}/${DATE}.log"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${NPZ_PATH}" && -s "${CSV_PATH}" ]]; then
    echo "[skip] ${DATE}: existing ${NPZ_PATH}"
    continue
  fi

  echo "[run] ${DATE} -> ${LOG_PATH}"
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  DURATION="${DURATION}" \
  BUCKET_SECONDS="${BUCKET_SECONDS}" \
  THREAD_NUM="${THREAD_NUM}" \
  TL_MODE="${TL_MODE}" \
  FILTER_FLOW="${FILTER_FLOW}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/run_server_xuancheng_one_day.sh" "${DATE}" >"${LOG_PATH}" 2>&1
  echo "[done] ${DATE}: ${NPZ_PATH}"
done

echo "[done] month run finished"

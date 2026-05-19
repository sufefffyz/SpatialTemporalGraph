#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/lane_agg_30d_official_raw_direct_npz}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/lane_agg_30d_official_raw_direct_$(date +%Y%m%d_%H%M%S)}"
DURATION="${DURATION:-86400}"
BUCKET_SECONDS="${BUCKET_SECONDS:-60}"
THREAD_NUM="${THREAD_NUM:-1}"
TL_MODE="${TL_MODE:-official_rl}"
WRITE_CSV="${WRITE_CSV:-0}"
DOWNLOAD_FIRST="${DOWNLOAD_FIRST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
WORKER_COUNT="${WORKER_COUNT:-1}"
WORKER_INDEX="${WORKER_INDEX:-0}"
DEFAULT_CITYFLOW_PYTHON="/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_CITYFLOW_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_CITYFLOW_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if (( WORKER_COUNT < 1 )); then
  echo "[error] WORKER_COUNT must be >= 1, got ${WORKER_COUNT}" >&2
  exit 2
fi
if (( WORKER_INDEX < 0 || WORKER_INDEX >= WORKER_COUNT )); then
  echo "[error] WORKER_INDEX must be in [0, WORKER_COUNT), got ${WORKER_INDEX}/${WORKER_COUNT}" >&2
  exit 2
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

echo "[info] official raw direct-engine lane month run"
echo "[info] data_root=${DATA_ROOT}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] log_dir=${LOG_DIR}"
echo "[info] duration=${DURATION} bucket=${BUCKET_SECONDS} tl_mode=${TL_MODE}"
echo "[info] thread_num=${THREAD_NUM} write_csv=${WRITE_CSV}"
echo "[info] worker_index=${WORKER_INDEX} worker_count=${WORKER_COUNT}"
echo "[info] python=${PYTHON_BIN}"

if [[ "${DOWNLOAD_FIRST}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/download_xuancheng_figshare.py" \
    --data-root "${DATA_ROOT}" \
    --all-days \
    --retries "${DOWNLOAD_RETRIES}"
fi

DATE_ORD=0
for DATE in "${DATES[@]}"; do
  CURRENT_ORD="${DATE_ORD}"
  DATE_ORD=$((DATE_ORD + 1))
  if (( CURRENT_ORD % WORKER_COUNT != WORKER_INDEX )); then
    continue
  fi

  NPZ_PATH="${OUTPUT_DIR}/xuancheng_${DATE}_lane_agg_${BUCKET_SECONDS}s.npz"
  CSV_PATH="${OUTPUT_DIR}/xuancheng_${DATE}_lane_agg_${BUCKET_SECONDS}s.csv.gz"
  LOG_PATH="${LOG_DIR}/${DATE}.log"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${NPZ_PATH}" ]]; then
    if [[ "${WRITE_CSV}" != "1" || -s "${CSV_PATH}" ]]; then
      echo "[skip] ${DATE}: existing ${NPZ_PATH}"
      continue
    fi
  fi

  echo "[run] ${DATE} -> ${LOG_PATH}"
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  DURATION="${DURATION}" \
  BUCKET_SECONDS="${BUCKET_SECONDS}" \
  THREAD_NUM="${THREAD_NUM}" \
  TL_MODE="${TL_MODE}" \
  WRITE_CSV="${WRITE_CSV}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/run_server_xuancheng_official_raw_lane_one_day.sh" "${DATE}" >"${LOG_PATH}" 2>&1
  echo "[done] ${DATE}: ${NPZ_PATH}"
done

echo "[done] lane month run finished"

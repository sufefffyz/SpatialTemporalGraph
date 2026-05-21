#!/usr/bin/env bash
set -euo pipefail

DATE="${1:-2023-04-03}"
DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/dtignn_turn_1h_10s_paper_mp}"
DURATION="${DURATION:-3600}"
START_SECOND="${START_SECOND:-0}"
BUCKET_SECONDS="${BUCKET_SECONDS:-10}"
THREAD_NUM="${THREAD_NUM:-1}"
TL_MODE="${TL_MODE:-official_rl}"
DOWNLOAD_FIRST="${DOWNLOAD_FIRST:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
MISSING_RATIOS="${MISSING_RATIOS:-0.1,0.3,0.5,0.7,0.9}"
MASK_SEED="${MASK_SEED:-2026}"
DEFAULT_CITYFLOW_PYTHON="/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_CITYFLOW_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_CITYFLOW_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${TL_MODE}" != "official_rl" ]]; then
  echo "[error] DTIGNN-style generation requires TL_MODE=official_rl, got ${TL_MODE}" >&2
  exit 2
fi

echo "[info] Xuancheng DTIGNN-style one-day generation"
echo "[info] date=${DATE} data_root=${DATA_ROOT}"
echo "[info] start=${START_SECOND}s duration=${DURATION}s bucket=${BUCKET_SECONDS}s"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] python=${PYTHON_BIN}"

if [[ "${DOWNLOAD_FIRST}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/download_xuancheng_figshare.py" \
    --data-root "${DATA_ROOT}" \
    --days "${DATE}" \
    --retries "${DOWNLOAD_RETRIES}"
fi

if ! "${PYTHON_BIN}" -c "import cityflow" >/dev/null 2>&1; then
  echo "[error] cityflow is not importable in ${PYTHON_BIN}" >&2
  exit 2
fi

CONFIG_PATH="${DATA_ROOT}/configs/config_xuancheng_${DATE}_dtignn.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_cityflow_config.py" \
  --data-root "${DATA_ROOT}" \
  --date "${DATE}" \
  --tl-mode "${TL_MODE}" \
  --output "${CONFIG_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_cityflow_dtignn_generation.py" \
  --config "${CONFIG_PATH}" \
  --date "${DATE}" \
  --output-dir "${OUTPUT_DIR}" \
  --duration "${DURATION}" \
  --start-second "${START_SECOND}" \
  --bucket-seconds "${BUCKET_SECONDS}" \
  --thread-num "${THREAD_NUM}" \
  --missing-ratios "${MISSING_RATIOS}" \
  --mask-seed "${MASK_SEED}"

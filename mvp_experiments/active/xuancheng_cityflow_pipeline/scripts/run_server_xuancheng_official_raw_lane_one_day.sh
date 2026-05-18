#!/usr/bin/env bash
set -euo pipefail

DATE="${1:-2023-04-03}"
DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
DURATION="${DURATION:-3600}"
BUCKET_SECONDS="${BUCKET_SECONDS:-60}"
THREAD_NUM="${THREAD_NUM:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/lane_agg_official_raw_direct}"
TL_MODE="${TL_MODE:-official_rl}"
WRITE_CSV="${WRITE_CSV:-0}"
DEFAULT_CITYFLOW_PYTHON="/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_CITYFLOW_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_CITYFLOW_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[info] official raw direct-engine lane one-day run"
echo "[info] date=${DATE} data_root=${DATA_ROOT} duration=${DURATION} bucket=${BUCKET_SECONDS}"
echo "[info] tl_mode=${TL_MODE} write_csv=${WRITE_CSV}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_xuancheng_figshare.py" \
  --data-root "${DATA_ROOT}" \
  --days "${DATE}"

if ! "${PYTHON_BIN}" -c "import cityflow" >/dev/null 2>&1; then
  echo "[error] cityflow is not importable in ${PYTHON_BIN}" >&2
  exit 2
fi

CONFIG_PATH="${DATA_ROOT}/configs/config_xuancheng_${DATE}_laneagg.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_cityflow_config.py" \
  --data-root "${DATA_ROOT}" \
  --date "${DATE}" \
  --tl-mode "${TL_MODE}" \
  --output "${CONFIG_PATH}"

CSV_ARGS=()
if [[ "${WRITE_CSV}" == "1" ]]; then
  CSV_ARGS=(--write-csv)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_cityflow_lane_aggregation.py" \
  --config "${CONFIG_PATH}" \
  --date "${DATE}" \
  --duration "${DURATION}" \
  --bucket-seconds "${BUCKET_SECONDS}" \
  --thread-num "${THREAD_NUM}" \
  --output-dir "${OUTPUT_DIR}" \
  "${CSV_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

DATE="${1:-2023-04-03}"
DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
DURATION="${DURATION:-86400}"
BUCKET_SECONDS="${BUCKET_SECONDS:-60}"
THREAD_NUM="${THREAD_NUM:-1}"
TL_MODE="${TL_MODE:-fixed_time}"
DEFAULT_CITYFLOW_PYTHON="/home/yuzhang_fei/miniconda3/envs/xuancheng_cityflow/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_CITYFLOW_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_CITYFLOW_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
DOCKER_IMAGE="${DOCKER_IMAGE:-kingsleycl/cityflow_env:latest}"
FILTER_FLOW="${FILTER_FLOW:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"

echo "[info] repo=${REPO_ROOT}"
echo "[info] date=${DATE} data_root=${DATA_ROOT} duration=${DURATION} bucket=${BUCKET_SECONDS}"
echo "[info] python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_xuancheng_figshare.py" \
  --data-root "${DATA_ROOT}" \
  --days "${DATE}"

if "${PYTHON_BIN}" -c "import cityflow" >/dev/null 2>&1; then
  echo "[info] using native Python cityflow"
  CONFIG_PATH="${DATA_ROOT}/configs/config_xuancheng_${DATE}_roadagg.json"
  FLOW_ARGS=()
  if [[ "${FILTER_FLOW}" == "1" ]]; then
    FILTERED_FLOW="${DATA_ROOT}/raw/data_${DATE//-/_}_type_filtered.valid.json"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/filter_cityflow_flow.py" \
      --data-root "${DATA_ROOT}" \
      --date "${DATE}" \
      --output "${FILTERED_FLOW}"
    FLOW_ARGS=(--flow-file "${FILTERED_FLOW}")
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/make_cityflow_config.py" \
    --data-root "${DATA_ROOT}" \
    --date "${DATE}" \
    --tl-mode "${TL_MODE}" \
    "${FLOW_ARGS[@]}" \
    --output "${CONFIG_PATH}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_cityflow_road_aggregation.py" \
    --config "${CONFIG_PATH}" \
    --date "${DATE}" \
    --duration "${DURATION}" \
    --bucket-seconds "${BUCKET_SECONDS}" \
    --thread-num "${THREAD_NUM}" \
    --output-dir "${DATA_ROOT}/road_agg"
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "[error] cityflow is not importable and docker is unavailable." >&2
    exit 2
  fi
  echo "[info] native cityflow unavailable; using Docker image ${DOCKER_IMAGE}"
  docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -v "${DATA_ROOT}:/data" \
    "${DOCKER_IMAGE}" \
    bash -lc "cd /workspace && \
      python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/make_cityflow_config.py \
        --data-root /data \
        --date '${DATE}' \
        --tl-mode '${TL_MODE}' \
        --output /data/configs/config_xuancheng_${DATE}_roadagg.json && \
      python mvp_experiments/active/xuancheng_cityflow_pipeline/scripts/run_cityflow_road_aggregation.py \
        --config /data/configs/config_xuancheng_${DATE}_roadagg.json \
        --date '${DATE}' \
        --duration '${DURATION}' \
        --bucket-seconds '${BUCKET_SECONDS}' \
        --thread-num '${THREAD_NUM}' \
        --output-dir /data/road_agg"
fi

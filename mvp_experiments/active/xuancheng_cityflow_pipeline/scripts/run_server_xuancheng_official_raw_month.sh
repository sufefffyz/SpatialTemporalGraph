#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/road_agg_30d_official_raw_direct}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/road_agg_30d_official_raw_direct_$(date +%Y%m%d_%H%M%S)}"
TL_MODE="${TL_MODE:-official_rl}"
FILTER_FLOW="${FILTER_FLOW:-0}"
DOWNLOAD_FIRST="${DOWNLOAD_FIRST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ "${FILTER_FLOW}" != "0" ]]; then
  echo "[error] official raw run requires FILTER_FLOW=0, got ${FILTER_FLOW}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[info] official raw direct-engine month run"
echo "[info] tl_mode=${TL_MODE} filter_flow=${FILTER_FLOW}"
echo "[info] output_dir=${OUTPUT_DIR}"
echo "[info] log_dir=${LOG_DIR}"

DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LOG_DIR="${LOG_DIR}" \
TL_MODE="${TL_MODE}" \
FILTER_FLOW="${FILTER_FLOW}" \
DOWNLOAD_FIRST="${DOWNLOAD_FIRST}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
  bash "${SCRIPT_DIR}/run_server_xuancheng_month.sh"

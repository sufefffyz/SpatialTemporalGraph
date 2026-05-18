#!/usr/bin/env bash
set -euo pipefail

DATE="${1:-2023-04-03}"
DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/road_agg_official_raw_direct}"
TL_MODE="${TL_MODE:-official_rl}"
FILTER_FLOW="${FILTER_FLOW:-0}"

if [[ "${FILTER_FLOW}" != "0" ]]; then
  echo "[error] official raw run requires FILTER_FLOW=0, got ${FILTER_FLOW}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[info] official raw direct-engine one-day run"
echo "[info] date=${DATE} tl_mode=${TL_MODE} filter_flow=${FILTER_FLOW}"
echo "[info] output_dir=${OUTPUT_DIR}"

DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TL_MODE="${TL_MODE}" \
FILTER_FLOW="${FILTER_FLOW}" \
  bash "${SCRIPT_DIR}/run_server_xuancheng_one_day.sh" "${DATE}"

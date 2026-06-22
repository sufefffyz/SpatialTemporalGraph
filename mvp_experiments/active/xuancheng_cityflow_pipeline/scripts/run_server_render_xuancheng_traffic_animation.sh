#!/usr/bin/env bash
set -euo pipefail

DATE="${DATE:-2023-04-03}"
DATA_ROOT="${DATA_ROOT:-/data/yuzhang_fei/xuancheng_cityflow}"
MODE="${MODE:-roadagg}" # roadagg or paper_mp
AGGREGATE_FACTOR="${AGGREGATE_FACTOR:-5}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-}"
FOCUS_ROADS="${FOCUS_ROADS:-34180205108,34180205109_1,34180205109,34180200419}"
TITLE="${TITLE:-Xuancheng ${DATE} road-level traffic evolution}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python}"

ROADNET="${ROADNET:-${DATA_ROOT}/raw/roadnet_xuancheng250319.json}"
SUMO_NET="${SUMO_NET:-${DATA_ROOT}/raw/xuancheng.net.xml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/mvp_experiments/active/xuancheng_cityflow_pipeline/results/traffic_animation}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${MODE}" == "roadagg" ]]; then
  NPZ="${NPZ:-${DATA_ROOT}/road_agg_official_raw_direct/xuancheng_${DATE}_road_agg_60s.npz}"
  OUTPUT_HTML="${OUTPUT_HTML:-${OUTPUT_DIR}/xuancheng_${DATE}_roadagg_traffic_animation.html}"
elif [[ "${MODE}" == "paper_mp" ]]; then
  NPZ="${NPZ:-${DATA_ROOT}/road_agg_paper_mp/xuancheng_${DATE}_paper_mp_road_agg_60s_start0_dur86400.npz}"
  OUTPUT_HTML="${OUTPUT_HTML:-${OUTPUT_DIR}/xuancheng_${DATE}_paper_mp_traffic_animation.html}"
else
  echo "[error] MODE must be roadagg or paper_mp, got ${MODE}" >&2
  exit 2
fi

ARGS=(
  "${SCRIPT_DIR}/render_xuancheng_traffic_animation.py"
  --roadnet "${ROADNET}"
  --sumo-net "${SUMO_NET}"
  --npz "${NPZ}"
  --output-html "${OUTPUT_HTML}"
  --date "${DATE}"
  --title "${TITLE}"
  --aggregate-factor "${AGGREGATE_FACTOR}"
  --start-frame "${START_FRAME}"
  --focus-roads "${FOCUS_ROADS}"
)

if [[ -n "${END_FRAME}" ]]; then
  ARGS+=(--end-frame "${END_FRAME}")
fi

echo "[info] mode=${MODE}"
echo "[info] npz=${NPZ}"
echo "[info] output=${OUTPUT_HTML}"
"${PYTHON_BIN}" "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${CITY_TRAFFIC_DATA_DIR:-/data/yuzhang_fei/Urban_Traffic_Benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/full_delay_audit}"
MAX_TIME_STEPS="${MAX_TIME_STEPS:-10000}"
MAX_EDGES="${MAX_EDGES:-6000}"
MAX_LAG="${MAX_LAG:-12}"
RESIDUALIZE="${RESIDUALIZE:-time_of_day}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
COVERAGE_CHUNK_SIZE="${COVERAGE_CHUNK_SIZE:-2048}"

run_one() {
  local name="$1"
  local npz="$2"
  local min_pair_coverage="$3"
  local output_dir="${OUTPUT_ROOT}/${name}"

  if [[ ! -f "${npz}" ]]; then
    echo "Missing dataset: ${npz}" >&2
    return 1
  fi

  mkdir -p "${output_dir}"
  echo "[delay-audit] ${name}"
  echo "  dataset: ${npz}"
  echo "  output : ${output_dir}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_quick_delay_audit.py" \
    --dataset-npz "${npz}" \
    --output-dir "${output_dir}" \
    --split train \
    --max-time-steps "${MAX_TIME_STEPS}" \
    --max-edges "${MAX_EDGES}" \
    --max-lag "${MAX_LAG}" \
    --min-pair-coverage "${min_pair_coverage}" \
    --coverage-chunk-size "${COVERAGE_CHUNK_SIZE}" \
    --score-chunk-size "${SCORE_CHUNK_SIZE}" \
    --residualize "${RESIDUALIZE}" \
    --min-corr "${MIN_CORR:-0.20}" \
    --min-improvement "${MIN_IMPROVEMENT:-0.03}" \
    --distance-bins-m "${DISTANCE_BINS_M:-0,25,50,100,200,500,1000,2000,5000,inf}" \
    --seed "${SEED:-42}"
}

run_one "city_traffic_l_speed_full_train" "${DATA_DIR}/city_traffic_l_speed.npz" "${CITY_L_MIN_PAIR_COVERAGE:-0.80}"
run_one "city_traffic_m_speed_full_train" "${DATA_DIR}/city_traffic_m_speed.npz" "${CITY_M_MIN_PAIR_COVERAGE:-0.50}"

echo "Full delay audit outputs written to: ${OUTPUT_ROOT}"

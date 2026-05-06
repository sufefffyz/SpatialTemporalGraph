#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-delay_selective/outputs/dataset_profile}"
TARGET_MAX_LOAD_GB="${TARGET_MAX_LOAD_GB:-64}"

DATA_ROOTS=()
if [[ -n "${CITY_TRAFFIC_DATA_DIR:-}" ]]; then
  DATA_ROOTS+=("${CITY_TRAFFIC_DATA_DIR}")
fi
DATA_ROOTS+=("data")
DATA_ROOTS+=("/data/yuzhang_fei/Urban_Traffic_Benchmark")

SPECS=()
TARGET_SPECS=()

add_spec() {
  local name="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    return 0
  fi
  local spec="${name}:${path}"
  for existing in "${SPECS[@]}"; do
    if [[ "${existing%%:*}" == "${name}" ]]; then
      return 0
    fi
    if [[ "${existing}" == "${spec}" ]]; then
      return 0
    fi
  done
  SPECS+=("${spec}")
}

add_target_spec() {
  local name="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    return 0
  fi
  local spec="${name}:${path}"
  for existing in "${TARGET_SPECS[@]}"; do
    if [[ "${existing}" == "${spec}" ]]; then
      return 0
    fi
  done
  TARGET_SPECS+=("${spec}")
}

for root in "${DATA_ROOTS[@]}"; do
  add_spec "city_traffic_m_speed_full" "${root}/city_traffic_m_speed.npz"
  add_spec "city_traffic_m_volume_full" "${root}/city_traffic_m_volume.npz"
done

for root in "${DATA_ROOTS[@]}"; do
  add_spec "city_traffic_m_speed_category_1_0" "${root}/city_traffic_m_speed__category__1_0.npz"
  add_spec "city_traffic_m_volume_category_1_0" "${root}/city_traffic_m_volume__category__1_0.npz"
  add_spec "city_traffic_m_speed_category_1_0" "${root}/subgraphs_city_m/speed/city_traffic_m_speed__category__1_0.npz"
  add_spec "city_traffic_m_volume_category_1_0" "${root}/subgraphs_city_m/volume/city_traffic_m_volume__category__1_0.npz"
done

add_target_spec \
  "city_traffic_m_speed_category_1_0" \
  "product/traffic_city_traffic_m_speed__category__1_0/targets.npy"
add_target_spec \
  "city_traffic_m_volume_category_1_0" \
  "product/traffic_city_traffic_m_volume__category__1_0/targets.npy"

if [[ "${#SPECS[@]}" -eq 0 ]]; then
  echo "No city-traffic-M speed/volume NPZ files found."
  echo "Set CITY_TRAFFIC_DATA_DIR=/path/to/Urban_Traffic_Benchmark and rerun."
  exit 1
fi

ARGS=()
for spec in "${SPECS[@]}"; do
  ARGS+=(--dataset "${spec}")
done
for spec in "${TARGET_SPECS[@]}"; do
  ARGS+=(--target-array "${spec}")
done

printf 'Profiling datasets:\n'
printf '  %s\n' "${SPECS[@]}"

"${PYTHON_BIN}" delay_selective/profile_city_traffic_datasets.py \
  --output-dir "${OUTPUT_DIR}" \
  --sample-time-steps "${SAMPLE_TIME_STEPS:-8000}" \
  --sample-nodes "${SAMPLE_NODES:-1500}" \
  --hist-bins "${HIST_BINS:-100}" \
  --target-max-load-gb "${TARGET_MAX_LOAD_GB}" \
  --seed "${SEED:-42}" \
  "${ARGS[@]}"

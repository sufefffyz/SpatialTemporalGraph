#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_largest_gla_gba_coreset_queue.sh <gpu> <plan|launch>" >&2
  exit 1
fi

GPU="$1"
ACTION="$2"
RATIO="${RATIO:-0.1}"
SEED="${SEED:-2023}"
TEMPORAL_PERIOD="${TEMPORAL_PERIOD:-96}"
RUN_TS="${RUN_TS:-$(date '+%Y%m%d_%H%M%S')}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-logs/st_coreset_largest_gla_gba/${RUN_TS}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASICTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASICTS_ROOT}"

DATASETS=(GBA GLA)
MODELS=(stid gwnet)
STRATEGIES=(random temporal kcenter temporal_kcenter temporal_kcenter_difficulty)

config_for() {
  local dataset="$1"
  local model="$2"
  local mode="$3"
  case "${model}:${mode}" in
    stid:full) echo "baselines/STID/${dataset}.py" ;;
    stid:coreset) echo "baselines/STID/${dataset}_coreset.py" ;;
    gwnet:full) echo "baselines/GWNet/${dataset}_largest_aligned.py" ;;
    gwnet:coreset) echo "baselines/GWNet/${dataset}_largest_aligned_coreset.py" ;;
    *)
      echo "Unsupported combination: ${model}:${mode}" >&2
      exit 2
      ;;
  esac
}

verify_dataset_ready() {
  local dataset="$1"
  local mode="${2:-strict}"
  local dataset_dir="datasets/${dataset}"
  local raw_dir="datasets/raw_data/${dataset}"

  if [[ -f "${dataset_dir}/desc.json" && -f "${dataset_dir}/data.dat" && -f "${dataset_dir}/adj_mx.pkl" && -f "${dataset_dir}/meta.csv" ]]; then
    return 0
  fi

  if [[ -f "${raw_dir}/${dataset}.h5" && -f "${raw_dir}/adj_${dataset}.npy" && -f "${raw_dir}/meta_${dataset}.csv" ]]; then
    echo "[WARN] Dataset ${dataset} is not materialized yet." >&2
    echo "[WARN] Run: ${PYTHON_BIN} scripts/data_preparation/${dataset}/generate_training_data.py" >&2
    [[ "${mode}" == "warn" ]] && return 0
    return 1
  fi

  echo "[WARN] Missing both materialized dataset and raw files for ${dataset}." >&2
  echo "[WARN] Expected raw files under ${raw_dir}/" >&2
  [[ "${mode}" == "warn" ]] && return 0
  return 1
}

build_train_command() {
  local dataset="$1"
  local model="$2"
  local mode="$3"
  local strategy="${4:-}"
  local config
  local tag
  local log_dir
  local log_name
  config="$(config_for "${dataset}" "${model}" "${mode}")"
  log_dir="${LOG_ROOT}/${dataset}/${model}"
  mkdir -p "${log_dir}"

  if [[ "${mode}" == "full" ]]; then
    tag="stcore_${RUN_TS}_${dataset}_${model}_full"
    log_name="full.log"
    cat <<EOF
env CUDA_VISIBLE_DEVICES="${GPU}" WANDB_RUN_GROUP="st_coreset_${dataset}_${model}_${RUN_TS}" BASICTS_RUN_TAG="${tag}" BASICTS_SEED="${SEED}" ${PYTHON_BIN} experiments/train.py -c "${config}" -g "${GPU}" | tee "${log_dir}/${log_name}"
EOF
  else
    tag="stcore_${RUN_TS}_${dataset}_${model}_${strategy}_r${RATIO}"
    log_name="${strategy}_r${RATIO}.log"
    cat <<EOF
env CUDA_VISIBLE_DEVICES="${GPU}" WANDB_RUN_GROUP="st_coreset_${dataset}_${model}_${RUN_TS}" BASICTS_RUN_TAG="${tag}" BASICTS_SEED="${SEED}" CORESET_STRATEGY="${strategy}" CORESET_RATIO="${RATIO}" CORESET_SEED="${SEED}" CORESET_TEMPORAL_PERIOD="${TEMPORAL_PERIOD}" ${PYTHON_BIN} experiments/train.py -c "${config}" -g "${GPU}" | tee "${log_dir}/${log_name}"
EOF
  fi
}

print_manifest() {
  local dataset model strategy
  for dataset in "${DATASETS[@]}"; do
    verify_dataset_ready "${dataset}" warn
    for model in "${MODELS[@]}"; do
      for strategy in "${STRATEGIES[@]}"; do
        printf '[PLAN] dataset=%s model=%s strategy=%s\n' "${dataset}" "${model}" "${strategy}"
        build_train_command "${dataset}" "${model}" "coreset" "${strategy}"
      done
      printf '[PLAN] dataset=%s model=%s strategy=%s\n' "${dataset}" "${model}" "full"
      build_train_command "${dataset}" "${model}" "full"
    done
  done
}

run_queue() {
  local dataset model strategy

  if [[ "${ALLOW_STCORE_GWNET_CLEAR:-0}" != "1" ]]; then
    echo "Refusing to launch: set ALLOW_STCORE_GWNET_CLEAR=1 after confirming the current stcore_gwnet job is finished." >&2
    exit 3
  fi
  if [[ "${ALLOW_REMOTE_SYNC_READY:-0}" != "1" ]]; then
    echo "Refusing to launch: set ALLOW_REMOTE_SYNC_READY=1 after syncing the new local configs/scripts to the remote repo." >&2
    exit 4
  fi

  for dataset in "${DATASETS[@]}"; do
    verify_dataset_ready "${dataset}" strict
    for model in "${MODELS[@]}"; do
      for strategy in "${STRATEGIES[@]}"; do
        eval "$(build_train_command "${dataset}" "${model}" "coreset" "${strategy}")"
      done
      eval "$(build_train_command "${dataset}" "${model}" "full")"
    done
  done
}

case "${ACTION}" in
  plan)
    print_manifest
    ;;
  launch)
    run_queue
    ;;
  *)
    echo "Unknown action: ${ACTION}. Use plan or launch." >&2
    exit 2
    ;;
esac

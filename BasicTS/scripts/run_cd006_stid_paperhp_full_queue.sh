#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
ROOT="${ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
RUN_TAG="${RUN_TAG:-cd006_stid_paperhp_full_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-SD PEMS08}"
SEEDS="${SEEDS:-2023 2024 2025 2026 2027}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
WAIT_FOR_SCREENS="${WAIT_FOR_SCREENS:-}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
LOG_DIR="${ROOT}/logs/${RUN_TAG}"
QUEUE_LOG="${LOG_DIR}/queue_stid_paperhp_full.log"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if (( MAX_PARALLEL_JOBS < 1 )); then
    echo "MAX_PARALLEL_JOBS must be >= 1, got ${MAX_PARALLEL_JOBS}." >&2
    exit 2
fi

if [[ -n "${WAIT_FOR_SCREENS}" ]]; then
    while screen -ls | grep -Eq "${WAIT_FOR_SCREENS}"; do
        echo "[$(date '+%F %T')] WAIT screens matching ${WAIT_FOR_SCREENS}" | tee -a "${QUEUE_LOG}"
        sleep "${WAIT_INTERVAL_SECONDS}"
    done
fi

running_jobs() {
    jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
    while (( "$(running_jobs)" >= MAX_PARALLEL_JOBS )); do
        wait -n
    done
}

wait_for_all() {
    while (( "$(running_jobs)" > 0 )); do
        wait -n
    done
}

run_job() {
    local dataset="$1"
    local seed="$2"
    local cfg="baselines/DataPruning/STID_${dataset}_paperhp_full.py"
    local log_file="${LOG_DIR}/stid_${dataset,,}_paperhp_full_s${seed}.log"
    if [[ -f "${log_file}" ]] && grep -q "DONE STID_${dataset}_paperhp_full" "${log_file}"; then
        echo "[$(date '+%F %T')] SKIP completed STID_${dataset}_paperhp_full seed=${seed}" | tee -a "${QUEUE_LOG}"
        return
    fi
    echo "[$(date '+%F %T')] START STID_${dataset}_paperhp_full seed=${seed} gpu=${GPU_ID}" | tee -a "${QUEUE_LOG}"
    DYNAMIC_PRUNING_SEED="${seed}" \
    BASICTS_SEED="${seed}" \
    DYNAMIC_PRUNING_CKPT_SUFFIX="${DYNAMIC_PRUNING_CKPT_SUFFIX:-paperhp_full_control_v1}" \
    DYNAMIC_PRUNING_DISABLE_EARLY_STOPPING="${DYNAMIC_PRUNING_DISABLE_EARLY_STOPPING:-1}" \
    python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    echo "[$(date '+%F %T')] DONE STID_${dataset}_paperhp_full seed=${seed} log=${log_file}" | tee -a "${log_file}" "${QUEUE_LOG}"
}

for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
        wait_for_slot
        run_job "${dataset}" "${seed}" &
    done
done
wait_for_all
echo "[$(date '+%F %T')] STID paperhp full queue finished." | tee -a "${QUEUE_LOG}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
GPU_ID="${GPU_ID:-0}"
STRATEGIES="${STRATEGIES:-soft_random infobatch epsilon_greedy}"
RATIOS="${RATIOS:-0.1 0.3}"
SEEDS="${SEEDS:-2023 2024 2025 2026 2027}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-3}"
LOG_DIR="${ROOT}/logs/${RUN_TAG}"
QUEUE_LOG="${LOG_DIR}/queue_gwnet_gpu0_pems08.log"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if (( MAX_PARALLEL_JOBS < 1 )); then
    echo "MAX_PARALLEL_JOBS must be >= 1, got ${MAX_PARALLEL_JOBS}." >&2
    exit 2
fi

ratio_tag() {
    echo "${1/./p}"
}

log_file_for() {
    local strategy="$1"
    local ratio="$2"
    local seed="$3"
    echo "${LOG_DIR}/gwnet_pems08_${strategy}_r$(ratio_tag "${ratio}")_s${seed}.log"
}

reserve_job() {
    local strategy="$1"
    local ratio="$2"
    local seed="$3"
    local log_file
    log_file="$(log_file_for "${strategy}" "${ratio}" "${seed}")"
    if [[ -f "${log_file}" ]] && grep -q "Result <test>" "${log_file}" && grep -q "DONE" "${log_file}"; then
        return
    fi
    {
        echo "[$(date '+%F %T')] DONE RESERVED_FOR_GPU0 GWNet_PEMS08_${strategy} ratio=${ratio} seed=${seed}"
        echo "This placeholder prevents the original GPU1 queue from duplicating a GPU0 job."
    } > "${log_file}"
}

run_job() {
    local strategy="$1"
    local ratio="$2"
    local seed="$3"
    local cfg="baselines/DataPruning/GWNet_PEMS08_${strategy}.py"
    local log_file tmp_log
    log_file="$(log_file_for "${strategy}" "${ratio}" "${seed}")"
    tmp_log="${log_file}.gpu0.running"

    if [[ -f "${log_file}" ]] && grep -q "Result <test>" "${log_file}" && grep -q "DONE" "${log_file}"; then
        echo "[$(date '+%F %T')] SKIP completed GWNet_PEMS08_${strategy} ratio=${ratio} seed=${seed}" | tee -a "${QUEUE_LOG}"
        return
    fi
    if [[ ! -f "${cfg}" ]]; then
        echo "[$(date '+%F %T')] SKIP missing config ${cfg}" | tee -a "${QUEUE_LOG}"
        return
    fi

    echo "[$(date '+%F %T')] START GWNet_PEMS08_${strategy} ratio=${ratio} seed=${seed} gpu=${GPU_ID}" | tee -a "${QUEUE_LOG}"
    if [[ "${strategy}" == "epsilon_greedy" ]]; then
        DYNAMIC_PRUNING_RATIO="${ratio}" \
        DYNAMIC_PRUNING_SEED="${seed}" \
        BASICTS_SEED="${seed}" \
        DYNAMIC_PRUNING_TARGET_FORWARD_RATIO="" \
        DYNAMIC_PRUNING_MATCH_EPOCHS=0 \
        DYNAMIC_PRUNING_EPSILON="${DYNAMIC_PRUNING_EPSILON:-0.1}" \
        DYNAMIC_PRUNING_PRUNING_PERIOD="${DYNAMIC_PRUNING_PRUNING_PERIOD:-10}" \
        DYNAMIC_PRUNING_SCORE_ALPHA="${DYNAMIC_PRUNING_SCORE_ALPHA:-0.8}" \
        DYNAMIC_PRUNING_RESCALE="${DYNAMIC_PRUNING_RESCALE:-0}" \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${tmp_log}"
    elif [[ "${strategy}" == "infobatch" ]]; then
        DYNAMIC_PRUNING_RATIO="${ratio}" \
        DYNAMIC_PRUNING_SEED="${seed}" \
        BASICTS_SEED="${seed}" \
        DYNAMIC_PRUNING_PRUNE_PROBABILITY="${DYNAMIC_PRUNING_PRUNE_PROBABILITY:-0.5}" \
        DYNAMIC_PRUNING_TARGET_FORWARD_RATIO="${DYNAMIC_PRUNING_TARGET_FORWARD_RATIO:-${ratio}}" \
        DYNAMIC_PRUNING_MATCH_EPOCHS="${DYNAMIC_PRUNING_MATCH_EPOCHS:-1}" \
        DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO="${DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO:-0.78}" \
        DYNAMIC_PRUNING_RESCALE="${DYNAMIC_PRUNING_RESCALE:-1}" \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${tmp_log}"
    else
        DYNAMIC_PRUNING_RATIO="${ratio}" \
        DYNAMIC_PRUNING_SEED="${seed}" \
        BASICTS_SEED="${seed}" \
        DYNAMIC_PRUNING_TARGET_FORWARD_RATIO="" \
        DYNAMIC_PRUNING_MATCH_EPOCHS=0 \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${tmp_log}"
    fi
    echo "[$(date '+%F %T')] DONE GWNet_PEMS08_${strategy} ratio=${ratio} seed=${seed} log=${log_file}" | tee -a "${tmp_log}" "${QUEUE_LOG}"
    mv "${tmp_log}" "${log_file}"
}

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

for strategy in ${STRATEGIES}; do
    for ratio in ${RATIOS}; do
        for seed in ${SEEDS}; do
            reserve_job "${strategy}" "${ratio}" "${seed}"
        done
    done
done

for strategy in ${STRATEGIES}; do
    for ratio in ${RATIOS}; do
        for seed in ${SEEDS}; do
            wait_for_slot
            run_job "${strategy}" "${ratio}" "${seed}" &
        done
    done
done
wait_for_all
echo "[$(date '+%F %T')] GPU0 PEMS08 queue finished." | tee -a "${QUEUE_LOG}"

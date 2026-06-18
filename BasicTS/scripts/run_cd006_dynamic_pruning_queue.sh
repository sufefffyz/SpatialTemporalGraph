#!/usr/bin/env bash
set -euo pipefail

QUEUE="${1:?Usage: $0 <stid|gwnet> <gpu_id>}"
GPU_ID="${2:?Usage: $0 <stid|gwnet> <gpu_id>}"
ROOT="${ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
RUN_TAG="${RUN_TAG:-cd006_dynamic_pruning_$(date +%Y%m%d_%H%M%S)}"
STRATEGIES="${STRATEGIES:-soft_random infobatch epsilon_greedy}"
DATASETS="${DATASETS:-SD PEMS04 PEMS08}"
RATIOS="${RATIOS:-0.1 0.3}"
SEEDS="${SEEDS:-2023 2024 2025 2026 2027}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
LOG_DIR="${ROOT}/logs/${RUN_TAG}"
WAIT_FOR_SCREENS="${WAIT_FOR_SCREENS:-}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
SKIP_REGEX="${SKIP_REGEX:-}"
SKIP_EPSILON_GREEDY_EXISTING="${SKIP_EPSILON_GREEDY_EXISTING:-0}"

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
        echo "[$(date '+%F %T')] WAIT screens matching ${WAIT_FOR_SCREENS}" | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
        sleep "${WAIT_INTERVAL_SECONDS}"
    done
fi

upper_model() {
    case "$1" in
        stid) echo "STID" ;;
        gwnet) echo "GWNet" ;;
        *) echo "Unknown queue: $1. Expected stid or gwnet." >&2; exit 2 ;;
    esac
}

run_job() {
    local model="$1"
    local dataset="$2"
    local strategy="$3"
    local ratio="$4"
    local seed="$5"
    local cfg="baselines/DataPruning/${model}_${dataset}_${strategy}.py"
    local ratio_tag="${ratio/./p}"
    local job_id="${model,,}_${dataset,,}_${strategy}_s${seed}"
    local log_file="${LOG_DIR}/${model,,}_${dataset,,}_${strategy}_r${ratio_tag}_s${seed}.log"

    if [[ -n "${SKIP_REGEX}" ]] && [[ "${job_id}" =~ ${SKIP_REGEX} ]]; then
        echo "[$(date '+%F %T')] SKIP by regex ${job_id}" | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
        return
    fi
    if [[ "${SKIP_EPSILON_GREEDY_EXISTING}" == "1" && "${strategy}" == "epsilon_greedy" && "${dataset}" != "PEMS04" ]]; then
        echo "[$(date '+%F %T')] SKIP existing epsilon_greedy ${job_id}" | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
        return
    fi
    if [[ ! -f "${cfg}" ]]; then
        echo "[$(date '+%F %T')] SKIP missing config ${cfg}" | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
        return
    fi
    if [[ -f "${log_file}" ]] && grep -q "DONE" "${log_file}"; then
        echo "[$(date '+%F %T')] SKIP completed ${model}_${dataset}_${strategy}_r${ratio_tag}_s${seed}" | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
        return
    fi

    echo "[$(date '+%F %T')] START ${model}_${dataset}_${strategy} ratio=${ratio} seed=${seed} gpu=${GPU_ID}" | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
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
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    elif [[ "${strategy}" == "infobatch" || "${strategy}" == "infobatch_norm" || "${strategy}" == "proxy_gap" || "${strategy}" == "proxy_gap_dlinear" || "${strategy}" == "proxy_gap_lowpass" ]]; then
        DYNAMIC_PRUNING_RATIO="${ratio}" \
        DYNAMIC_PRUNING_SEED="${seed}" \
        BASICTS_SEED="${seed}" \
        DYNAMIC_PRUNING_PRUNE_PROBABILITY="${DYNAMIC_PRUNING_PRUNE_PROBABILITY:-0.5}" \
        DYNAMIC_PRUNING_TARGET_FORWARD_RATIO="${DYNAMIC_PRUNING_TARGET_FORWARD_RATIO:-${ratio}}" \
        DYNAMIC_PRUNING_MATCH_EPOCHS="${DYNAMIC_PRUNING_MATCH_EPOCHS:-1}" \
        DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO="${DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO:-0.78}" \
        DYNAMIC_PRUNING_RESCALE="${DYNAMIC_PRUNING_RESCALE:-1}" \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    elif [[ "${strategy}" == "proxy_gap_lowpass_node_hard" || "${strategy}" == "lowpass_node_hard" || "${strategy}" == "node_hard_lowpass" ]]; then
        DYNAMIC_PRUNING_RATIO="${ratio}" \
        DYNAMIC_PRUNING_NODE_RATIO="${DYNAMIC_PRUNING_NODE_RATIO:-${ratio}}" \
        DYNAMIC_PRUNING_SEED="${seed}" \
        BASICTS_SEED="${seed}" \
        DYNAMIC_PRUNING_TARGET_FORWARD_RATIO="" \
        DYNAMIC_PRUNING_MATCH_EPOCHS=0 \
        DYNAMIC_PRUNING_NODE_REVISIT_PROBABILITY="${DYNAMIC_PRUNING_NODE_REVISIT_PROBABILITY:-0.5}" \
        DYNAMIC_PRUNING_RESCALE="${DYNAMIC_PRUNING_RESCALE:-1}" \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    else
        DYNAMIC_PRUNING_RATIO="${ratio}" \
        DYNAMIC_PRUNING_SEED="${seed}" \
        BASICTS_SEED="${seed}" \
        DYNAMIC_PRUNING_TARGET_FORWARD_RATIO="" \
        DYNAMIC_PRUNING_MATCH_EPOCHS=0 \
        DYNAMIC_PRUNING_RESCALE="${DYNAMIC_PRUNING_RESCALE:-0}" \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    fi
    echo "[$(date '+%F %T')] DONE ${model}_${dataset}_${strategy} ratio=${ratio} seed=${seed} log=${log_file}" | tee -a "${log_file}" "${LOG_DIR}/queue_${QUEUE}.log"
}

running_jobs() {
    jobs -pr | wc -l | tr -d ' '
}

FAILURES=0

wait_for_slot() {
    if (( MAX_PARALLEL_JOBS <= 1 )); then
        return
    fi
    while (( "$(running_jobs)" >= MAX_PARALLEL_JOBS )); do
        if ! wait -n; then
            FAILURES=$((FAILURES + 1))
        fi
    done
}

wait_for_all() {
    if (( MAX_PARALLEL_JOBS <= 1 )); then
        return
    fi
    while (( "$(running_jobs)" > 0 )); do
        if ! wait -n; then
            FAILURES=$((FAILURES + 1))
        fi
    done
    if (( FAILURES > 0 )); then
        echo "[$(date '+%F %T')] ${FAILURES} background job(s) failed." | tee -a "${LOG_DIR}/queue_${QUEUE}.log"
        exit 1
    fi
}

run_or_queue_job() {
    if (( MAX_PARALLEL_JOBS <= 1 )); then
        run_job "$@"
    else
        wait_for_slot
        run_job "$@" &
    fi
}

MODEL="$(upper_model "${QUEUE}")"
for strategy in ${STRATEGIES}; do
    for dataset in ${DATASETS}; do
        for ratio in ${RATIOS}; do
            for seed in ${SEEDS}; do
                run_or_queue_job "${MODEL}" "${dataset}" "${strategy}" "${ratio}" "${seed}"
            done
        done
    done
done
wait_for_all

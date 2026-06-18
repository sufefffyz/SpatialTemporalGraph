#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Usage: $0 <gpu_id> <config_stem> [config_stem ...]}"
shift
if (( "$#" < 1 )); then
    echo "Usage: $0 <gpu_id> <config_stem> [config_stem ...]" >&2
    exit 2
fi

ROOT="${ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
SEED="${SEED:-2023}"
RUN_TAG="${RUN_TAG:-cd007_proxygap_ref_cluster_seed${SEED}}"
LOG_DIR="${ROOT}/logs/${RUN_TAG}"
GPU_MAX_MEMORY_USED_MIB="${GPU_MAX_MEMORY_USED_MIB:-3500}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-120}"
WAIT_FOR_SCREENS="${WAIT_FOR_SCREENS:-}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ -n "${WAIT_FOR_SCREENS}" ]]; then
    while screen -ls | grep -Eq "${WAIT_FOR_SCREENS}"; do
        echo "[$(date '+%F %T')] WAIT screens matching ${WAIT_FOR_SCREENS}" | tee -a "${LOG_DIR}/queue_gpu${GPU_ID}.log"
        sleep "${WAIT_INTERVAL_SECONDS}"
    done
fi

wait_for_gpu() {
    local gpu_id="$1"
    local mem_used
    local util
    while true; do
        IFS=',' read -r mem_used util < <(
            nvidia-smi -i "${gpu_id}" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits
        )
        mem_used="${mem_used//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        if (( mem_used <= GPU_MAX_MEMORY_USED_MIB && util <= GPU_MAX_UTIL )); then
            echo "[$(date '+%F %T')] GPU ${gpu_id} ready mem=${mem_used}MiB util=${util}%"
            return
        fi
        echo "[$(date '+%F %T')] WAIT GPU ${gpu_id} mem=${mem_used}MiB util=${util}% thresholds mem<=${GPU_MAX_MEMORY_USED_MIB}MiB util<=${GPU_MAX_UTIL}%" \
            | tee -a "${LOG_DIR}/queue_gpu${gpu_id}.log"
        sleep "${WAIT_INTERVAL_SECONDS}"
    done
}

for config_stem in "$@"; do
    cfg="baselines/DataPruning/${config_stem}.py"
    log_file="${LOG_DIR}/${config_stem}_s${SEED}.log"
    if [[ ! -f "${cfg}" ]]; then
        echo "[$(date '+%F %T')] MISSING ${cfg}" | tee -a "${LOG_DIR}/queue_gpu${GPU_ID}.log"
        exit 2
    fi
    if [[ -f "${log_file}" ]] && grep -q "DONE ${config_stem}" "${log_file}"; then
        echo "[$(date '+%F %T')] SKIP completed ${config_stem}" | tee -a "${LOG_DIR}/queue_gpu${GPU_ID}.log"
        continue
    fi

    wait_for_gpu "${GPU_ID}"
    echo "[$(date '+%F %T')] START ${config_stem} seed=${SEED} gpu=${GPU_ID}" | tee -a "${LOG_DIR}/queue_gpu${GPU_ID}.log"
    env -u DYNAMIC_PRUNING_TARGET_FORWARD_RATIO \
        -u DYNAMIC_PRUNING_MATCH_EPOCHS \
        -u DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO \
        DYNAMIC_PRUNING_SEED="${SEED}" \
        BASICTS_SEED="${SEED}" \
        CLUSTER_SUBGRAPH_TARGET_EFFECTIVE_RATIO="${CLUSTER_SUBGRAPH_TARGET_EFFECTIVE_RATIO:-0.1}" \
        python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    echo "[$(date '+%F %T')] DONE ${config_stem} seed=${SEED} gpu=${GPU_ID} log=${log_file}" \
        | tee -a "${log_file}" "${LOG_DIR}/queue_gpu${GPU_ID}.log"
done

#!/usr/bin/env bash
set -euo pipefail

QUEUE="${1:?Usage: $0 <stid|gwnet> <gpu_id>}"
GPU_ID="${2:?Usage: $0 <stid|gwnet> <gpu_id>}"
ROOT="${ROOT:-/home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
RUN_TAG="${RUN_TAG:-cd006_epsilon_greedy_raju_strict_$(date +%Y%m%d_%H%M%S)}"
EPSILON="${DYNAMIC_PRUNING_EPSILON:-0.1}"
PRUNING_PERIOD="${DYNAMIC_PRUNING_PRUNING_PERIOD:-10}"
SCORE_ALPHA="${DYNAMIC_PRUNING_SCORE_ALPHA:-0.8}"
LOG_DIR="${ROOT}/logs/${RUN_TAG}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ROOT}"
mkdir -p "${LOG_DIR}"

run_job() {
    local cfg="$1"
    local ratio="$2"
    local name="$3"
    local ratio_tag="${ratio/./p}"
    local log_file="${LOG_DIR}/${name}_r${ratio_tag}_eps${EPSILON/./p}.log"

    echo "[$(date '+%F %T')] START ${name} ratio=${ratio} epsilon=${EPSILON} pruning_period=${PRUNING_PERIOD} score_alpha=${SCORE_ALPHA} gpu=${GPU_ID}"
    DYNAMIC_PRUNING_RATIO="${ratio}" \
    DYNAMIC_PRUNING_EPSILON="${EPSILON}" \
    DYNAMIC_PRUNING_PRUNING_PERIOD="${PRUNING_PERIOD}" \
    DYNAMIC_PRUNING_SCORE_ALPHA="${SCORE_ALPHA}" \
    DYNAMIC_PRUNING_RESCALE="${DYNAMIC_PRUNING_RESCALE:-0}" \
    python experiments/train.py -c "${cfg}" -g "${GPU_ID}" 2>&1 | tee "${log_file}"
    echo "[$(date '+%F %T')] DONE ${name} ratio=${ratio} log=${log_file}"
}

case "${QUEUE}" in
    stid)
        run_job "baselines/DataPruning/STID_SD_epsilon_greedy.py" "0.1" "stid_sd_epsilon_greedy"
        run_job "baselines/DataPruning/STID_SD_epsilon_greedy.py" "0.3" "stid_sd_epsilon_greedy"
        run_job "baselines/DataPruning/STID_PEMS08_epsilon_greedy.py" "0.1" "stid_pems08_epsilon_greedy"
        run_job "baselines/DataPruning/STID_PEMS08_epsilon_greedy.py" "0.3" "stid_pems08_epsilon_greedy"
        ;;
    gwnet)
        run_job "baselines/DataPruning/GWNet_SD_epsilon_greedy.py" "0.1" "gwnet_sd_epsilon_greedy"
        run_job "baselines/DataPruning/GWNet_SD_epsilon_greedy.py" "0.3" "gwnet_sd_epsilon_greedy"
        run_job "baselines/DataPruning/GWNet_PEMS08_epsilon_greedy.py" "0.1" "gwnet_pems08_epsilon_greedy"
        run_job "baselines/DataPruning/GWNet_PEMS08_epsilon_greedy.py" "0.3" "gwnet_pems08_epsilon_greedy"
        ;;
    *)
        echo "Unknown queue: ${QUEUE}. Expected stid or gwnet." >&2
        exit 2
        ;;
esac

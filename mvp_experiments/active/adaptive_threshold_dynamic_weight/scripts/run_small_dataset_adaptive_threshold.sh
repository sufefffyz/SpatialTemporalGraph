#!/usr/bin/env bash
set -euo pipefail

DATASET_KEY="${1:-}"
MODEL="${2:-}"
GRAPH="${3:-}"
GPU="${4:-0}"
RUN_TAG="${5:-smallbench_$(date '+%Y%m%d_%H%M%S')}"
PROJECT="${WANDB_PROJECT:-adaptive_threshold_smallbench}"

if [[ "${DATASET_KEY}" != "pems04" && "${DATASET_KEY}" != "metrla" ]]; then
  echo "Usage: $0 {pems04|metrla} {gwnet|dcrnn|stgcn} {original|osrmK64|gspK64|dynFull|dynOsrmK64|dynGspK64|originalAddapt|osrmK64Addapt|gspK64Addapt|dynFullAddapt|dynOsrmK64Addapt|dynGspK64Addapt} [gpu_id] [run_tag]" >&2
  exit 2
fi
if [[ "${MODEL}" != "gwnet" && "${MODEL}" != "dcrnn" && "${MODEL}" != "stgcn" ]]; then
  echo "MODEL must be one of: gwnet, dcrnn, stgcn" >&2
  exit 2
fi

case "${DATASET_KEY}" in
  pems04)
    BASE_DATASET="PEMS04"
    DIST_MTX="${PEMS04_DISTANCE_MTX:-datasets/PEMS04_osrm_shortest_distance_m.npy}"
    DATA_TAG="pems04"
    ;;
  metrla)
    BASE_DATASET="METR-LA"
    DIST_MTX="${METRLA_DISTANCE_MTX:-datasets/METR-LA_osrm_shortest_distance_m.npy}"
    DATA_TAG="metr-la"
    ;;
esac

ADDAPT="0"
CANDIDATE_DATASET=""
case "${GRAPH}" in
  original)
    DATASET_NAME="${BASE_DATASET}"
    GRAPH_TAG="original"
    GRAPH_KIND="fixed_original"
    DYNAMIC="0"
    ;;
  originalAddapt)
    DATASET_NAME="${BASE_DATASET}"
    GRAPH_TAG="original"
    GRAPH_KIND="fixed_original"
    DYNAMIC="0"
    ADDAPT="1"
    ;;
  osrmK64)
    DATASET_NAME="${BASE_DATASET}_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="fixed_osrm_topk"
    DYNAMIC="0"
    ;;
  osrmK64Addapt)
    DATASET_NAME="${BASE_DATASET}_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="fixed_osrm_topk"
    DYNAMIC="0"
    ADDAPT="1"
    ;;
  gspK64)
    DATASET_NAME="${BASE_DATASET}_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="fixed_gsp_topk"
    DYNAMIC="0"
    ;;
  gspK64Addapt)
    DATASET_NAME="${BASE_DATASET}_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="fixed_gsp_topk"
    DYNAMIC="0"
    ADDAPT="1"
    ;;
  dynFull)
    DATASET_NAME="${BASE_DATASET}"
    GRAPH_TAG="fullPair"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ;;
  dynFullAddapt)
    DATASET_NAME="${BASE_DATASET}"
    GRAPH_TAG="fullPair"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ADDAPT="1"
    ;;
  dynOsrmK64)
    DATASET_NAME="${BASE_DATASET}"
    CANDIDATE_DATASET="${BASE_DATASET}_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ;;
  dynOsrmK64Addapt)
    DATASET_NAME="${BASE_DATASET}"
    CANDIDATE_DATASET="${BASE_DATASET}_OSRMTOPK_K064"
    GRAPH_TAG="osrmK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ADDAPT="1"
    ;;
  dynGspK64)
    DATASET_NAME="${BASE_DATASET}"
    CANDIDATE_DATASET="${BASE_DATASET}_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ;;
  dynGspK64Addapt)
    DATASET_NAME="${BASE_DATASET}"
    CANDIDATE_DATASET="${BASE_DATASET}_GSPTOPK_K064"
    GRAPH_TAG="gspK64"
    GRAPH_KIND="dynamic_threshold"
    DYNAMIC="1"
    ADDAPT="1"
    ;;
  *)
    echo "Unsupported GRAPH=${GRAPH}" >&2
    exit 2
    ;;
esac

if [[ "${ADDAPT}" == "1" && "${MODEL}" != "gwnet" ]]; then
  echo "Addaptive adjacency variants are only valid for GWNet, got MODEL=${MODEL} GRAPH=${GRAPH}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASICTS_DIR="${REPO_ROOT}/BasicTS"
LOG_ROOT="${BASICTS_DIR}/logs/adaptive_threshold_dynamic_weight/smallbench/${RUN_TAG}/${DATASET_KEY}/${MODEL}"
mkdir -p "${LOG_ROOT}"

case "${MODEL}:${DYNAMIC}" in
  gwnet:0)
    CFG="baselines/GWNet/SD_osrm_gaussian_global.py"
    MODEL_NAME="GraphWaveNet"
    ;;
  gwnet:1)
    CFG="baselines/GWNet/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdGraphWaveNet"
    ;;
  dcrnn:0)
    CFG="baselines/DCRNN/SD_osrm_gaussian_global.py"
    MODEL_NAME="DCRNN"
    ;;
  dcrnn:1)
    CFG="baselines/DCRNN/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdDCRNN"
    ;;
  stgcn:0)
    CFG="baselines/STGCN/SD_osrm_gaussian_global.py"
    MODEL_NAME="STGCNChebGraphConv"
    ;;
  stgcn:1)
    CFG="baselines/STGCN/SD_dynamic_threshold.py"
    MODEL_NAME="DynamicThresholdSTGCN"
    ;;
  *)
    echo "Unsupported MODEL/DYNAMIC combination: ${MODEL}:${DYNAMIC}" >&2
    exit 2
    ;;
esac

log_file="${LOG_ROOT}/${MODEL}_${GRAPH}.log"
cd "${BASICTS_DIR}"

if [[ "${DYNAMIC}" == "1" && ! -f "${DIST_MTX}" ]]; then
  echo "Missing OSRM shortest-distance matrix: ${DIST_MTX}" >&2
  echo "Generate it with build_distance_matrices.py through OSRM before launching dynamic-threshold runs." >&2
  exit 1
fi
if [[ "${GRAPH}" == osrmK64* || "${GRAPH}" == gspK64* ]]; then
  if [[ ! -e "datasets/${DATASET_NAME}/adj_mx.pkl" ]]; then
    echo "Missing staged OSRM/GSP top-k dataset: datasets/${DATASET_NAME}/adj_mx.pkl" >&2
    exit 1
  fi
fi
if [[ -n "${CANDIDATE_DATASET}" && ! -e "datasets/${CANDIDATE_DATASET}/adj_mx.pkl" ]]; then
  echo "Missing dynamic-threshold candidate graph: datasets/${CANDIDATE_DATASET}/adj_mx.pkl" >&2
  exit 1
fi

export BASICTS_DATA_NAME="${DATASET_NAME}"
export BASICTS_GRAPH_TAG="${GRAPH_TAG}"
export BASICTS_RUN_TAG="${RUN_TAG}"
export BASICTS_SEED="${BASICTS_SEED:-2023}"
export BASICTS_NUM_EPOCHS="${BASICTS_NUM_EPOCHS:-100}"
export BASICTS_PATIENCE="${BASICTS_PATIENCE:-30}"
export BASICTS_BATCH_SIZE="${BASICTS_BATCH_SIZE:-64}"
export WANDB_PROJECT="${PROJECT}"
export WANDB_MODE="${WANDB_MODE:-online}"

ADDAPT_TAG="no-addaptadj"
if [[ "${ADDAPT}" == "1" ]]; then
  ADDAPT_TAG="addaptadj"
fi
CANDIDATE_TAG="full-pair"
if [[ -n "${CANDIDATE_DATASET}" ]]; then
  CANDIDATE_TAG="candidate-k64"
fi
export WANDB_RUN_GROUP="${DATA_TAG}_${MODEL}_${GRAPH_KIND}_${CANDIDATE_TAG}_${ADDAPT_TAG}"
export WANDB_TAGS="adaptive-threshold,${DATA_TAG},smallbench,${MODEL},${GRAPH_KIND},${GRAPH_TAG},${CANDIDATE_TAG},${ADDAPT_TAG}"
export WANDB_NAME="${MODEL_NAME}_${BASE_DATASET}_${GRAPH}_${GRAPH_KIND}_${RUN_TAG}"

if [[ "${MODEL}" == "gwnet" ]]; then
  export GWNET_ADDAPTADJ="${ADDAPT}"
fi

if [[ "${DYNAMIC}" == "1" ]]; then
  export DYNAMIC_GRAPH_MODE="hard"
  export DYNAMIC_GRAPH_WEIGHT_MODE="${DYNAMIC_GRAPH_WEIGHT_MODE:-binary}"
  export DYNAMIC_GRAPH_TARGET_AVG_DEGREE="${DYNAMIC_GRAPH_TARGET_AVG_DEGREE:-64}"
  export DYNAMIC_GRAPH_DIST_MTX="${DIST_MTX}"
  if [[ -n "${CANDIDATE_DATASET}" ]]; then
    export DYNAMIC_GRAPH_CANDIDATE_ADJ="datasets/${CANDIDATE_DATASET}/adj_mx.pkl"
  else
    unset DYNAMIC_GRAPH_CANDIDATE_ADJ
  fi
  export DYNAMIC_GWNET_ADDAPTADJ="${ADDAPT}"
fi

echo "[$(date '+%F %T')] Starting ${DATASET_KEY} ${MODEL} ${GRAPH} (${GRAPH_KIND}) on GPU ${GPU}" | tee -a "${log_file}"
/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python experiments/train.py -c "${CFG}" -g "${GPU}" 2>&1 | tee -a "${log_file}"
echo "[$(date '+%F %T')] Finished ${DATASET_KEY} ${MODEL} ${GRAPH} (${GRAPH_KIND})" | tee -a "${log_file}"

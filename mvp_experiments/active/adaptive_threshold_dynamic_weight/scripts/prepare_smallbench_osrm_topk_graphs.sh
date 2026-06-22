#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
META="${2:-}"
if [[ "${DATASET}" != "PEMS04" && "${DATASET}" != "METR-LA" && "${DATASET}" != "PEMS-BAY" ]]; then
  echo "Usage: $0 {PEMS04|METR-LA|PEMS-BAY} ordered_meta.csv" >&2
  exit 2
fi
if [[ -z "${META}" || ! -f "${META}" ]]; then
  echo "ordered_meta.csv is required and must exist." >&2
  exit 2
fi

REPO=${REPO:-/home/yuzhang_fei/code/SpatialTemporalGraph}
ARCHIVE=${ARCHIVE:-/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight}
PYTHON=${PYTHON:-/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python}
DIST_JOB="$REPO/mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/run_sd_osrm_distance_job.sh"
TOPK_BUILDER="$REPO/mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/build_topk_prior_graphs.py"
OUT_DIR="$ARCHIVE/distance_matrices/$DATASET"
GRAPH_DIR="$ARCHIVE/topk_prior_graphs_smallbench_$(date +%Y%m%d)/$DATASET"

echo "Preparing OSRM graph inputs for $DATASET"
echo "META=$META"
echo "OUT_DIR=$OUT_DIR"
COMPUTE=osrm DATASET="$DATASET" META="$META" OUT_DIR="$OUT_DIR" "$DIST_JOB"

DIST_SRC="$OUT_DIR/${DATASET}_osrm_shortest_distance_m.npy"
DIST_DST="$REPO/BasicTS/datasets/${DATASET}_osrm_shortest_distance_m.npy"
if [[ ! -f "$DIST_SRC" ]]; then
  echo "OSRM distance matrix was not generated: $DIST_SRC" >&2
  exit 1
fi
ln -sfn "$DIST_SRC" "$DIST_DST"

"$PYTHON" "$TOPK_BUILDER" \
  --dataset "$DATASET" \
  --distance "$DIST_DST" \
  --distance-name OSRMTOPK \
  --distance-label osrm_shortest_distance \
  --base-dataset "$REPO/BasicTS/datasets/$DATASET" \
  --output-graph-dir "$GRAPH_DIR" \
  --output-dataset-root "$REPO/BasicTS/datasets" \
  --k-list 64 \
  --gsp-pool-k 128 \
  --overwrite

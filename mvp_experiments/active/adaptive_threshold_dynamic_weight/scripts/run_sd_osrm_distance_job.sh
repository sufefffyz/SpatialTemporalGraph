#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/yuzhang_fei/code/SpatialTemporalGraph}
ARCHIVE=${ARCHIVE:-/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight}
DATASET=${DATASET:-SD}
META=${META:-$REPO/BasicTS/datasets/SD_5min_full/meta.csv}
PBF_SRC=${PBF_SRC:-$REPO/BasicTS/PEMS/graph_construction/california-latest.osm.pbf}
OSRM_DIR=${OSRM_DIR:-$ARCHIVE/osrm_california_20260511}
OUT_DIR=${OUT_DIR:-$ARCHIVE/distance_matrices/$DATASET}
LOG_DIR=${LOG_DIR:-$ARCHIVE/logs}
RUN_ID=${RUN_ID:-${DATASET}_distance_$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${RUN_ID}.log}

PYTHON=${PYTHON:-/home/yuzhang_fei/miniconda3/envs/STGraph/bin/python}
OSRM_IMAGE=${OSRM_IMAGE:-ghcr.io/project-osrm/osrm-backend:latest}
OSRM_PORT=${OSRM_PORT:-5012}
OSRM_BLOCK_SIZE=${OSRM_BLOCK_SIZE:-64}
OSRM_TIMEOUT=${OSRM_TIMEOUT:-300}
OSRM_RETRIES=${OSRM_RETRIES:-2}
OSRM_MAX_TABLE_SIZE=${OSRM_MAX_TABLE_SIZE:-1000000}

BUILDER="$REPO/mvp_experiments/active/adaptive_threshold_dynamic_weight/scripts/build_distance_matrices.py"
PBF_DST="$OSRM_DIR/$(basename "$PBF_SRC")"
OSRM_BASE="${PBF_DST%.osm.pbf}"
OSRM_MARKER="$OSRM_DIR/.california_mld_ready"
CONTAINER_NAME="stg_osrm_${RUN_ID}"

mkdir -p "$OSRM_DIR" "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

stage_start=0
job_start=$(date +%s)

begin_stage() {
  stage_start=$(date +%s)
  echo
  echo "== [$RUN_ID] $1 =="
  date "+%Y-%m-%d %H:%M:%S"
}

end_stage() {
  local now
  now=$(date +%s)
  echo "-- stage_elapsed_s=$((now - stage_start))"
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "RUN_ID=$RUN_ID"
echo "REPO=$REPO"
echo "ARCHIVE=$ARCHIVE"
echo "DATASET=$DATASET"
echo "META=$META"
echo "PBF_SRC=$PBF_SRC"
echo "OSRM_DIR=$OSRM_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "OSRM_PORT=$OSRM_PORT"
echo "OSRM_BLOCK_SIZE=$OSRM_BLOCK_SIZE"

begin_stage "preflight"
test -f "$META"
test -f "$PBF_SRC"
test -f "$BUILDER"
test -x "$PYTHON"
docker info >/dev/null
df -h "$ARCHIVE" "$PBF_SRC" || true
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep -F "$OSRM_IMAGE" || true
end_stage

begin_stage "copy_pbf"
src_size=$(stat -c "%s" "$PBF_SRC")
dst_size=0
if [ -f "$PBF_DST" ]; then
  dst_size=$(stat -c "%s" "$PBF_DST")
fi
if [ "$src_size" != "$dst_size" ]; then
  echo "Copying PBF to $PBF_DST"
  rm -f "$PBF_DST"
  cp "$PBF_SRC" "$PBF_DST"
else
  echo "PBF already present with matching size: $PBF_DST"
fi
end_stage

if [ ! -f "$OSRM_MARKER" ]; then
  begin_stage "osrm_extract"
  docker run --rm --user "$(id -u):$(id -g)" -v "$OSRM_DIR:/data" "$OSRM_IMAGE" \
    osrm-extract -p /opt/car.lua "/data/$(basename "$PBF_DST")"
  end_stage

  begin_stage "osrm_partition"
  docker run --rm --user "$(id -u):$(id -g)" -v "$OSRM_DIR:/data" "$OSRM_IMAGE" \
    osrm-partition "/data/$(basename "$OSRM_BASE").osrm"
  end_stage

  begin_stage "osrm_customize"
  docker run --rm --user "$(id -u):$(id -g)" -v "$OSRM_DIR:/data" "$OSRM_IMAGE" \
    osrm-customize "/data/$(basename "$OSRM_BASE").osrm"
  touch "$OSRM_MARKER"
  end_stage
else
  echo
  echo "== [$RUN_ID] osrm_preprocess =="
  echo "Using existing OSRM MLD files: $OSRM_MARKER"
fi

begin_stage "start_osrm_routed"
cleanup
docker run -d --rm --name "$CONTAINER_NAME" \
  -p "127.0.0.1:${OSRM_PORT}:5000" \
  -v "$OSRM_DIR:/data:ro" \
  "$OSRM_IMAGE" \
  osrm-routed --algorithm mld --max-table-size "$OSRM_MAX_TABLE_SIZE" "/data/$(basename "$OSRM_BASE").osrm"
for _ in $(seq 1 120); do
  if "$PYTHON" "$BUILDER" \
    --dataset "$DATASET" \
    --meta "$META" \
    --output-dir "$OUT_DIR/probe" \
    --osrm-url "http://127.0.0.1:${OSRM_PORT}" \
    --osrm-timeout 30 \
    --osrm-retries 0 \
    --probe-only >/tmp/"$RUN_ID"_probe.log 2>&1; then
    cat /tmp/"$RUN_ID"_probe.log
    break
  fi
  sleep 2
done
if ! grep -q "route_distance_m" /tmp/"$RUN_ID"_probe.log; then
  cat /tmp/"$RUN_ID"_probe.log || true
  echo "OSRM probe did not succeed."
  exit 1
fi
end_stage

begin_stage "build_distance_matrices"
"$PYTHON" "$BUILDER" \
  --dataset "$DATASET" \
  --meta "$META" \
  --output-dir "$OUT_DIR" \
  --compute straight,osrm \
  --osrm-url "http://127.0.0.1:${OSRM_PORT}" \
  --osrm-block-size "$OSRM_BLOCK_SIZE" \
  --osrm-timeout "$OSRM_TIMEOUT" \
  --osrm-retries "$OSRM_RETRIES" \
  --overwrite
end_stage

begin_stage "final_summary"
ls -lh "$OUT_DIR"
cat "$OUT_DIR/${DATASET}_distance_summary.json"
end_stage

job_end=$(date +%s)
echo
echo "TOTAL_ELAPSED_SECONDS=$((job_end - job_start))"
date "+%Y-%m-%d %H:%M:%S"

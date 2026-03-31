#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-benchmark}"
SCORE_N_JOBS="${SCORE_N_JOBS:-1}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
LOG_DIR="${LOG_DIR:-logs}"
METHODS="${METHODS:-var cc rp pcmci varlingam}"
N_VARS_LIST="${N_VARS_LIST:-3 5}"
REGIONS="${REGIONS:-east bav flood}"

mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  STRATEGIES=("$@")
else
  STRATEGIES=(debug_set random 1_random root_cause confounder close)
fi

read -r -a METHODS_ARRAY <<< "$METHODS"
read -r -a N_VARS_ARRAY <<< "$N_VARS_LIST"
read -r -a REGION_ARRAY <<< "$REGIONS"

method_override() {
  local method="$1"
  case "$method" in
    var)
      echo 'methods=[{name:var,max_lag:3,var_absolute_values:true,map_to_summary_graph:max,var_min_std:1.0e-12,var_verbose_failures:false}]'
      ;;
    cc)
      echo 'methods=[{name:cc,max_lag:3,cc_absolute_values:true,map_to_summary_graph:max,naive_child_selection:all}]'
      ;;
    rp)
      echo 'methods=[{name:rp,max_lag:3,rp_score_mode:difference,naive_child_selection:all}]'
      ;;
    pcmci)
      echo 'methods=[{name:pcmci,max_lag:3,map_to_summary_graph:max,pcmci_absolute_values:true,pcmci_pc_alpha:0.05,pcmci_alpha_level:0.05,pcmci_filter_nonsignificant:true,pcmci_verbosity:0}]'
      ;;
    varlingam)
      echo 'methods=[{name:varlingam,max_lag:3,map_to_summary_graph:max,varlingam_absolute_values:true,varlingam_criterion:null,varlingam_prune:true,varlingam_random_state:0}]'
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

region_data_path() {
  local region="$1"
  case "$region" in
    east)
      echo "product/rivers_ts_east_germany.csv"
      ;;
    bav)
      echo "product/rivers_ts_bavaria.csv"
      ;;
    flood)
      echo "product/rivers_ts_flood.csv"
      ;;
    *)
      echo "Unknown region: $region" >&2
      return 1
      ;;
  esac
}

label_count() {
  local label_path="$1"
  "$PYTHON_BIN" -c 'import pickle, sys; print(len(pickle.load(open(sys.argv[1], "rb"))))' "$label_path"
}

FAILED_RUNS=()
SKIPPED_RUNS=()

echo "Config       : $CONFIG_NAME"
echo "Regions      : ${REGION_ARRAY[*]}"
echo "Strategies   : ${STRATEGIES[*]}"
echo "n_vars list  : ${N_VARS_ARRAY[*]}"
echo "Methods      : ${METHODS_ARRAY[*]}"
echo "Score n_jobs : $SCORE_N_JOBS"
echo "Score chunk  : $SCORE_CHUNK_SIZE"
echo

for n_vars in "${N_VARS_ARRAY[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    label_dir="datasets/${strategy}_${n_vars}"

    if [ ! -d "$label_dir" ]; then
      echo "Missing label directory: $label_dir" >&2
      SKIPPED_RUNS+=("${label_dir}:missing-dir")
      echo >&2
      continue
    fi

    for region in "${REGION_ARRAY[@]}"; do
      label_path="${label_dir}/${region}.p"
      data_path="$(region_data_path "$region")" || {
        FAILED_RUNS+=("${region}:${strategy}:${n_vars}:unknown-region")
        continue
      }

      if [ ! -f "$label_path" ]; then
        echo "Missing label file: $label_path" >&2
        SKIPPED_RUNS+=("${region}:${strategy}:${n_vars}:missing-label")
        continue
      fi

      if [ ! -f "$data_path" ]; then
        echo "Missing data file: $data_path" >&2
        SKIPPED_RUNS+=("${region}:${strategy}:${n_vars}:missing-data")
        continue
      fi

      count="$(label_count "$label_path" 2>/dev/null || echo -1)"
      if [ "$count" -le 0 ]; then
        echo "Skipping empty label set: $label_path"
        SKIPPED_RUNS+=("${region}:${strategy}:${n_vars}:empty-labels")
        continue
      fi

      for method in "${METHODS_ARRAY[@]}"; do
        override="$(method_override "$method")" || {
          FAILED_RUNS+=("${region}:${strategy}:${n_vars}:${method}:unknown-method")
          continue
        }
        log_path="${LOG_DIR}/rivers_${region}_${strategy}_${n_vars}_${method}.log"

        echo "Running region   : $region"
        echo "Strategy         : $strategy"
        echo "n_vars           : $n_vars"
        echo "Method           : $method"
        echo "Label path       : $label_path"
        echo "Data path        : $data_path"
        echo "Label count      : $count"
        echo "Log              : $log_path"

        if OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
          "$PYTHON_BIN" benchmark.py --config-name "$CONFIG_NAME" \
            load_mode=eager \
            n_jobs=1 \
            score_n_jobs="$SCORE_N_JOBS" \
            score_chunk_size="$SCORE_CHUNK_SIZE" \
            label_path="$label_path" \
            data_path="$data_path" \
            "$override" \
            2>&1 | tee "$log_path"; then
          echo "Finished: ${region}/${strategy}/${n_vars}/${method}"
        else
          status=$?
          echo "Failed: ${region}/${strategy}/${n_vars}/${method} (exit=${status})" >&2
          FAILED_RUNS+=("${region}:${strategy}:${n_vars}:${method}:exit-${status}")
        fi

        echo
      done
    done
  done
done

if [ "${#SKIPPED_RUNS[@]}" -gt 0 ]; then
  echo "Skipped runs:"
  printf '  %s\n' "${SKIPPED_RUNS[@]}"
  echo
fi

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo "Completed with failures:" >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

echo "All rivers runs completed successfully."

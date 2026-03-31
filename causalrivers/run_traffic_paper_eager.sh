#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-benchmark_traffic_paper_baselines}"
DATASET_NAME="${DATASET_NAME:-city_traffic_m_volume__category__1_0}"
LABEL_TAG="${LABEL_TAG:-paperlike}"
N_VARS="${N_VARS:-3}"
SCORE_N_JOBS="${SCORE_N_JOBS:-32}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-512}"
LOG_DIR="${LOG_DIR:-logs}"
METHODS="${METHODS:-var cc rp combo pcmci varlingam dynotears cdmi}"

mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  STRATEGIES=("$@")
else
  STRATEGIES=(random confounder close root_cause)
fi

read -r -a METHODS_ARRAY <<< "$METHODS"

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
    combo)
      echo 'methods=[{name:combo,max_lag:3,cc_absolute_values:true,map_to_summary_graph:max,rp_score_mode:difference,combo_union_mode:max,naive_child_selection:all}]'
      ;;
    pcmci)
      echo 'methods=[{name:pcmci,max_lag:3,map_to_summary_graph:max,pcmci_absolute_values:true,pcmci_pc_alpha:0.05,pcmci_alpha_level:0.05,pcmci_filter_nonsignificant:true,pcmci_verbosity:0}]'
      ;;
    varlingam)
      echo 'methods=[{name:varlingam,max_lag:3,map_to_summary_graph:max,varlingam_absolute_values:true,varlingam_criterion:null,varlingam_prune:true,varlingam_random_state:0}]'
      ;;
    dynotears)
      echo 'methods=[{name:dynotears,max_lag:3,map_to_summary_graph:max,dynotears_absolute_values:true,dynotears_lambda_w:0.1,dynotears_lambda_a:0.1,dynotears_max_iter:100,dynotears_h_tol:1.0e-8,dynotears_w_threshold:0.0}]'
      ;;
    cdmi)
      echo 'methods=[{name:cdmi,max_lag:3,cdmi_epochs:15,cdmi_pred_len:1,cdmi_train_len:111,cdmi_num_layers:4,cdmi_num_cells:40,cdmi_num_samples:5,cdmi_dropout_rate:0.1,cdmi_step_size:1,cdmi_num_sliding_win:15,cdmi_alpha:0.10,cdmi_plot_forecasts:false}]'
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

FAILED_RUNS=()

echo "Config      : $CONFIG_NAME"
echo "Dataset     : $DATASET_NAME"
echo "Label tag   : $LABEL_TAG"
echo "n_vars      : $N_VARS"
echo "Strategies  : ${STRATEGIES[*]}"
echo "Methods     : ${METHODS_ARRAY[*]}"
echo "Score n_jobs: $SCORE_N_JOBS"
echo "Score chunk : $SCORE_CHUNK_SIZE"
if [ -n "${CDMI_REPO_PATH:-}" ]; then
  echo "CDMI repo   : $CDMI_REPO_PATH"
fi
if [ -n "${CDMI_PYTHON_BIN:-}" ]; then
  echo "CDMI python : $CDMI_PYTHON_BIN"
fi
echo

for strategy in "${STRATEGIES[@]}"; do
  label_path="datasets/traffic_${DATASET_NAME}/${strategy}_${N_VARS}_${LABEL_TAG}/${DATASET_NAME}.p"

  if [ ! -f "$label_path" ]; then
    echo "Missing label file: $label_path" >&2
    echo "Skip strategy: $strategy" >&2
    echo >&2
    continue
  fi

  for method in "${METHODS_ARRAY[@]}"; do
    override="$(method_override "$method")" || {
      FAILED_RUNS+=("${strategy}:${method}:unknown-method")
      continue
    }
    log_path="${LOG_DIR}/${DATASET_NAME}_${strategy}_${N_VARS}_${LABEL_TAG}_${method}_eager.log"

    echo "Running strategy: $strategy"
    echo "Method         : $method"
    echo "Label path     : $label_path"
    echo "Log            : $log_path"

    if OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON_BIN" benchmark.py --config-name "$CONFIG_NAME" \
        load_mode=eager \
        n_jobs=1 \
        score_n_jobs="$SCORE_N_JOBS" \
        score_chunk_size="$SCORE_CHUNK_SIZE" \
        label_path="$label_path" \
        "$override" \
        2>&1 | tee "$log_path"; then
      echo "Finished: ${strategy}/${method}"
    else
      status=$?
      echo "Failed: ${strategy}/${method} (exit=${status})" >&2
      FAILED_RUNS+=("${strategy}:${method}:exit-${status}")
    fi

    echo
  done
done

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo "Completed with failures:" >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

echo "All strategy/method runs completed successfully."

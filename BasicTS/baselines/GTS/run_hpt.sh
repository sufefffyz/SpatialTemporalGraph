#!/usr/bin/env bash
set -euo pipefail

########################
# 可配置参数
########################
GPUS=(0 1 2 3)                       # 物理 GPU id
MIN_FREE_MEM=20000                   # MB，显存不足就跳过
SLEEP_SEC=10

datasets=(PEMS03 PEMS04 PEMS07 PEMS08)
lrs=(0.1 0.01 0.001)
lamdas=(0 1)

LOG_DIR="logs/gts"
mkdir -p "${LOG_DIR}"

TMP_CFG_DIR="tmp_cfgs/gts"
mkdir -p "${TMP_CFG_DIR}"

########################
# 内部变量
########################
MAX_JOBS=${#GPUS[@]}                 # 同时最多跑 GPU 数个进程
JOB_COUNT=0

########################
# 工具函数
########################

running_jobs() {
  jobs -rp | wc -l
}

gpu_free_mem_mb() {
  local gpu_id="$1"
  nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
}

pick_gpu_with_mem() {
  local tries=${#GPUS[@]}
  local idx=$((JOB_COUNT % ${#GPUS[@]}))

  for ((t=0; t<tries; t++)); do
    local gpu=${GPUS[$idx]}
    local free_mem
    free_mem=$(gpu_free_mem_mb "$gpu" || echo 0)
    [[ -z "${free_mem}" ]] && free_mem=0

    if (( free_mem >= MIN_FREE_MEM )); then
      echo "$gpu"
      return 0
    fi
    idx=$(( (idx + 1) % ${#GPUS[@]} ))
  done
  return 1
}

wait_for_slot_and_mem() {
  while true; do
    if (( $(running_jobs) >= MAX_JOBS )); then
      sleep "${SLEEP_SEC}"
      continue
    fi
    if pick_gpu_with_mem >/dev/null 2>&1; then
      return 0
    fi
    sleep "${SLEEP_SEC}"
  done
}

########################
# 主循环
########################
for data in "${datasets[@]}"; do
  base_cfg="baselines/GTS/${data}.py"

  for lr in "${lrs[@]}"; do
    for lamda in "${lamdas[@]}"; do

      # 等待直到有空位且有足够显存的 GPU
      wait_for_slot_and_mem

      # 选 GPU（轮询 + 显存阈值）
      gpu=$(pick_gpu_with_mem)
      echo "Selected gpu=${gpu} (free_mem=$(gpu_free_mem_mb "$gpu")MB)"
      JOB_COUNT=$((JOB_COUNT + 1))
      lr_tag="${lr//./p}"
      # ? 每个实验复制一份临时 cfg，避免并行时互相覆盖
      ts=$(date +%Y%m%d_%H%M%S_%N)
      tmp_cfg="baselines/GTS/${data}_lr${lr_tag}_lamda${lamda}_gpu${gpu}_${ts}.py"
      cp "${base_cfg}" "${tmp_cfg}"

      # 修改 lr（限定在 CFG.TRAIN.OPTIM.PARAM 块内）
      sed -i '/CFG\.TRAIN\.OPTIM\.PARAM[[:space:]]*=/,/^[[:space:]]*}/ s/"lr":[[:space:]]*[0-9.eE+-]\+/"lr": '"$lr"'/' "${tmp_cfg}"

      # 修改 lamda（限定在 MODEL_PARAM 块内，不匹配 {）
      sed -i '/MODEL_PARAM[[:space:]]*=/,/^[[:space:]]*}/ s/"lamda":[[:space:]]*[0-9.eE+-]\+/"lamda": '"$lamda"'/' "${tmp_cfg}"


      # （可选）启动前自检：把改后的 lr/lamda 打到 stdout，方便排查
      echo "[CHECK] $(basename "$tmp_cfg") -> $(grep -n '"lr"' "$tmp_cfg" | head -n 1) | $(grep -n '"lamda"' "$tmp_cfg" | head -n 1)"

      log_file="${LOG_DIR}/${data}_lr${lr}_lamda${lamda}_gpu${gpu}.log"

      echo "? Launch ${data} | lr=${lr} | lamda=${lamda} | GPU=${gpu}"
      echo "? Log -> ${log_file}"

      # 用物理 GPU id（按你要求）
      nohup bash -c "python experiments/train.py -c '${tmp_cfg}' --gpus '${gpu}' > '${log_file}' 2>&1; rm -f '${tmp_cfg}'" &


    done
  done
done

wait
echo "? All experiments finished."

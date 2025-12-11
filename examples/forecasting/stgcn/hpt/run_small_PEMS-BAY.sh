#!/usr/bin/env bash

# 要搜索的学习率列表
lrs=(0.0007 0.001 0.003 0.005 0.009)

# 如果以后想搜多 seed，可以在这里加
seeds=(42)

# 固定的 weight_decay
weight_decay=0.0

for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    echo "========================================"
    echo "Running: seed=${seed}, lr=${lr}, weight_decay=${weight_decay}"
    echo "========================================"

    python examples/forecasting/stgcn/hpt/run_small.py \
      --seed="${seed}" \
      --lr="${lr}" \
      --weight_decay="${weight_decay}" \
      --data_name="PEMS-BAY" \
      --n_vertex=325 \
      --input_len=12 \
      --output_len=12 \
      --num_epochs=100 \
      --batch_size=64 \
      --gpus=2
  done
done

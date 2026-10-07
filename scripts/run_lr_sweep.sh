#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jk/work/models/OLMo-2-0425-1B}"
TRAIN_PATH="${TRAIN_PATH:-data/gsm8k/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/gsm8k/test.jsonl}"

run_one() {
  local lr="$1" policy_gpu="$2" inference_gpu="$3" port="$4" seed="$5" out="$6"
  uv run python -u scripts/train_grpo.py \
    --model-path "$MODEL_PATH" \
    --train-path "$TRAIN_PATH" \
    --validation-path "$VAL_PATH" \
    --train-limit 6400 \
    --validation-limit 1024 \
    --prompt-type r1_zero \
    --num-steps 200 \
    --rollout-batch-size 256 \
    --group-size 8 \
    --gradient-accumulation-steps 32 \
    --learning-rate "$lr" \
    --policy-device "cuda:${policy_gpu}" \
    --inference-gpu "$inference_gpu" \
    --port "$port" \
    --output-dir "$out" \
    --seed "$seed" \
    --eval-interval 10 \
    --rollout-log-interval 40
}

if [[ "${1:-stage1}" == "stage1" ]]; then
  run_one 5e-6 0 1 8010 0 experiments/lr_5e-6_seed0 &
  run_one 1e-5 2 3 8011 0 experiments/lr_1e-5_seed0 &
  run_one 2e-5 4 5 8012 0 experiments/lr_2e-5_seed0 &
  wait
elif [[ "$1" == "stage2" ]]; then
  # 根据 stage1 的结果，将最优的两个学习率传入本阶段。
  run_one "${LR_A:?set LR_A}" 0 1 8013 1 "experiments/lr_${LR_A}_seed1" &
  run_one "${LR_B:?set LR_B}" 2 3 8014 1 "experiments/lr_${LR_B}_seed1" &
  wait
else
  echo "用法: $0 [stage1|stage2]"
  exit 2
fi

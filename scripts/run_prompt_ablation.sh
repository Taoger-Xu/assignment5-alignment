#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jk/work/models/OLMo-2-0425-1B}"
TRAIN_PATH="${TRAIN_PATH:-data/gsm8k/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/gsm8k/test.jsonl}"
LR="${LR:-1e-5}"; STEPS="${STEPS:-200}"
OUT_ROOT="${OUT_ROOT:-experiments/grpo_prompt_ablation}"
OFFPOLICY_ROOT="${OFFPOLICY_ROOT:-experiments/grpo_offpolicy}"
GPU_PAIRS=(${GPU_PAIRS:-"0,1 2,3 4,5 6,7"})

echo "Waiting for off-policy experiments to finish..."
while [[ ! -f "${OFFPOLICY_ROOT}/.complete" ]]; do sleep 30; done
while pgrep -f 'scripts/train_grpo.py' >/dev/null; do sleep 30; done
echo "Off-policy experiments finished; starting prompt ablation."

run_one() {
  local prompt="$1" seed="$2" pair="$3" port="$4"
  local policy_gpu="${pair%,*}" inference_gpu="${pair#*,}"
  local out="${OUT_ROOT}/${prompt}_seed${seed}"
  mkdir -p "$out"
  uv run python -u scripts/train_grpo.py \
    --model-path "$MODEL_PATH" --train-path "$TRAIN_PATH" --validation-path "$VAL_PATH" \
    --prompt-type "$prompt" --num-steps "$STEPS" --train-limit 6400 --validation-limit 1024 \
    --rollout-batch-size 256 --group-size 8 --gradient-accumulation-steps 32 \
    --learning-rate "$LR" --loss-normalization sequence \
    --policy-device "cuda:${policy_gpu}" --inference-gpu "$inference_gpu" \
    --port "$port" --output-dir "$out" --seed "$seed" --eval-interval 10 --rollout-log-interval 40
}

jobs=(); slot=0; port=8300
for prompt in question_only r1_zero_three_shot; do
  for seed in 0 1 2 3; do
    run_one "$prompt" "$seed" "${GPU_PAIRS[$slot]}" "$port" &
    jobs+=("$!"); slot=$((slot + 1)); port=$((port + 1))
    if (( ${#jobs[@]} == ${#GPU_PAIRS[@]} )); then
      wait "${jobs[@]}"; jobs=(); slot=0
    fi
  done
done
(( ${#jobs[@]} == 0 )) || wait "${jobs[@]}"
echo "Prompt ablation experiments completed."

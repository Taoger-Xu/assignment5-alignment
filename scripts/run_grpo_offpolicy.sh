#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jk/work/models/OLMo-2-0425-1B}"
TRAIN_PATH="${TRAIN_PATH:-data/gsm8k/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/gsm8k/test.jsonl}"
LR="${LR:-1e-5}"; STEPS="${STEPS:-200}"
OUT_ROOT="${OUT_ROOT:-experiments/grpo_offpolicy}"
GPU_PAIRS=(${GPU_PAIRS:-"0,1 2,3 4,5 6,7"})

echo "Waiting for previous GRPO jobs to release GPUs..."
while pgrep -f 'scripts/train_grpo.py' >/dev/null; do sleep 30; done
echo "Previous jobs finished; starting off-policy experiments."

declare -A METHOD=([offpolicy_naive]=none [offpolicy_noclip]=noclip [offpolicy_clip]=grpo [offpolicy_gspo]=gspo)
declare -A CLIP=([offpolicy_naive]="" [offpolicy_noclip]="" [offpolicy_clip]=0.2 [offpolicy_gspo]=0.0003)

run_one() {
  local name="$1" seed="$2" pair="$3" port="$4"
  local policy_gpu="${pair%,*}" inference_gpu="${pair#*,}"
  local out="${OUT_ROOT}/${name}_seed${seed}"
  mkdir -p "$out"
  local args=(--importance-reweighting-method "${METHOD[$name]}" --train-batch-size 8)
  [[ -n "${CLIP[$name]}" ]] && args+=(--cliprange "${CLIP[$name]}")
  uv run --locked --extra gpu python -u scripts/train_grpo.py \
    --model-path "$MODEL_PATH" --train-path "$TRAIN_PATH" --validation-path "$VAL_PATH" \
    --prompt-type r1_zero --num-steps "$STEPS" --train-limit 6400 --validation-limit 1024 \
    --rollout-batch-size 256 --group-size 8 --gradient-accumulation-steps 1 \
    --learning-rate "$LR" --loss-normalization sequence "${args[@]}" \
    --policy-device "cuda:${policy_gpu}" --inference-gpu "$inference_gpu" \
    --port "$port" --output-dir "$out" --seed "$seed" --eval-interval 10 --rollout-log-interval 40
}

jobs=(); slot=0; port=8200
for name in offpolicy_naive offpolicy_noclip offpolicy_clip offpolicy_gspo; do
  for seed in 0 1 2 3; do
    run_one "$name" "$seed" "${GPU_PAIRS[$slot]}" "$port" &
    jobs+=("$!"); slot=$((slot + 1)); port=$((port + 1))
    if (( ${#jobs[@]} == ${#GPU_PAIRS[@]} )); then
      wait "${jobs[@]}"; jobs=(); slot=0
    fi
  done
done
(( ${#jobs[@]} == 0 )) || wait "${jobs[@]}"
echo "All off-policy experiments completed."
touch "${OUT_ROOT}/.complete"

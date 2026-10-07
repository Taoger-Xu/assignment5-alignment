#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jk/work/models/OLMo-2-0425-1B}"
TRAIN_PATH="${TRAIN_PATH:-data/gsm8k/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/gsm8k/test.jsonl}"
LR="${LR:-1e-5}"
STEPS="${STEPS:-200}"
OUT_ROOT="${OUT_ROOT:-experiments/grpo_variants}"
GPU_PAIRS=(${GPU_PAIRS:-"0,1 2,3 4,5 6,7"})

declare -A BASELINE=([GRPO_constant]=mean [Dr_GRPO]=mean [RFT]=none [MaxRL]=mean)
declare -A ADV_NORM=([GRPO_constant]=std [Dr_GRPO]=none [RFT]=none [MaxRL]=mean)

run_one() {
  local method="$1" seed="$2" pair="$3" port="$4"
  local policy_gpu="${pair%,*}" inference_gpu="${pair#*,}"
  local out="${OUT_ROOT}/${method}_seed${seed}"
  mkdir -p "$out"
  echo "[$method seed=$seed] GPUs=$pair port=$port output=$out"
  uv run python -u scripts/train_grpo.py \
    --model-path "$MODEL_PATH" --train-path "$TRAIN_PATH" --validation-path "$VAL_PATH" \
    --prompt-type r1_zero --num-steps "$STEPS" \
    --train-limit 6400 --validation-limit 1024 \
    --rollout-batch-size 256 --group-size 8 --gradient-accumulation-steps 32 \
    --learning-rate "$LR" --baseline "${BASELINE[$method]}" \
    --advantage-normalizer "${ADV_NORM[$method]}" --loss-normalization constant \
    --policy-device "cuda:${policy_gpu}" --inference-gpu "$inference_gpu" \
    --port "$port" --output-dir "$out" --seed "$seed" \
    --eval-interval 10 --rollout-log-interval 40
}

jobs=()
slot=0
port=8100
for method in GRPO_constant Dr_GRPO RFT MaxRL; do
  for seed in 0 1 2 3; do
    pair="${GPU_PAIRS[$slot]}"
    run_one "$method" "$seed" "$pair" "$port" &
    jobs+=("$!")
    slot=$((slot + 1))
    port=$((port + 1))
    if (( ${#jobs[@]} % ${#GPU_PAIRS[@]} == 0 )); then
      wait "${jobs[@]}"
      jobs=()
      slot=0
    fi
  done
done
(( ${#jobs[@]} == 0 )) || wait "${jobs[@]}"
echo "All GRPO variant runs completed. Results: ${OUT_ROOT}/<method>_seed<seed>/"

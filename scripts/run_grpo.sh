#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
exec uv run --locked --extra gpu python -u scripts/train_grpo.py \
  --model-path "${MODEL_PATH:-/home/jk/work/models/OLMo-2-0425-1B}" \
  --policy-device "${POLICY_DEVICE:-cuda:0}" \
  --inference-gpu "${INFERENCE_GPU:-1}" \
  --port "${VLLM_PORT:-8000}" \
  --output-dir "${OUTPUT_DIR:-experiments/grpo}" "$@"

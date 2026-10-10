# 本机 GPU 训练环境

环境：Python 3.12，PyTorch 2.10.0 + CUDA 12.9，vLLM 0.19.1，FlashAttention 2.8.3，WandB 0.26.1。

安装或恢复 GPU 依赖：

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra gpu
```

启动标准 GRPO（GPU 0 训练，GPU 1 推理，默认 200 steps）：

```bash
bash scripts/run_grpo.sh
```

小规模试跑：

```bash
bash scripts/run_grpo.sh --num-steps 1 --train-limit 8 --validation-limit 2 \
  --rollout-batch-size 4 --group-size 2 --gradient-accumulation-steps 2 \
  --max-tokens 32 --eval-interval 1 --rollout-log-interval 1 \
  --gpu-memory-utilization 0.3 --port 8199 --output-dir results/gpu-smoke
```

可用 MODEL_PATH、POLICY_DEVICE、INFERENCE_GPU、OUTPUT_DIR 环境变量修改启动默认值，也可以直接传入命令行参数。模型默认位于 `/home/jk/work/models/OLMo-2-0425-1B`，数据位于 `data/gsm8k/`。

默认只记录本地日志。使用 WandB 时添加 `--wandb-project 项目名`，并先配置账号。

其他实验入口：`scripts/run_lr_sweep.sh`、`scripts/run_grpo_variants.sh`、`scripts/run_grpo_offpolicy.sh`、`scripts/run_prompt_ablation.sh`。这些脚本已明确启用 GPU extra。

后续运行 uv 时保留 `--extra gpu`，避免普通 `uv sync` 将可选 GPU 依赖移除。

本次验证：216 个包的兼容性检查通过；CUDA BF16 矩阵乘法和 FlashAttention kernel 通过；`tests/test_grpo.py` 共 19 项通过；OLMo + vLLM + NCCL 权重同步 + 单步训练 + 评估完整执行成功。试跑仅生成 32 tokens，reward 为 0，验证目标是运行链路。

试跑日志和指标：`results/gpu-smoke-20261008/`。

本次修复了 `cs336_alignment/models/logprobs.py` 的可选 attention_mask 参数。该目录被现有 `.gitignore` 的 `models/` 规则忽略，文件修改已在本机保存；提交代码时需要显式纳入版本管理。

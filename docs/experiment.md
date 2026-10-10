# GSM8K 实验

依据：[作业 PDF](../cs336_spring2026_assignment5_alignment.pdf) §4.3、第 5–7 章。命令从项目根目录执行；日志目录在运行时创建。

## 公共配置

入口：[train_grpo.py](../scripts/train_grpo.py)，单次启动：[run_grpo.sh](../scripts/run_grpo.sh)。训练循环为同步权重 → vLLM 生成 → policy-gradient 更新 → 定期验证。

- 模型：`/home/jk/work/models/OLMo-2-0425-1B`。
- 数据：`data/gsm8k/train.jsonl`、`data/gsm8k/test.jsonl`，使用 6400 / 1024 条。
- 默认参数：200 轮、LR `1e-5`、rollout/train batch 256、group size 8、梯度累积 32、temperature 1.0、max tokens 512、max grad norm 1.0。
- AdamW：betas=(0.9, 0.95)，weight_decay=0。
- 每 10 轮验证，每 40 轮保存样例；默认 GPU 0 训练、GPU 1 推理。批量脚本使用四对 GPU。

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra gpu
```

## 实验与日志目录

| 实验 | 思路与规模 | 脚本 | 输出目录 |
|---|---|---|---|
| 标准 GRPO 调试，PDF p25 | r1_zero，约 50 轮，观察奖励和回答是否改善 | [run_grpo.sh](../scripts/run_grpo.sh) | 指定 `experiments/grpo_debug_seed0/` |
| 标准 GRPO 正式，p25 | r1_zero，200 轮 × 4 seeds；最终平均准确率至少 25% | 同上，逐个指定 seed | 建议 `experiments/grpo_standard/seed<seed>/` |
| 学习率搜索，p26 | `5e-6 / 1e-5 / 2e-5`，比较最终验证奖励；候选值追加 seeds | [run_lr_sweep.sh](../scripts/run_lr_sweep.sh) | `experiments/lr_<lr>_seed<seed>/` |
| Prompt ablation，p26 | question_only、r1_zero_three_shot，各 4 seeds，与 r1_zero 对照 | [run_prompt_ablation.sh](../scripts/run_prompt_ablation.sh) | `experiments/grpo_prompt_ablation/<prompt>_seed<seed>/` |
| On-policy 变体，p31–32 | 四种方法各 4 seeds，固定其他参数 | [run_grpo_variants.sh](../scripts/run_grpo_variants.sh) | `experiments/grpo_variants/<method>_seed<seed>/` |
| Off-policy，p37–38 | 四种方法各 4 seeds；每轮 256 个回答分成 32 次更新 | [run_grpo_offpolicy.sh](../scripts/run_grpo_offpolicy.sh) | `experiments/grpo_offpolicy/<method>_seed<seed>/` |
| 自定义估计器，p38 | 相对已有方法只改一个因素，多 seed 比较并解释机制 | 尚未实现；扩展 advantages.py 或 policy_gradient.py | 建议 `experiments/grpo_custom/<method>_seed<seed>/` |

### 标准 GRPO

```bash
bash scripts/run_grpo.sh \
  --num-steps 50 --seed 0 --prompt-type r1_zero \
  --eval-interval 10 --rollout-log-interval 40 \
  --output-dir experiments/grpo_debug_seed0
```

正式实验分别传入 `--seed 0/1/2/3 --num-steps 200` 和独立输出目录。默认算法为 baseline=`mean`、advantage normalizer=`std`、loss normalization=`sequence`。

### 学习率与 prompt

```bash
bash scripts/run_lr_sweep.sh stage1
# 根据 stage1 结果选择候选值；下面仅为示例
LR_A=5e-6 LR_B=1e-5 bash scripts/run_lr_sweep.sh stage2
LR=1e-5 STEPS=200 bash scripts/run_prompt_ablation.sh
```

stage1 每个 LR 使用 seed 0；stage2 为两个候选 LR 增加 seed 1。是否增加更多 seeds 取决于方差。

Prompt 批量脚本会等待 `experiments/grpo_offpolicy/.complete`；独立运行可用 `run_grpo.sh --prompt-type question_only` 或 `r1_zero_three_shot`。Question-only 使用专用 reward，比较格式奖励时需考虑评分规则。

### On-policy 变体

```bash
LR=1e-5 STEPS=200 bash scripts/run_grpo_variants.sh
```

| 方法 | baseline | advantage normalizer | loss normalization |
|---|---|---|---|
| 标准 GRPO | mean | std | sequence |
| GRPO_constant | mean | std | constant |
| Dr_GRPO | mean | none | constant |
| RFT | none | none | constant |
| MaxRL | mean | mean | constant |

constant 分母为 `rollout_batch_size * max_tokens`。各方法使用相同学习率、prompt 和训练参数，与标准基线比较。

### Off-policy

```bash
# 每轮 32 次更新，clip fraction 自动写入 metrics.jsonl
LR=1e-5 STEPS=200 bash scripts/run_grpo_offpolicy.sh
```

训练 batch=8，梯度累积=1；重加权方法在更新前计算并固定 old_log_probs。

| 方法 | importance reweighting | cliprange |
|---|---|---|
| offpolicy_naive | none | — |
| offpolicy_noclip | noclip | — |
| offpolicy_clip | grpo：token ratio | 0.2 |
| offpolicy_gspo | gspo：sequence ratio | 0.0003 |

## 日志与分析

每个输出目录自动生成 `metrics.jsonl` 和 `rollouts.jsonl`。另保存 `config.json` 和 `checkpoints/step_<六位步数>/`。控制台日志需自行重定向到 `startup.log`。WandB 可通过 `--wandb-project` 启用，批量脚本默认只写本地日志。

| 指标 | split | JSON 字段 |
|---|---|---|
| loss / gradient norm / entropy | train | `loss` / `gradient_norm` / `token_entropy` |
| 训练 total / format / answer reward | train | `mean_reward` / `mean_format_reward` / `mean_answer_reward` |
| 验证 total / format / answer reward | val | `total_reward` / `format_reward` / `answer_reward` |
| 验证准确率 / 回答长度 | val | `accuracy` / `avg_response_length`（字符数） |
| clip fraction | train | `clip_fraction`：GRPO 按参与 forward 的有效 token，GSPO 按非空序列统计 ratio 越界比例；非裁剪方法为 0 |

`step` 表示 rollout 轮数。Off-policy 每轮内部更新指标目前只记录均值。Rollout 日志包含题目、回答、标准答案和嵌套 reward，step 0 保存验证基线；step 1、每 40 轮和最后一轮保存完整训练 batch 与前 20 个验证样例。

分析时按 step 对齐 seeds，绘制各指标的均值±标准差或 min/max；学习率搜索另画最终验证 reward 对 LR 的图。比较训练前后同一组题目的回答。使用调优 LR 时，标准基线也应使用相同 LR。

## 实现状态与限制

- naive 已按 `train_batch_size` 拆批：256 条 rollout、batch 8 时执行 32 次更新，与 reweighting 设置无关。
- 无跨 seed 汇总/绘图工具。
- RFT 与 GRPO 已跳过零 advantage 样本，保留原 batch 的 loss 分母和原 microbatch 大小；`num_active_sequences` 记录实际 forward 数量。裁剪后 entropy 与 `num_response_tokens` 仅统计参与 forward 的样本；整批为零时 entropy 为 0，并保持优化器更新语义。
- 日志追加写入，重跑需使用新目录；逐任务检查完成状态，不能只依赖 `.complete`。
- `cs336_alignment/models/logprobs.py` 被 `models/` 忽略规则排除，迁移前需纳入版本管理。

现有单步数据：`results/gpu-smoke-20261008/`。历史 50 步说明见 [grpo-experiment.md](grpo-experiment.md)，其引用的原始实验日志当前缺失。

## 多 GPU 端口

8 张 GPU 默认组成四对，每个并行实验需要独立端口。现有批量脚本分配：

| 入口 | vLLM 端口 |
|---|---|
| run_grpo.sh | 默认 8000；用 `--port` 或 `VLLM_PORT` 覆盖 |
| run_lr_sweep.sh | 8010–8014 |
| run_grpo_variants.sh | 8100–8115 |
| run_grpo_offpolicy.sh | 8200–8215 |
| run_prompt_ablation.sh | 8300–8307 |

手动同时启动四个标准实验时，可分别使用端口 8000–8003，GPU 对 0/1、2/3、4/5、6/7。重复启动同一批量脚本仍会复用端口。启动检查会拒绝已占用或已被其他实验预留的端口，不再终止已有服务；进程锁在服务关闭后释放。NCCL 权重同步端口由 vLLM 动态选择空闲端口，与这里的 HTTP 服务端口不同。不同脚本端口不重叠，但仍需避免使用同一 GPU 对。

## Checkpoint 保存与恢复

默认每 50 个 rollout steps 及最后一轮保存到 `<output-dir>/checkpoints/step_000050/`，包含模型、tokenizer、优化器、步数、训练配置与 Python/PyTorch/CUDA 随机状态。`--checkpoint-interval 10` 调整间隔，`--checkpoint-interval 0` 禁用保存，`--checkpoint-dir` 指定目录。

```bash
bash scripts/run_grpo.sh --num-steps 50 --checkpoint-interval 10 \
  --output-dir experiments/grpo_with_checkpoints

bash scripts/run_grpo.sh \
  --resume-from experiments/grpo_with_checkpoints/checkpoints/step_000050 \
  --num-steps 200 --output-dir experiments/grpo_with_checkpoints
```

`num_steps` 是目标总轮数，恢复后从已完成轮数 +1 开始。CLI 自动继承 checkpoint 中的原运行参数，训练/采样配置须一致；总轮数和日志、保存间隔可调整。恢复到原日志目录时清除 checkpoint 之后的未完成记录。仅恢复本项目生成的可信 checkpoint；vLLM 的内部随机状态未保存，不保证 GPU 生成逐位复现。

## 离线批量队列

入口：`scripts/run_experiment_queue.py --root <新目录>`。当前批次目录保存在 `experiments/latest_queue.txt`。先执行 16 个一步检查，全部通过后依次运行 50 步调试、四种子标准实验、学习率搜索、prompt、on-policy 和 off-policy 实验。四对 GPU 并行运行同一阶段的任务，端口 8500–8503 每批复用。

队列直接使用本地 `.venv`、模型和数据，HF 离线模式、WandB disabled，不运行 uv 同步。`queue.log` 保存队列日志，`status.json` 保存每个任务状态；各任务目录保存 `startup.log`、指标、rollouts 和 checkpoint。学习率搜索的 1e-5 复用标准四种子数据，其他两个 LR 各四 seeds；后续使用平均最终验证奖励最高的 LR。断网或关闭聊天不影响脱离终端的队列，关机/重启会中断。

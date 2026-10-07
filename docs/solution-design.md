# Assignment 5 第 4 章：On-policy GRPO 设计方案

> 本文把当前 `assignment5-alignment` 项目 PDF 第 4 章 **Implementing on-policy GRPO** 拆解为可实现的模块和任务边界。文档只描述架构、数据流、接口职责和验证方法，不提供作业实现代码。

## 1. 范围与设计目标

本方案针对以下任务：

1. `tokenize_prompt_and_output`
2. `get_response_log_probs`
3. `compute_rollout_rewards`
4. `compute_group_normalized_rewards_grpo`
5. `compute_policy_gradient_loss_on_policy`
6. `aggregate_loss_across_microbatch_sequence`
7. `grpo_train_step_standard_on_policy`
8. `grpo_experiments_standard_on_policy`

设计参考 `homework_spring2026/hw4/docs/design.md` 的分层思想：任务与数据、模型、rollout、RL 计算、训练、评估彼此分离，通过明确的数据对象连接。

```text
任务与数据
    ↓
Prompt 构造与 Tokenization
    ↓
模型推理 / vLLM Rollout
    ↓
Reward 计算
    ↓
Group Advantage 计算
    ↓
Policy Gradient Loss
    ↓
Microbatch 聚合与参数更新
    ↓
验证、日志、Checkpoint、实验报告
```

核心原则：

- `tests/adapters.py` 只负责测试接口适配，核心逻辑放在 `cs336_alignment/`。
- 数学组件保持独立，便于分别验证张量形状和边界条件。
- Trainer 负责流程编排，不直接实现所有底层算法。
- 数据对象明确保存 prompt、response、mask、reward 和 advantage，避免训练循环中使用大量未约束的字典。
- 先用小规模、固定随机种子的 sanity check，再运行完整实验。

## 2. 推荐目录结构

```text
assignment5-alignment/
├── cs336_alignment/
│   ├── data/
│   │   ├── gsm8k.py
│   │   ├── schemas.py
│   │   └── batching.py
│   ├── prompting/
│   │   ├── templates.py
│   │   └── tokenization.py
│   ├── models/
│   │   ├── policy.py
│   │   ├── logprobs.py
│   │   └── entropy.py
│   ├── rollout/
│   │   ├── sampler.py
│   │   ├── vllm_sampler.py
│   │   └── weight_sync.py
│   ├── rewards/
│   │   ├── gsm8k_reward.py
│   │   └── reward_pipeline.py
│   ├── rl/
│   │   ├── advantages.py
│   │   ├── policy_gradient.py
│   │   └── grpo_step.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── checkpointing.py
│   ├── checkpoint.py
│   └── vllm_utils.py
├── scripts/
│   ├── evaluate_prompting_baselines.py
│   ├── train_grpo.py
│   └── evaluate_grpo.py
├── tests/
│   ├── adapters.py
│   ├── test_grpo.py
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_dpo.py
├── data/
├── docs/
└── pyproject.toml
```

这些目录可以随着实现逐步创建，不需要一开始全部建立。

## 3. 核心数据对象

### 3.1 `GSM8KExample`

表示一条数据集样本，至少包含：

```text
question
answer
  ground_truth
prompt
metadata
```

`GSM8KTask` 负责读取 JSONL、提取最终答案、构造 prompt 和提供 batch。它不负责模型生成、log probability 或 optimizer 更新。

### 3.2 `TokenizedPromptOutput`

表示 prompt 和 response 的 token 化结果：

```text
input_ids
labels
response_mask
prompt_length
response_length
```

必须满足：

```text
len(input_ids) == len(labels) == len(response_mask)
```

prompt 区域的 `response_mask` 为 0，response 区域为 1。后续 loss 只在 response token 上计算。

### 3.3 `RolloutBatch`

表示一批采样结果：

```text
prompts
responses
input_ids
labels
response_mask
old_log_probs
rewards
format_rewards
answer_rewards
advantages
group_ids
```

如果原始 prompt 数量为 `B`，每个 prompt 生成 `G` 个 response，则 rollout 数量为：

```text
N = B × G
```

典型形状为：

```text
input_ids:      [N, sequence_length]
response_mask:  [N, sequence_length - 1]
old_log_probs:  [N, sequence_length - 1]
rewards:        [N]
advantages:     [N]
```

`group_ids` 用于把同一个 prompt 生成的多个 response 分组。

## 4. 各模块的职责

### 4.1 数据与任务层

推荐文件：

```text
cs336_alignment/data/gsm8k.py
cs336_alignment/data/schemas.py
cs336_alignment/data/batching.py
```

`GSM8KTask` 的职责：

1. 加载 train/validation JSONL；
2. 读取问题和标准答案；
3. 提取最终数值答案；
4. 调用 PromptRenderer 构造 prompt；
5. 组织 batch 和重复 prompt；
6. 保存样本与 group 的对应关系。

数据流为：

```text
JSONL → GSM8KExample → PromptRenderer → 重复 group_size 次 → vLLM
```

### 4.2 Prompt 与 Tokenization

推荐文件：

```text
cs336_alignment/prompting/templates.py
cs336_alignment/prompting/tokenization.py
```

`PromptRenderer` 负责：

- `question_only`
- `r1_zero`
- `r1_zero_three_shot`

它只生成字符串，不直接调用 tokenizer。

`tokenize_prompt_and_output` 的职责是：

1. 分别 tokenize prompt 和 response；
2. 避免错误添加额外 special token；
3. 拼接两部分 token；
4. 生成与 labels 对齐的 response mask；
5. 记录 prompt 和 response 长度。

应检查以下不变量：

```text
完整长度 = prompt 长度 + response 长度
prompt 区域 mask 全为 0
response 区域 mask 全为 1
mask 与 labels 长度一致
```

### 4.3 模型与 Log Probability

推荐文件：

```text
cs336_alignment/models/policy.py
cs336_alignment/models/logprobs.py
cs336_alignment/models/entropy.py
```

`PolicyModel` 封装 Hugging Face 模型、tokenizer、device 和 forward 操作。

`get_response_log_probs` 的逻辑链为：

```text
input_ids → model forward → logits → log_softmax → 根据 labels 取 token log probability
```

必须检查：

- causal language modeling 的 shift 是否正确；
- logits 和 labels 是否错位一个 token；
- padding 是否被屏蔽；
- 返回的序列长度是否与 response mask 对齐。

### 4.4 Rollout 与 vLLM

推荐文件：

```text
cs336_alignment/rollout/sampler.py
cs336_alignment/rollout/vllm_sampler.py
cs336_alignment/rollout/weight_sync.py
```

当前仓库的 `cs336_alignment/vllm_utils.py` 可以作为底层 vLLM 服务实现。上层建议使用统一的 `RolloutSampler` 接口，使 Trainer 不依赖具体推理后端。

vLLM 层负责：

- 根据 prompt 生成 response；
- temperature、top-p 和最大 token 数；
- `</answer>` 停止条件；
- 多 GPU 服务配置。

权重同步流程为：

```text
训练模型更新 → 同步到 vLLM → 生成新 rollout → 返回训练流程
```

该流程建议由 `WeightSynchronizer` 封装，不直接散落在训练循环中。

### 4.5 Reward 层

推荐文件：

```text
cs336_alignment/rewards/gsm8k_reward.py
cs336_alignment/rewards/reward_pipeline.py
```

`compute_rollout_rewards` 应返回：

```text
raw_rewards
format_rewards
answer_rewards
metadata
```

建议把流程拆成：

```text
AnswerParser → ParsedAnswer → FormatChecker → CorrectnessChecker → RewardCalculator
```

这样可以区分：

- 答案正确但格式错误；
- 格式正确但答案错误；
- 格式和答案都错误。

### 4.6 Group Advantage 层

推荐文件：

```text
cs336_alignment/rl/advantages.py
```

`compute_group_normalized_rewards_grpo` 对同一 prompt 的 `G` 个 reward 做组内处理：

```text
r_i - mean(group_rewards)
```

如果启用标准差归一化，再除以：

```text
std(group_rewards) + eps
```

需要支持：

```text
baseline = mean
advantage_normalization = std
advantage_normalization = none
```

必须处理组内标准差为 0 的情况，避免产生 NaN。

### 4.7 Policy Gradient Loss

推荐文件：

```text
cs336_alignment/rl/policy_gradient.py
```

`compute_policy_gradient_loss_on_policy` 使用 response token 的 log probability 和 advantage 计算逐 token loss。第 4 章的 on-policy 设置支持：

```text
importance_sampling = none
```

loss 只作用于 response token，prompt 和 padding 位置由 `response_mask` 排除。建议函数返回逐 token loss 及必要的统计信息，而不是直接在函数内部做最终 batch 聚合。

### 4.8 Microbatch 聚合

推荐文件：

```text
cs336_alignment/rl/policy_gradient.py
```

建议使用两级聚合：

```text
response token loss → 每条序列的平均 loss → microbatch 中序列的平均 loss
```

这样 response 较长的样本不会因为 token 数更多而自动获得更大权重。

需要明确处理：

- padding；
- response 长度为 0；
- gradient accumulation 中不同 microbatch 的权重。

### 4.9 单步 GRPO 训练

推荐文件：

```text
cs336_alignment/rl/grpo_step.py
```

`grpo_train_step_standard_on_policy` 负责连接前面的组件：

```text
prompts
  → responses
  → rewards
  → group advantages
  → tokenization
  → current log probs
  → policy gradient loss
  → microbatch aggregation
  → gradient accumulation
  → gradient clipping
  → optimizer.step()
```

应支持的配置包括：

```text
group_size
gradient_accumulation_steps
max_grad_norm
baseline
advantage_normalization
loss_aggregation
```

建议返回：

```text
loss
metrics
```

metrics 至少包括：

```text
loss
gradient_norm
token_entropy
mean_total_reward
mean_format_reward
mean_answer_reward
mean_response_length
num_valid_tokens
```

该函数不应负责读取文件、初始化 wandb、启动 vLLM 或绘图。

## 5. 训练器与实验脚本

推荐文件：

```text
cs336_alignment/training/trainer.py
cs336_alignment/training/evaluator.py
cs336_alignment/training/metrics.py
cs336_alignment/training/checkpointing.py
scripts/train_grpo.py
```

`GRPOTrainer` 只负责长期训练流程编排：

```text
for step:
    synchronize_weights()
    generate_rollouts()
    run_train_step()
    log_metrics()
    periodic_evaluate()
    periodic_checkpoint()
```

建议组成：

```text
GRPOTrainer
├── RolloutSampler
├── RewardEvaluator
├── GRPOTrainStep
├── Evaluator
├── MetricsLogger
└── CheckpointManager
```

`Evaluator` 负责固定验证集上的 response 生成、reward 统计、平均长度和样例保存。训练集和验证集应复用相同的 reward 逻辑。

实验脚本需要配置：

```text
model_name
tokenizer_name
train_file
validation_file
prompt_template
num_steps
batch_size
group_size
learning_rate
gradient_accumulation_steps
max_grad_norm
temperature
top_p
max_response_tokens
eval_interval
checkpoint_interval
seed
```

正式实验应支持多个随机种子，并保存训练曲线、验证曲线、最终 checkpoint 和训练前后的生成样例。

## 6. 任务与文件映射

| 作业任务 | 推荐实现位置 | 主要验证内容 |
|---|---|---|
| `tokenize_prompt_and_output` | `prompting/tokenization.py` | prompt/output 拼接与 mask |
| `get_response_log_probs` | `models/logprobs.py` | causal shift 和 response log prob |
| `compute_rollout_rewards` | `rewards/reward_pipeline.py` | format、correctness、total reward |
| `compute_group_normalized_rewards_grpo` | `rl/advantages.py` | group mean、std、eps |
| `compute_policy_gradient_loss_on_policy` | `rl/policy_gradient.py` | advantage 与 log probability |
| `aggregate_loss_across_microbatch_sequence` | `rl/policy_gradient.py` | sequence-level normalization |
| `grpo_train_step_standard_on_policy` | `rl/grpo_step.py` | 梯度累积、裁剪和更新 |
| `grpo_experiments_standard_on_policy` | `training/trainer.py`、`scripts/train_grpo.py` | 完整实验流程 |

## 7. 测试适配层

`tests/adapters.py` 不应成为核心实现文件。它的职责是：

```text
测试输入 → 调用 cs336_alignment 的正式实现 → 统一输出格式
```

推荐映射：

```text
run_tokenize_prompt_and_output
    → prompting.tokenization

run_get_response_log_probs
    → models.logprobs

run_compute_rollout_rewards
    → rewards.reward_pipeline

run_compute_group_normalized_rewards
    → rl.advantages

run_compute_policy_gradient_loss
    → rl.policy_gradient

run_aggregate_loss_across_microbatch
    → rl.policy_gradient

run_grpo_train_step
    → rl.grpo_step
```

这样测试接口和训练代码可以独立演化。

## 8. 推荐实现顺序

### 阶段一：数据和 tokenization

先完成 `GSM8KExample`、`PromptRenderer` 和 `tokenize_prompt_and_output`。重点检查拼接长度、mask 对齐和不同 response 长度。

### 阶段二：Log Probability

完成 `get_response_log_probs`，用小输入检查 causal shift、padding 和 response mask。

### 阶段三：Reward 与 Advantage

完成 reward pipeline 和 group normalization。先用手工 reward 验证均值、标准差、`eps` 和零方差情况。

### 阶段四：Loss

完成 policy gradient loss 和 sequence aggregation。使用长度不同的 response 验证长序列不会自动获得更大的权重。

### 阶段五：单步训练

完成 `grpo_train_step_standard_on_policy`。先使用小 batch、小 group size、少量 gradient accumulation 和固定随机种子，确认 loss 有限且参数发生变化。

### 阶段六：vLLM 集成

先独立验证 vLLM 启动、生成、stop token 和权重同步，再接入 Trainer。

### 阶段七：完整实验

先运行约 50 个 rollout steps 做 sanity check，确认 reward 和生成样例合理后，再进行正式多 seed 实验。

## 9. 训练与评估指标

训练阶段建议记录：

```text
train/loss
train/gradient_norm
train/token_entropy
train/total_reward
train/format_reward
train/answer_reward
train/response_length
train/advantage_mean
train/advantage_std
```

验证阶段建议记录：

```text
val/total_reward
val/format_reward
val/answer_reward
val/avg_response_length
val/accuracy
```

同时保存少量：

```text
prompt
generated_response
parsed_answer
ground_truth
format_reward
answer_reward
```

这些样例可以帮助定位格式错误、答案解析错误、重复生成、response 过长和训练退化等问题。

## 10. 最终对象关系

```text
GRPOExperimentRunner
        │
        ├── GRPOTrainer
        │      ├── Dataset
        │      ├── RolloutSampler
        │      ├── RewardEvaluator
        │      ├── GRPOTrainStep
        │      ├── Evaluator
        │      ├── MetricsLogger
        │      └── CheckpointManager
        │
        ├── PolicyModel
        ├── TokenizerProcessor
        └── WeightSynchronizer
```

单步计算链为：

```text
RolloutBatch
    ↓
RewardPipeline
    ↓
AdvantageEstimator
    ↓
LogProbCalculator
    ↓
PolicyGradientObjective
    ↓
LossAggregator
    ↓
Optimizer
```

该结构把实验流程、模型推理和 RL 数学组件分开，适合先完成单元测试，再逐步接入 vLLM 和完整训练。

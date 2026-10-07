# Assignment 5：任务清单与交付地图

依据本地 Spring 2026 主作业与可选 safety/RLHF 补充 PDF 整理。主作业版本为 26.0.0。项目架构、实战术语与资源状态见 [design.md](design.md)。任务 ID 保留原文，便于在 PDF 中检索；章节号对应原手册。本清单不包含推导答案、实现代码或自拟估计器方案。

## 1. 范围与提交

主作业是必做 Reasoning RL；整个 safety/RLHF supplement 是可选部分。主作业最后的 `try_your_own` 是正式任务，不应因为它具有探索性就当成可选。

主作业要求提交 `writeup.pdf` 与 `code.zip`，书面回答需要排版；仓库提供 `test_and_make_submission.sh` 打包入口。每一项完成都应同时检查：实现是否接入测试、实验是否留下证据、报告是否回答全部子问题。

## 2. 主作业：28 个任务

### 2.1 Prompting 与 baseline 理论

| 任务 ID／位置 | 分值 | 具体任务与交付物 |
|---|---:|---|
| `prompting_baselines`／§3 | 5 | 评估 OLMo-2-0425-1B 在 GSM8K 上的 question_only、r1_zero、r1_zero_three_shot 三种提示。(a) 统计“格式与答案都正确”“格式正确但答案不正确”“两者都不正确”三类生成；至少观察第二类十个样例，并检查第二、三类是否有正确但未被解析的回答。(b) 比较三种提示诱导的行为。交付指标、数句分析及 prompt/response 样例。 |
| `baseline_calcs`／§4.1 | 5 | 对手册给定二元动作策略，(a) 推导无 baseline 梯度估计器方差；(b) 推导带常数 baseline 的方差并讨论；(c) 代入 population mean baseline，与无 baseline 比较。交付相应表达式、推导与讨论。 |

这一阶段的实战目标是建立未训练模型的行为参照。baseline 实验和理论题中的 reward baseline 是不同含义。

### 2.2 标准 on-policy GRPO 组件

关联文件：学生实现放在 `cs336_alignment/`，通过 `tests/adapters.py` 对接 `tests/test_grpo.py`。下表描述接口责任，具体张量形状以 PDF §4.2 和 adapters 为准。

| 任务 ID | 分值 | 要求与验收 |
|---|---:|---|
| `tokenize_prompt_and_output` | 1 | 按手册契约处理 prompt/output，返回 input_ids、labels、与 labels 对齐的 response_mask；满足 tokenization 测试。 |
| `get_response_log_probs` | 1 | 返回每个目标 token 的条件 log-prob，支持可选 token entropy；满足概率与熵测试。 |
| `compute_rollout_rewards` | 1 | 对回答与对应 gold 调用评分，返回 raw rewards 和统计 metadata，至少记录平均总奖励与格式奖励；满足 rewards 测试。 |
| `compute_group_normalized_rewards_grpo` | 1 | 首先支持 group mean baseline、std advantage normalizer 与稳定项；返回优势及 metadata，满足标准 GRPO 归一化测试。 |
| `compute_policy_gradient_loss_on_policy` | 1 | 支持 on-policy 的 `importance_reweighting_method="none"`，接收奖励或优势，返回 token 级 loss 与 metadata；满足 on-policy loss 测试。 |
| `aggregate_loss_across_microbatch_sequence` | 0.5 | 支持 `loss_normalization="sequence"`，按 response mask 聚合为可反向传播的标量；满足 sequence aggregation 测试。 |
| `grpo_train_step_standard_on_policy` | 5 | 实现一次标准 on-policy 更新，整合前述组件、梯度累积、可选梯度裁剪与日志 metadata；仅需支持 mean/std/none/sequence 的标准组合，满足对应 train-step 测试。 |

这里的实现题交付函数/方法及测试连接。通过局部测试后，仍需用完整实验验证数据、采样策略、权重同步与日志是否一起工作。

### 2.3 标准 GRPO 实验／§4.3

| 任务 ID | 分值／手册预算 | 具体任务与交付物 |
|---|---|---|
| `grpo_experiments_standard_on_policy` | 10／2 B200 hrs | (a) 编写可配置模型、prompt、数据、采样与训练参数的 GRPO 训练脚本。(b) 先运行约 50 步，提供让你相信端到端训练正确的证据。(c) 用 r1_zero 和手册设置运行 4 个随机种子，记录规定曲线及跨种子变化；观察并展示训练前后回答。交付脚本、运行证据、每项指标曲线与评论、回答样例；最终验证准确率跨种子平均至少 25%。 |
| `grpo_learning_rate` | 3／4 B200 hrs | 学习率 sweep 至少包括比默认更小与更大的值；根据已有方差决定每组运行的种子数。交付最终验证 reward 对学习率的图及评论，注明发散情况。 |
| `grpo_prompt_ablation` | 3／4 B200 hrs | 用 question_only 和三示例 prompt、多随机种子，与已有 r1_zero 结果比较；讨论平均奖励、方差、其他指标差异和结论可信度。交付分析及支撑曲线。 |

手册建议配置，作为理解实验规模的参照：

| 参数 | 默认值 |
|---|---|
| 训练／验证样本数 | 6400／1024 |
| rollout steps | 200 |
| learning rate | 1e-5 |
| rollout batch／train batch | 256／256，计数单位为回答 |
| group size | 8，即每轮 32 个 prompt |
| gradient accumulation | 32 |
| sampling temperature／max tokens | 1.0／512 |
| max gradient norm | 1.0 |
| optimizer | AdamW，betas=(0.9, 0.95)，weight decay=0 |
| 验证／保存训练回答的建议频率 | 每 10／40 个 rollout batch |

规定指标：loss、梯度范数、token entropy、训练总奖励与格式奖励、验证总奖励与格式奖励、验证平均回答长度。曲线需要展示跨运行方差，不能只给平均值。手册允许使用 `train.jsonl` 训练、`test.jsonl` 验证。

### 2.4 On-policy RL 变体／§5

| 任务 ID／位置 | 分值 | 具体任务与交付物 |
|---|---:|---|
| `think_about_length_normalization`／§5.1 | 1 | 在实验前讨论逐序列长度归一化与统一常数归一化的优缺点及适用情境；交付数句讨论。 |
| `compute_group_normalized_rewards_drgrpo`／§5.1 | 0.5 | 扩展归一化函数，支持 `advantage_normalizer="none"`，同时增加 `baseline="none"`；通过对应测试。 |
| `aggregate_loss_across_microbatch_constant`／§5.1 | 0.5 | 扩展 loss 聚合，支持 `loss_normalization="constant"`；通过对应测试。 |
| `think_about_rft`／§5.2 | 2 | 对手册中的二元奖励 RFT 与 Dr. GRPO 梯度，讨论期望、方差和可能的适用情境；交付数句讨论。 |
| `derive_difficulty_reweightings`／§5.3 | 6 | 在题设的极限与常数归一化条件下，分别推导 (a) Dr. GRPO、(b) GRPO、(c) MaxRL 的 prompt 难度重加权函数；各项交付表达式与理由。 |
| `think_about_advantage_normalization`／§5.3 | 2 | 实验前比较 std、mean 与不归一化的优缺点和适用情境；交付数句讨论。 |
| `compute_group_normalized_rewards_maxrl`／§5.3 | 0.5 | 支持 `advantage_normalizer="mean"` 及稳定项；通过对应测试。 |
| `grpo_train_step_variants_on_policy`／§5.4 | 2.5 | 扩展 train step 支持所有题设 on-policy baseline、advantage normalizer 和 loss normalization 选项；避免将零优势序列送入训练模型，保持训练数学语义；通过对应测试。 |
| `grpo_experiments_variants_on_policy`／§5.4 | 10 | 固定与标准 GRPO 可比的超参数和 r1_zero，运行 GRPO_constant、Dr_GRPO、RFT、MaxRL，每种 4 个随机种子。比较平均表现、方差、调参需求与结论可信度，交付分析及曲线。手册预算：8 B200 hrs。 |

实验所需组合来自手册，不是额外方案：

| 变体 | baseline | advantage normalizer | loss normalization |
|---|---|---|---|
| 标准 GRPO | mean | std | sequence |
| GRPO_constant | mean | std | constant |
| Dr_GRPO | mean | none | constant |
| RFT | none | none | constant |
| MaxRL（本作业版本） | mean | mean | constant |

本作业的 MaxRL 采用常数 loss normalization，与原论文的 token 数归一化不同。报告需要明确所评估的是手册版本。共用 baseline 学习率时，也应说明没有对每种算法分别充分调参的限制。

### 2.5 Off-policy／§6

| 任务 ID／位置 | 分值 | 具体任务与交付物 |
|---|---:|---|
| `derive_surrogate_objectives`／§6.2 | 2 | 推导题设 pairwise importance reweighting 估计器优化的 surrogate objective；交付表达式与推导。 |
| `compute_policy_gradient_loss_off_policy`／§6.2 | 1 | 扩展 loss 支持 `noclip` 与 `grpo`，使用生成策略的 old_log_probs 和适用的 cliprange；通过 token-level off-policy 测试。 |
| `think_about_importance_reweighting`／§6.3 | 2 | 比较不重加权、clipped token-level、GSPO clipped sequence-level 的 bias/variance 与适用情境；交付数句讨论。 |
| `compute_policy_gradient_loss_off_policy_gspo`／§6.3 | 1 | 支持 `gspo` 及题设数值稳定性要求，使用 old_log_probs、cliprange；通过 GSPO 测试。 |
| `grpo_train_step_off_policy`／§6.4 | 2.5 | 将 off-policy 参数整合进 train step，支持完整接口范围；通过对应测试。 |
| `grpo_experiments_off_policy`／§6.4 | 10 | 运行 naive、noclip、clip、gspo 四种 off-policy 变体，各 4 个种子，与 on-policy 比较。除已有指标外记录 clip fraction，讨论性能、稳定性、方差、裁剪差异、调参需求与可信度；交付分析和曲线。手册预算：8 B200 hrs。 |

手册的 off-policy 设置：rollout batch=256，train batch=8，gradient accumulation=1，即每批生成数据做 32 次参数更新。默认 GRPO cliprange=0.2，GSPO cliprange=3e-4。继续使用 r1_zero；可选择前一节喜欢的 RL 变体，但需保持比较口径。

CISPO 及把 GSPO 扩展到常数归一化算法是手册提到的可选探索，不是新增必做任务。

### 2.6 自拟估计器／§7

**`try_your_own`，10 分。** 自己提出一种 policy gradient estimator，相对于至少一种已有方法只改变一个因素。说明理论依据或直觉，在相同 OLMo/GSM8K 条件下、多随机种子比较，保持其余设置可比。交付分析、支撑曲线，必要时附数学推导。这里需要你自己提出和论证方案。

## 3. 可选补充作业：16 个任务

这一部分以 Llama 3.1 8B 为训练模型、Llama 3.3 70B Instruct 为自动 judge。共享模型和 SFT 数据路径见 [design.md](design.md)。

### 3.1 四项零样本评估／§3

| 任务 ID | 分值 | 子任务与交付物 |
|---|---:|---|
| `mmlu_baseline` | 4 | (a) 解析模型输出到选项字母，无法解析返回 None，连接 parser 测试。(b) 编写生成、评估、序列化脚本。(c) 统计解析失败并展示样例。(d) 测量生成吞吐。(e) 报告指标。(f) 随机观察 10 个错误例，写 2–4 句分析。 |
| `gsm8k_baseline` | 4 | (a) 按补充作业规定解析最终数值，无法解析返回 None，连接 parser 测试。(b) 编写评估脚本。(c–f) 报告解析失败、吞吐、指标，并分析 10 个错误例。注意不能沿用主线 boxed-answer 评分口径。 |
| `alpaca_eval_baseline` | 4 | (a) 编写预测脚本，将 instruction/output/generator/dataset 保存为 JSON 数组。(b) 报告生成吞吐。(c) 用指定 Llama judge 对照 GPT-4 Turbo 参考，报告 winrate 和 length-controlled winrate。(d) 分析随机 10 个落败样例，讨论是否认同自动评判。 |
| `sst_baseline` | 4 | (a) 编写 SimpleSafetyTests 预测脚本，保存 JSONL，至少含 prompts_final/output。(b) 报告生成吞吐。(c) 使用已有 evaluate_safety.py，报告安全比例。(d) 分析随机 10 个被判不安全的例子，讨论 judge 分歧。 |

Baseline 共用 `zero_shot_system_prompt.prompt`。MMLU、GSM8K 先使用各自任务模板再套系统模板；AlpacaEval 与 SST 使用各自 instruction。MMLU/GSM8K 手册规定 greedy decoding，temperature=0、top-p=1；其他生成设置按补充 PDF 对应小节执行。

### 3.2 SFT 数据与训练／§4

| 任务 ID | 分值 | 子任务与交付物 |
|---|---:|---|
| `look_at_sft` | 4 | 随机查看正式指令训练数据中的 10 个例子，观察任务类型和 prompt/response 质量，交付 2–4 句具体分析。 |
| `data_loading` | 3 | (a) 实现题设 packed instruction-tuning Dataset，输出规定长度的 input_ids/labels；连接 `get_packed_sft_dataset` 测试。(b) 实现覆盖一整个 epoch 的 batch 迭代，连接 `run_iterate_batches` 测试。 |
| `sft_script` | 4 | 编写 Llama 3.1 8B 指令微调脚本，支持配置、梯度累积与周期日志。可改用之前的训练脚本，但禁止使用 Hugging Face Trainer。交付训练脚本。 |
| `sft` | 6 | 完成 SFT 并保存模型及 tokenizer。交付训练设置、最终 validation loss、学习曲线与序列化产物。手册预算：3 B200 hrs。 |

正式 SFT 数据为 safety-augmented UltraChat 单轮格式，每条有 prompt/response；位于共享卷的 train/test gzip 文件。手册建议一个 epoch、context length=512、每次更新 32 条序列；参考学习率 2e-5、cosine decay、3% warmup、weight decay=0.1、gradient clipping=1.0。区分建议设置与必须交付的内容。

### 3.3 SFT 后评估与红队／§5

| 任务 ID | 分值 | 子任务与交付物 |
|---|---:|---|
| `mmlu_sft` | 4 | (a) 评估脚本、吞吐及与 baseline 比较。(b) 指标与比较。(c) 分析随机 10 个错误样例和输出行为变化。 |
| `gsm8k_sft` | 4 | (a) 评估脚本、吞吐及比较。(b) 指标与比较。(c) 分析随机 10 个错误样例和行为变化。 |
| `alpaca_eval_sft` | 4 | (a) 保存 SFT 预测并比较生成吞吐。(b) 报告两种 winrate，与 baseline 比较。(c) 分析 10 个相对 GPT-4 Turbo 落败样例与 judge 分歧。 |
| `sst_sft` | 4 | (a) 保存预测并比较吞吐。(b) 报告安全比例与 baseline 比较。(c) 分析 10 个被判不安全的例子及 judge 分歧。 |
| `red_teaming` | 4 | (a) 概述手册例子之外三种潜在滥用场景，交付 1–3 句。(b) 按作业要求进行三类交互式红队评估，每类记录尝试时间、方法、结果和定性观察，交付每类 2–4 句描述。本文不提供攻击提示或具体滥用方案。 |

SFT 后评估使用训练时的 `alpaca_sft.prompt`，不继续套 baseline 系统模板。MMLU/GSM8K 先格式化任务提示再放入 Alpaca 模板。为公平比较，保持各 benchmark 的生成设置与 baseline 一致。评估题通常需要 1–2 句吞吐说明、1–2 句指标比较和 2–4 句错误分析；AlpacaEval 指标部分允许 1–3 句。

### 3.4 HH 与 DPO／§6

| 任务 ID | 分值 | 子任务与交付物 |
|---|---:|---|
| `look_at_hh` | 2 | (a) 读取并合并四个 HH 训练集，排除题设多轮对话，分离 instruction/chosen/rejected 并保留数据来源；交付加载函数。(b) 随机看 helpful、harmless 各 3 例，讨论偏好差异与是否认同标签；交付 2–4 句。 |
| `dpo_loss` | 2 | 实现 per-instance DPO loss，使用 Alpaca 模板和 response EOS，遵守训练模型所在设备的返回要求；连接 `run_compute_per_instance_dpo_loss` 并满足测试。 |
| `dpo_training` | 4 | (a) 基于 SFT checkpoint 在 HH 上训练一个 epoch，保存 validation accuracy 最佳模型，交付训练脚本与验证准确率截图。(b) AlpacaEval 两种胜率对比 SFT，1–2 句。(c) SST 对比 SFT，1–2 句。(d) GSM8K/MMLU 评估，观察 alignment tax，2–3 句。手册预算：1 B200 hr。 |

手册建议两张 GPU 分别容纳训练模型和 reference model，使用梯度累积与 RMSprop；小规模 validation split 的示例为 200 条，参考 effective batch=64、beta=0.1、learning rate=1e-6。这些是手册建议，具体实现需要你自行完成。

## 4. 按依赖组织工作

下面是学习与实验排期建议，不是额外作业要求：

1. 阅读 [design.md](design.md)，先确认模型、数据、环境与远程资源边界。
2. 完成 prompting baseline，保留指标与生成样例；并行学习手册的 baseline 理论，但无需等待完整训练结果才能做理论题。
3. 完成标准 GRPO 组件及测试连接，再做单次 train-step 验证。
4. 完成约 50 步端到端验证，确认日志与回答，再进入 4 种子正式实验。
5. 做学习率与 prompt 实验，保留一个清楚、可复用的 baseline。
6. 完成 §5 的理论、组件扩展与控制变量实验。
7. 完成 §6 的理论和 off-policy 扩展，核对数据所对应的 sampling policy，再进行正式比较。
8. 自己设计 §7 的单因素变更，比较并撰写结论。
9. 如做补充，按四项 baseline → 数据检查 → SFT → 四项评估 → HH/DPO → 四项评估推进；补充主线不依赖主作业 GRPO 的最终性能。
10. 核对每个 task 的报告与产物，再执行课程要求的测试、打包和提交。

## 5. 用交付物判断进度

- **实现题**：学生实现、adapter 连接、规定测试结果。
- **理论题**：所有子问题的表达式、推导或讨论，不能只给结论。
- **实验题**：配置、随机种子、指标、原始回答与支撑曲线。
- **比较题**：同一口径的 baseline、控制变量与方差说明。
- **训练产物题**：可加载的模型与 tokenizer，及选择 checkpoint 的依据。

手册 B200 hrs 是规划估算，不是本机或其他 GPU 上的耗时保证，也不能直接当成总费用；多种子、judge 与资源排队会影响实际安排。

## 6. 来源

- [主作业 PDF](../cs336_spring2026_assignment5_alignment.pdf)
- [可选 safety/RLHF PDF](../cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)
- [测试适配器](../tests/adapters.py)
- [项目 README](../README.md)

本清单用于导航；精确接口、公式、生成设置与提交政策以对应 PDF 和官方测试为准。

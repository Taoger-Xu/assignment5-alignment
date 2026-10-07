# CS336 作业 5（对齐）：推理强化学习

**版本：26.0.0 · CS336 课程团队 · 2026 年春季**

> 中文译文依据[原始 PDF](../cs336_spring2026_assignment5_alignment.pdf)整理，保留原章节、任务 ID、分值、交付要求和公式编号。去除分页页码，将公式重排为 LaTeX。代码、接口、模型名称、路径和英文提示词保留原文，中文说明不应替换实验使用的提示词。本文是手册翻译，不包含额外解答。

## 1 作业概述

本次作业将让你获得一些实践经验：训练语言模型，使其能够推理并解决下游任务。

### 你将实现什么

1. 零样本、少样本和思维链提示。
2. 组相对策略优化（Group Relative Policy Optimization，GRPO）：利用外部奖励提高模型表现的强化学习算法。
3. 策略梯度估计的变体，用于探索方差降低和重要性权重裁剪策略。

### 你将运行什么

1. 测量 OLMo-2-0425-1B 在 GSM8K 上的提示性能。
2. 对 OLMo-2-0425-1B 运行同策略（on-policy）GRPO，提高 GSM8K 表现。
3. 运行 RFT、Dr. GRPO、MaxRL 等 RL 变体，探索强化学习中的算法选择。
4. 运行异策略（off-policy）GRPO，加快训练并探索不同裁剪策略。

### 代码结构

作业代码和本手册均位于 [GitHub 仓库](https://github.com/stanford-cs336/assignment5-alignment)。请使用 `git clone` 克隆仓库。如果有更新，我们会通知你，你可以使用 `git pull` 获取最新版本。

1. `cs336_alignment/*`：编写作业 5 代码的位置。除下述起始代码之外，这里没有现成实现，因此你可以从零开始自行组织代码。
2. `cs336_alignment/vllm_utils.py`：运行 vLLM 服务、生成回答和同步权重的代码。
3. `cs336_alignment/drgrpo_grader.py`：对模型的数学题输出进行评分的代码。
4. `cs336_alignment/prompts/*`：为方便使用而提供的提示词文本文件。
5. `tests/*.py`：必做 GRPO 测试和可选对齐／安全补充作业测试。

`tests/test_grpo.py` 中的必做测试调用 `tests/adapters.py` 定义的钩子。你需要实现适配器，将自己的代码连接到测试。增加测试或修改测试代码有助于调试，但你的实现应当通过最初提供的测试套件。

`README.md` 包含环境配置的基本说明。

### 提交方式

向 Gradescope 提交：

- `writeup.pdf`：回答所有书面问题，请对回答进行排版。
- `code.zip`：包含你编写的全部代码。

运行 `test_and_make_submission.sh` 中的脚本生成 `code.zip`。

## 2 引言

### 2.1 背景

本课程前四次作业讨论了如何预训练基础模型。现在我们准备学习后训练：获得基础模型后，怎样把它变成能够解决下游任务的有用工具？

后训练的一个组成部分是对齐。预训练的数据混合和训练目标，使模型获得广泛的知识和行为。但要求模型解决任务时，我们希望它表现出某种特定行为，即成为有帮助且无害的助手。把通用基础模型转变为专注的聊天模型的过程称为“指令微调”或“对齐”，这些技术将在作业 5 的可选补充部分介绍。

另一个组成部分是强化学习。预训练旨在让模型获得广泛的知识，因此使用广泛的数据，例如网上文本。后训练 RL 的目标更集中：让模型在数学解题等特定任务上获得较高准确率。这种目标至少在两方面不同于预训练：(a) 可用数据不再那么多；(b) 训练目标从“覆盖”（拥有广泛知识）变为“精确”（产生准确回答）。这些区别使我们需要一种新技术，即强化学习（RL）。

在 RL 中，我们获得问题数据集和评分函数，评分函数判断回答是否正确解决问题。例如，编程任务可能要求“编写一个反转列表的 Python 函数”，评分函数包含一组测试，如 `assert f([0, 1, 2]) == [2, 1, 0]`。数学任务可能是：“四年前 Tom 的年龄是 John 的一半，John 现在二十岁，Tom 现在几岁？”评分函数解析最终答案，判断其是否等于 12。

与预训练不同，我们没有供模型模仿的回答数据集：编程示例没有给出正确程序，数学示例没有给出正确推理链。RL 直接以模型准确率为目标进行梯度更新。概括地说，它从模型采样回答，用评分函数评价，再提高正确回答的权重。

本作业从数学和实验两方面学习 RL。RL 很难，因为它既慢又不稳定；研究 RL 也很难，因为不同随机种子的运行结果方差很大，看似微小的实现细节会产生重大影响。本作业将介绍 LLM 强化学习并探索其中的挑战。

### 2.2 模型与数据集

本作业使用 OLMo-2-0425-1B 基础模型，它先在 OLMo-mix-1124 上预训练，再在 Dolmino-mix-1124 上进行中期训练（mid-training），总训练量为 4 万亿 token。OLMo-mix-1124 主要由课堂介绍过的 DCLM-Baseline [J. Li 等，2024] 构成。Dolmino-mix-1124 的数据更集中，大约 50% 是 DCLM，另外 50% 是指令遵循、数学、代码、STEM 论文与维基数据。其他模型细节见 OLMo 2 技术报告 [T. OLMo 等，2024]。学到这里，你应已具备理解各项设计决策的背景知识。

下游任务使用 GSM8K [K. Cobbe 等，2021]，仓库中位于 `data/gsm8k/train.jsonl` 与 `data/gsm8k/test.jsonl`，也可以在线获取。它包含相对简单的小学数学推理应用题。例如：

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

这个例子的问题是：Natalia 四月卖了 48 个发夹，五月卖出四月的一半，两个月共卖多少个？回答给出推理过程，最后用 `#### 72` 标出答案。

在 RL 中，模型将学习为这类问题产生推理链，提高数学解题能力。

关于模型与数据集的选择：受资源限制，我们只选取一个小规模模型—数据集组合作为 RL 试验平台。该小模型训练了非常多的 token（约为参数量的 4000 倍！），所以能力较强，能让我们在真实数据集和较小规模上观察到合理的 RL 效果。课程中此前训练的模型很遗憾还不足以解决数学题。完成作业后，如果感兴趣，可以将自己训练的模型用于更简单任务的 RL。RL 动态高度依赖模型与数据集，因此本作业观察到的结果未必能够迁移到其他组合。

### 2.3 符号

作业涉及数学，以下列出后文使用的符号。语言建模与强化学习经常用不同术语指代同一对象，所以表中同时列出两类名称。忘记符号含义时可以返回此表。

| 符号 | 语言模型术语 | 强化学习术语 | 含义 |
|---|---|---|---|
| $\rho$ | 提示分布／数据集 | 初始状态分布 | prompt／问题的分布 |
| $x$ | prompt／问题 | 初始状态 | 从 $\rho$ 采样的 prompt／问题 |
| $y$ | 回答、补全、生成、样本 | rollout、轨迹、采样动作序列 | 对 prompt $x$ 采样得到的回答 |
| $y_t$ | token | 动作 | $y$ 中生成的第 $t$ 个 token |
| $y_{<t}$ | 前缀 | — | 位置 $t$ 前的全部生成 token，即 $y_1,\ldots,y_{t-1}$ |
| $\pi_\theta$ | 模型 | 策略 | 参数为 $\theta$ 的模型，为给定 $x$ 的回答 $y$ 分配概率 $\pi_\theta(y\mid x)$ |
| $\pi_\theta(y_t\mid x,y_{<t})$ | 下一个 token 的分布 | 时刻 $t$ 的策略 | 给定 prompt 和已生成 token 后，生成 $y_t$ 的条件概率 |
| $r(y\mid x)$ | — | 奖励 | 表示采样回答是否正确的标量；本作业为 0 或 1 |
| $B$ | 每批 prompt 数 | — | 每个推理 batch 中的 prompt 数 |
| $G$ | 每个 prompt 的生成数 | group size | 每道题采样的回答数量 |
| $\operatorname{len}(y)$ 或 $L$ | 回答长度 | horizon | 回答中的生成 token 数 |
| $A^{(i,j)}$ | — | 优势 | 经过 baseline 与归一化后，分配给问题 $i$ 的第 $j$ 个回答的权重 |

## 3 提示（Prompting）

将预训练基础模型用于下游任务的第一步是提示。基础模型在预训练中学习了广泛行为；提示是引导其进入解决任务模式的轻量方法。后文还会看到提示选择如何影响 RL 的动态与探索。

最基本的方法是直接提供问题，从模型的下一个 token 分布采样，生成回答；我们称为 `question_only`。它将与 `r1_zero` 比较，后者同时包含问题和要求模型进行思维链推理的指令 [DeepSeek-AI 等，2025]。

### 3.1 使用 vLLM 推理

生成回答需要推理引擎。实现推理引擎超出本作业范围，因此使用 vLLM [W. Kwon 等，2023]。它实现了快速 CUDA kernel、用于高效 attention KV 缓存的 PagedAttention 等优化。启动服务与生成的代码位于 `cs336_alignment/vllm_utils.py`，接口如下（保留手册原接口）：

```python
@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None

@dataclass
class VLLMServer:
    model_id: str
    gpu: int = 0
    seed: int = 0
    gpu_memory_utilization: float = 0.9

    def start(self) -> None: ...

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]: ...
```

### 3.2 零样本、少样本与思维链提示

除非另有说明，GSM8K 实验使用来自 DeepSeek R1-Zero [DeepSeek-AI 等，2025] 的下列提示，称为 `r1_zero`：

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves
it. The Assistant first thinks about the reasoning process in the mind and then provides the
User with the answer. The reasoning process is enclosed within <think> </think> and the
answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning
process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

提示位于 `cs336_alignment/prompts/r1_zero.prompt`。它描述用户提问、助手先思考再作答的对话，分别用 `<think>` 和 `<answer>` 标签包裹推理与答案。

`question` 是插入的问题，例如前面的发夹题。我们期望模型扮演助手，从推理过程开始生成（开头的 `<think>` 已由提示提供），用 `</think>` 结束推理，然后在 answer 标签内给出最终符号答案，如 `<answer> 4x + 10 </answer>`。这些标签便于解析输出、与真实答案比较，也便于在遇到 `</answer>` 时停止生成。

另一种方法是少样本提示：在实际问题前添加几个问答对。`r1_zero` 的少样本版本如下：

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves
it. The Assistant first thinks about the reasoning process in the mind and then provides the
User with the answer. The reasoning process is enclosed within <think> </think> and the
answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning
process here </think> <answer> answer here </answer>.
User: {question-1}
Assistant: <think> {reasoning-1} </think> <answer> {answer-1} </answer>
User: {question-2}
Assistant: <think> {reasoning-2} </think> <answer> {answer-2} </answer>
User: {question-3}
Assistant: <think> {reasoning-3} </think> <answer> {answer-3} </answer>
User: {question}
Assistant: <think>
```

少样本提示通过提供待解决任务的示例提高模型表现。开放基础模型公布的 benchmark 指标通常使用少样本设置，例如 OLMo-2-0425-1B 模型卡上的 GSM8K 结果使用 8-shot。我们提供了 3-shot 版本 `cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt`，示例来自 OLMES 仓库。

最后，作为基线，还会使用 `cs336_alignment/prompts/question_only.prompt`：

```text
{question} Please put your final answer within \\boxed{{}}.
```

虽然叫作 `question_only`，它仍要求把最终答案放在 boxed 格式中，以便评分器解析。下一节讨论这一点。

### 3.3 评分函数

模型生成回答后，需要判断是否正确。数学题提供真实答案，如 0.5，但模型可以用多种正确形式回答，例如 `<answer> 1/2 </answer>` 或 `The answer is 0.5`。因此需要答案解析函数，输入模型输出与已知真实答案，返回表示正确与否的布尔值。

实验采用近期推理 RL 工作 [Z. Liu 等，2025] 中使用的快速、较准确解析器。`r1_zero` 提示对应 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`。`question_only` 不要求 `<think>` 和 `<answer>` 标签，因此应使用同一文件中的 `cs336_alignment.drgrpo_grader.question_only_reward_fn`；后者在 `\boxed{}` 中寻找最终答案。

这些函数返回总奖励，以及独立的格式奖励与答案奖励，分别表示输出是否符合格式、解析后的答案是否正确。有些研究对格式正确但答案错误的回答给予部分分数，即非零总奖励。本实验不提供部分分数，总奖励就是答案奖励；格式奖励只用于日志。

注意，`ground_truth` 参数应只包含答案。GSM8K 的真实回答格式是 `{rationale} #### {answer}`，因此应以 `####` 分割并去掉首尾空白，提取最终答案。

### 3.4 实验

现在可以评估基础模型的提示表现。

**生成超参数。** 使用 temperature=1.0、top-p=1.0、最大生成长度 512。`r1_zero` 要求用 `</answer>` 结束回答，因此可以让 vLLM 遇到该字符串时停止：

```python
# Based on Dr. GRPO: stop when the model completes its answer
# https://github.com/sail-sg/understand-r1-zero/blob/
#   c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params['stop'] = ["</answer>"]
sampling_params['include_stop_str_in_output'] = True
```

此停止字符串只用于 `r1_zero` 与 `r1_zero_three_shot`，不要用于 `question_only`。

#### 任务（prompting_baselines）：在 GSM8K 上运行 OLMo-2-0425-1B（5 分）

**(a)** 编写脚本，分别用零样本 `question_only`、零样本 `r1_zero`、少样本 `r1_zero_three_shot` 评估模型在 GSM8K 上的表现。

运行脚本并观察输出。对每种提示，分别有多少生成属于：(1) 格式奖励与正确性奖励均为 1；(2) 格式奖励为 1、正确性奖励为 0；(3) 两者均为 0？至少观察第二类的十个例子，有多少其实正确但没有被正确解析？第三类又如何？

**交付物：** 数句评论、评估指标，以及一些 prompt 和回答示例。

**(b)** 根据输出描述每种提示下的模型行为。例如，如果希望模型回答问题，仅提供问题是否足够？模型还会表现出哪些回答以外的行为？零样本 `r1_zero` 和少样本 `r1_zero_three_shot` 如何塑造其行为？

**交付物：** 数句评论与支撑示例。

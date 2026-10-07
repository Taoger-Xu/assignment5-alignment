# RL 算法变体：Policy Gradient Estimator 分析

本节统一说明原始序列级 policy gradient，以及标准 GRPO、GRPO constant、Dr. GRPO、RFT、MaxRL 和 off-policy 方法分别修改了什么。

## 统一符号

- **B**：rollout batch 中的 prompt 数量；
- **G**：每个 prompt 生成的 response 数量；
- **xᵢ**：第 i 个 prompt；
- **yᵢⱼ**：第 i 个 prompt 生成的第 j 个 response；
- **rᵢⱼ**：response 的 reward；
- **Lᵢⱼ**：response 的 token 数；
- **sᵢⱼₜ**：第 t 个 token 的 score-function gradient：

$$
s_{i,j,t}=\nabla_\theta\log\pi_\theta(y_{i,j,t}\mid x_i,y_{i,j,<t})
$$

## 原始序列级 policy gradient

把整条 response 视为一个 action，原始 estimator 为：

$$
\hat g_{\text{PG}}=
\frac{1}{BG}\sum_{i=1}^{B}\sum_{j=1}^{G}
r_{i,j}\sum_{t=1}^{L_{i,j}}s_{i,j,t}
$$

它没有 response length normalization，也没有 reward 标准差归一化。因为梯度是对所有 token 求和，较长 response 自然贡献更多总梯度。

可以加入不依赖当前 action 的 baseline：

$$
\hat g_{\text{baseline}}=
\frac{1}{BG}\sum_{i,j}(r_{i,j}-b_i)
\sum_t s_{i,j,t}
$$

baseline 理论上不改变梯度期望，主要用于降低方差。

例如，同一道题采样两个回答，reward 为 `[1, 0]`。不使用 baseline 时，advantage 为 `[1, 0]`；使用组内平均 reward `b=0.5` 后，advantage 变为 `[0.5, -0.5]`。前者只提高正确回答，梯度幅度容易随采样结果大幅变化；后者同时提高正确回答、降低错误回答，并把权重缩小，因此不同 batch 的梯度估计通常更稳定。baseline 改变的是每次更新的波动，不改变长期平均的梯度方向。

## 标准 GRPO

先计算组内均值：

$$
\mu_i=\frac{1}{G}\sum_{j=1}^{G}r_{i,j}
$$

再计算标准化 advantage：

$$
A_{i,j}=\frac{r_{i,j}-\mu_i}{\sigma_i+\epsilon}
$$

其中 **σᵢ** 是该组 reward 的标准差。

std normalization 作用在中心化后的 advantage 上，而不是直接放大所有 reward。例如：

```text
Group A: reward=[0.6, 0.4]，centered=[+0.1, -0.1]，std=0.1，normalized=[+1, -1]
Group B: reward=[1.0, 0.0]，centered=[+0.5, -0.5]，std=0.5，normalized=[+1, -1]
```

当分子和组内 std 一起变化时，std normalization 会把不同 group 的 advantage 调整到相近尺度。相同的 centered advantage 在较小 std 下会被放大，在较大 std 下会被压缩；因此它会放大相对于组内波动较大的差异，并压缩相对于组内波动较小的差异，从而减少不同 group 之间梯度尺度的差异。

标准 GRPO 对每条 response 的 token loss 求平均：

$$
\hat g_{\text{GRPO}}=
\frac{1}{BG}\sum_{i,j}
\frac{A_{i,j}}{L_{i,j}}
\sum_{t=1}^{L_{i,j}}s_{i,j,t}
$$

相对于原始 estimator，它修改了三项：

1. 用组内中心化 reward 替代原始 reward；
2. 除以组内 reward 标准差；
3. 每条 response 再除以自身长度。

因此长回答不会因为 token 更多而自动得到更大权重。

## GRPO constant

GRPO constant 保留标准 GRPO 的 advantage，但取消逐 response 的长度归一化：

$$
\hat g_{\text{GRPO-constant}}=
\frac{1}{Z}\sum_{i,j}A_{i,j}
\sum_{t=1}^{L_{i,j}}s_{i,j,t}
$$

作业中通常设置：

$$
Z=BGL_{\max},\qquad L_{\max}=512
$$

与标准 GRPO 相比，较长 response 会因为包含更多 token 而贡献更大的总梯度。

## Dr. GRPO

Dr. GRPO 同时取消标准差归一化和逐 response 长度归一化：

$$
A_{i,j}=r_{i,j}-\mu_i
$$

$$
\hat g_{\text{Dr.GRPO}}=
\frac{1}{Z}\sum_{i,j}(r_{i,j}-\mu_i)
\sum_{t=1}^{L_{i,j}}s_{i,j,t}
$$

相对于原始 estimator，它保留组内 mean baseline，但不再使用 std normalization。它更接近 baseline policy gradient，但梯度方差可能更大，且长 response 的影响更强。

梯度方差变大的原因包括：

- 不同 group 的 reward 波动不会再被 std normalization 缩放；
- 长 response 包含更多 token，可能贡献更大的总梯度；
- 某个 batch 中的长回答或高 reward 回答可能主导参数更新；
- gradient norm 可能出现更明显的波动。

因此 Dr. GRPO 的核心取舍是：

```text
减少人为归一化 → 更接近原始 policy gradient
但同时       → 梯度尺度和方差可能更大
```

## RFT

RFT 不使用组内 baseline，也不进行 advantage normalization：

$$
A_{i,j}=r_{i,j}
$$

$$
\hat g_{\text{RFT}}=
\frac{1}{Z}\sum_{i,j}r_{i,j}
\sum_{t=1}^{L_{i,j}}s_{i,j,t}
$$

对于二元 reward，错误回答的 reward 为 0，因此错误回答不产生训练项。相对于 Dr. GRPO，RFT 的主要修改是移除 mean baseline。它只强化正确回答，不显式降低错误回答的概率；如果一个 batch 没有正确回答，梯度就可能为 0。

## MaxRL

本作业的 MaxRL 使用组内均值进行 advantage normalization：

$$
A_{i,j}=\frac{r_{i,j}-\mu_i}{\mu_i+\epsilon}
$$

并使用 constant loss normalization：

$$
\hat g_{\text{MaxRL}}=
\frac{1}{Z}\sum_{i,j}
\frac{r_{i,j}-\mu_i}{\mu_i+\epsilon}
\sum_t s_{i,j,t}
$$

相对于 Dr. GRPO，MaxRL 只将 advantage normalization 从 `none` 改为 `mean`。

当某个 group 的成功率较低时，其组内平均 reward **μᵢ** 也较小。由于 MaxRL 使用 **μᵢ + ε** 作为归一化分母，这类 group 会得到更大的 advantage 权重，因此模型会更加关注困难的 prompt。

但当 **μᵢ** 接近 0 时，归一化结果可能变得很大，从而导致更高的梯度方差和数值不稳定。

## On-policy 变体对比

| 算法 | Advantage | Response length normalization | 最终 normalization |
|---|---|---:|---:|
| 原始 sequence PG | `rᵢⱼ` 或 `rᵢⱼ − bᵢ` | 否 | `BG` |
| 标准 GRPO | `(rᵢⱼ − μᵢ) / (σᵢ + ε)` | 是 | `sequence` |
| GRPO constant | `(rᵢⱼ − μᵢ) / (σᵢ + ε)` | 否 | `Z = B G Lₘₐₓ` |
| Dr. GRPO | `rᵢⱼ − μᵢ` | 否 | `Z = B G Lₘₐₓ` |
| RFT | `rᵢⱼ` | 否 | `Z = B G Lₘₐₓ` |
| MaxRL | `(rᵢⱼ − μᵢ) / (μᵢ + ε)` | 否 | `Z = B G Lₘₐₓ` |

## Off-policy estimator

当 response 由旧策略 (pi_{old}) 生成，而当前策略为 (pi_\theta) 时，需要重要性比率：

$$
\rho_{i,j,t}=\frac{\pi_\theta(y_{i,j,t}\mid x_i,y_{i,j,<t})}
{\pi_{old}(y_{i,j,t}\mid x_i,y_{i,j,<t})}
$$

naive off-policy 忽略该比率，因此会产生偏差。Noclip 使用完整比率：

$$
\hat g_{\text{noclip}}=\sum_{i,j,t}
\rho_{i,j,t}A_{i,j}s_{i,j,t}
$$

它更接近目标 estimator，但 ratio 过大时方差可能爆炸。

GRPO clipping 使用：

$$
\operatorname{clip}(\rho_{i,j,t},1-\epsilon,1+\epsilon)
$$

它限制策略变化，降低方差，但会引入 clipping bias。GSPO 则先把一条 response 内的 token ratio 聚合成 sequence-level ratio，再进行 clipping，减少 token-level clipping 噪声，但会改变序列权重。

## 统一理解

所有 estimator 都可以写成：

$$
\hat g=\sum_{i,j,t}w_{i,j,t}s_{i,j,t}
$$

不同算法的差别只在权重 (w_{i,j,t})：

- reward 或 advantage；
- baseline；
- std 或 mean normalization；
- response length normalization；
- 固定常数 normalization；
- importance ratio；
- clipping。

因此实验中应同时观察最终 reward 和以下诊断指标：

- gradient norm；
- token entropy；
- response length；
- advantage scale；
- clip fraction；
- 不同随机种子的方差。

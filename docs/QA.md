# RL Algorithm Variants

## `think_about_length_normalization`

### 题目翻译

在运行任何实验之前，思考以下两种做法的区别：

1. 按每条 response 的序列长度进行归一化；
2. 对所有 response 使用同一个固定常数进行归一化。

讨论两种方法各自的优点和缺点，并说明在哪些设置或例子中其中一种方法可能更合适。

**交付要求：** 用几句话进行讨论。

### 回答

设第 **i,j** 条 response 的 advantage 为 **Aᵢⱼ**，长度为 **Lᵢⱼ**。

#### 按序列长度归一化

这种方法先计算每条 response 的平均 token loss：

$$
L_{\text{sequence}}
=
\frac{1}{BG}
\sum_{i,j}
\frac{A_{i,j}}{L_{i,j}}
\sum_{t=1}^{L_{i,j}}
-\log\pi_\theta(y_{i,j,t})
$$

它使每条 response 的整体权重主要由 advantage 决定，而不是由 token 数量决定。

优点是长 response 不会仅因为 token 更多就主导梯度，训练通常更稳定，也能减少模型通过生成冗长文本获得更大权重的风险。缺点是长回答中的每个 token 会被更强地稀释；如果长回答确实包含更多有用推理，这种归一化可能损失部分学习信号。

#### 固定常数归一化

这种方法对整个 batch 使用同一个常数 **Z**：

$$
L_{\text{constant}}
=
\frac{1}{Z}
\sum_{i,j}A_{i,j}
\sum_{t=1}^{L_{i,j}}
-\log\pi_\theta(y_{i,j,t})
$$

如果两条 response 的 advantage 相同，而长度分别为 10 和 100，那么长度为 100 的 response 会贡献大约 10 倍的 token 梯度。

固定常数归一化更接近原始序列级 policy-gradient estimator，能够保留长 reasoning response 中每个 token 的自然贡献。如果任务确实需要较长推理，它可能更合适。但它也会让长回答主导 batch 梯度，造成更大的 gradient norm 波动、长度偏置，甚至诱发不必要的回答变长。

#### 适用场景和结论

当 response 长度差异很大、训练稳定性优先，或长回答可能包含重复内容时，按序列长度归一化更合适。数学推理中，如果较长回答通常代表更多有效推理，并且生成长度受到控制，固定常数归一化可能保留更多有用梯度。

因此，两种方法体现的是一个权衡：

```text
sequence normalization：更公平、更稳定，但可能压低长回答的有效信号；
constant normalization：更接近原始梯度，但长回答权重更大、方差更高。
```

实验中应同时观察 validation reward、accuracy、response length、gradient norm、token entropy 以及不同随机种子之间的方差。

## `think_about_rft`

### 题目翻译

RFT 使用固定常数归一化时，目标函数为：

$$
J_\theta
=
\frac{1}{Z}
\sum_x\sum_{j=1}^{G}
\mathbf{1}\{r(y^{(j)}\mid x)=1\}
\log\pi_\theta(y^{(j)}\mid x)
$$

其中，**x** 是 prompt，**y⁽ʲ⁾** 是从当前策略 **πθ(· | x)** 独立采样的第 **j** 个 response，**G** 是采样数量，**Z** 是固定归一化常数，**r** 是 reward 函数。RFT 的梯度为 **∇θ Jθ**。

Dr. GRPO 的 on-policy policy-gradient estimator 为：

$$
\frac{1}{Z}
\sum_x\sum_{j=1}^{G}
\left(r(y^{(j)}\mid x)-\mu\right)
\nabla_\theta\log\pi_\theta(y^{(j)}\mid x)
$$

其中：

$$
\mu=\frac{1}{G}\sum_{j=1}^{G}r(y^{(j)}\mid x)
$$

假设 reward 是二元的，比较这两个 estimator 的期望和方差，并讨论各自适用的情况。

### 回答

对于二元 reward，RFT 只保留正确回答：

$$
\hat g_{\text{RFT}}
=
\frac{1}{Z}\sum_j r_j s_j
$$

其中 (r_j\in\{0,1\})，(s_j=\nabla_\theta\log\pi_\theta(y_j\mid x))。错误回答的 reward 为 0，因此不会产生梯度。

Dr. GRPO 使用组内中心化 reward：

$$
\hat g_{\text{Dr.GRPO}}
=
\frac{1}{Z}\sum_j(r_j-\mu)s_j
$$

因此错误回答也会产生负向梯度，模型会降低它们的概率。

两者的期望并不完全相同。由于 (mu) 使用同一组样本计算，它与当前样本的 reward 相关。对于 (G) 个独立采样：

$$
\mathbb{E}[\hat g_{\text{Dr.GRPO}}]
=
\left(1-\frac{1}{G}\right)
\mathbb{E}[\hat g_{\text{RFT}}]
$$

因此，Dr. GRPO 与 RFT 的期望梯度方向相同，但有限 (G) 时会有 (1-1/G) 的缩放。随着 (G) 增大，两者的期望差异减小。

Dr. GRPO 通常具有更低的方差，因为它使用组内均值作为 baseline，并同时利用正确和错误回答的信息。例如 reward 为：

```text
[1, 0, 0, 0]
```

RFT 的权重为：

```text
[1, 0, 0, 0]
```

而 Dr. GRPO 的组均值为 `0.25`，权重为：

```text
[0.75, -0.25, -0.25, -0.25]
```

Dr. GRPO 同时提高正确回答、降低错误回答，通常比只训练正确回答的 RFT 提供更密集的更新信号。不过，使用同一组样本计算 baseline 会带来轻微的期望缩放；当 (G) 很小或 reward 很稀疏时，方差仍可能较大。

RFT 更适合只强化高质量回答、且正确回答已经足够常见的情况。它实现简单，但如果一个 batch 中没有正确回答，梯度就会变成 0。Dr. GRPO 更适合利用组内相对质量并同时抑制错误回答的场景，通常具有更好的样本效率，但需要承担 group baseline 带来的期望缩放和额外实现复杂度。

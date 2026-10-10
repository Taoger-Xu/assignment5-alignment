from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Callable

import torch

from cs336_alignment.data import GSM8KDataset
from cs336_alignment.prompting.templates import (
    PromptRenderer,
)
from cs336_alignment.rewards.reward_pipeline import (
    compute_rollout_rewards,
)
from cs336_alignment.rl.grpo_step import (
    grpo_train_step_standard_on_policy,
)
from cs336_alignment.models.logprobs import get_response_log_probs
from cs336_alignment.prompting.tokenization import tokenize_prompt_and_output
from cs336_alignment.rollout.sampler import (
    RolloutSampler,
    SamplingConfig,
)
from cs336_alignment.training.evaluator import (
    GSM8KEvaluator,
)
from cs336_alignment.training.metrics import (
    MetricsLogger,
)


@dataclass
class GRPOTrainerConfig:
    train_rollout_batch_size: int = 256  # 每轮训练的 response 总数，即原始 prompt 数 × group_size
    group_size: int = 8  # 每个 prompt 生成的 response 数；用于组内 reward 归一化
    num_steps: int = 200  # rollout 轮数；每轮的更新次数由 train_batch_size 决定
    gradient_accumulation_steps: int = 32  # 将训练 batch 拆成多个 microbatch 累积梯度；均分时每批 256/32=8 条
    max_grad_norm: float | None = 1.0  # 参数更新前的最大梯度范数；None 表示不裁剪

    baseline: str = "mean"  # "mean"：reward 减去组内均值；"none"：不减 baseline
    advantage_eps: float = 1e-6  # 加在归一化分母上的小常数，防止零分母
    advantage_normalizer: str = "std"  # advantage 缩放方式："std" 除以组内标准差；"mean" 除以均值；"none" 不缩放
    loss_normalization: str = "sequence"  # "sequence"：先平均每条 response 的 token loss，再平均各序列
    normalization_constant: int | None = None  # constant loss 的全局 Z

    eval_interval: int = 10  # 每多少轮 rollout 执行一次验证
    rollout_log_interval: int = 40  # 每多少轮保存训练和验证样例
    validation_limit: int | None = 1024  # 最多使用多少条验证样本；None 表示使用整个验证集

    rollout_request_batch_size: int | None = None  # 每次 HTTP 请求发送的 prompt 数；None 表示一次发送全部，不改变 rollout 总数
    train_batch_size: int | None = None  # off-policy 每次更新使用的 response 数
    importance_reweighting_method: str = "none"
    cliprange: float | None = None
    checkpoint_interval: int = 50
    checkpoint_dir: str | None = None
"""
初始化 vLLM
初始化权重同步

for step in range(num_steps):
    采样训练样本
    构造 prompt
    重复 group_size
    同步 policy 权重
    生成 rollout
    执行 GRPO train step
    记录 train metrics

    如果到验证间隔：
        执行 evaluator
        记录 val metrics
        记录 rollout 样例

关闭 vLLM
关闭 logger
"""

class GRPOTrainer:
    """组织标准 on-policy GRPO 的训练、验证和日志流程。"""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        train_dataset: GSM8KDataset,
        validation_dataset: GSM8KDataset,
        renderer: PromptRenderer,
        sampler: RolloutSampler,
        reward_fn: Callable[
            [str, str],
            dict[str, float],
        ],
        sampling_config: SamplingConfig,
        logger: MetricsLogger,
        config: GRPOTrainerConfig,
        seed: int = 0,
        run_args: dict | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.renderer = renderer
        self.sampler = sampler
        self.reward_fn = reward_fn
        self.sampling_config = sampling_config
        self.logger = logger
        self.config = config

        self.rng = random.Random(seed)
        self.start_step = 0
        self.run_args = run_args or {}
        if config.eval_interval <= 0 or config.rollout_log_interval <= 0:
            raise ValueError("验证和 rollout 日志间隔必须为正数")
        if config.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval 不能为负数")

        if (
            config.train_rollout_batch_size
            % config.group_size
            != 0
        ):
            raise ValueError(
                "train_rollout_batch_size 必须能被 "
                "group_size 整除"
            )

        self.prompt_batch_size = (
            config.train_rollout_batch_size
            // config.group_size
        )
    def _sample_training_examples(self):
        """采样一批原始 GSM8K 样本。"""
        if self.prompt_batch_size > len(
            self.train_dataset
        ):
            raise ValueError(
                "prompt_batch_size 大于训练集大小"
            )

        indices = self.rng.sample(
            range(len(self.train_dataset)),
            self.prompt_batch_size,
        )

        return [
            self.train_dataset[index]
            for index in indices
        ]

    def _prepare_repeated_batch(self, examples):
        prompts = self.renderer.render_batch(
            examples
        )

        ground_truths = [
            example.ground_truth
            for example in examples
        ]

        repeated_prompts = [
            prompt
            for prompt in prompts
            for _ in range(self.config.group_size)
        ]

        repeated_ground_truths = [
            ground_truth
            for ground_truth in ground_truths
            for _ in range(self.config.group_size)
        ]

        return (
            repeated_prompts,
            repeated_ground_truths,
        )

    def save_checkpoint(self, step: int) -> Path:
        from cs336_alignment.checkpoint import save_training_checkpoint
        root = Path(self.config.checkpoint_dir or self.logger.output_dir / "checkpoints")
        path = save_training_checkpoint(
            root / f"step_{step:06d}", self.model, self.tokenizer,
            self.optimizer, step, self.rng, asdict(self.config),
            asdict(self.sampling_config), self.run_args,
        )
        print(f"[checkpoint] {path}", flush=True)
        return path

    def load_checkpoint(self, path: str) -> None:
        from cs336_alignment.checkpoint import load_training_checkpoint
        metadata = json.loads((Path(path) / "checkpoint_config.json").read_text())
        adjustable = {"num_steps", "eval_interval", "rollout_log_interval",
                      "checkpoint_interval", "checkpoint_dir"}
        for key, value in metadata["trainer_config"].items():
            if key not in adjustable and asdict(self.config).get(key) != value:
                raise ValueError(f"恢复时训练参数不匹配: {key}")
        if metadata["sampling_config"] != json.loads(json.dumps(asdict(self.sampling_config))):
            raise ValueError("恢复时采样参数不匹配")
        self.start_step = load_training_checkpoint(path, self.model, self.optimizer, self.rng)
        if self.start_step >= self.config.num_steps:
            raise ValueError("num_steps 必须大于 checkpoint 已完成的步数")
        self.logger.truncate_after(self.start_step)

    def _log_rollouts(self, step, split, prompts, responses, ground_truths, max_examples):
        rewards = [self.reward_fn(response, truth)
                   for response, truth in zip(responses, ground_truths)]
        self.logger.log_rollouts(
            step=step, split=split, prompts=prompts, responses=responses,
            ground_truths=ground_truths, rewards=rewards, max_examples=max_examples,
        )

    def _evaluate(self, evaluator, step, log_rollouts):
        result = evaluator.evaluate(self.model)
        self.logger.log_metrics(step=step, split="val", metrics={
            key: getattr(result, key) for key in (
                "total_reward", "format_reward", "answer_reward",
                "accuracy", "avg_response_length",
            )
        })
        if log_rollouts:
            self._log_rollouts(step, "val", result.prompts, result.responses,
                               result.ground_truths, max_examples=20)

    def train(self) -> None:
        """运行完整的标准 on-policy GRPO 训练循环。"""

        evaluator = GSM8KEvaluator(
            dataset=self.validation_dataset,
            renderer=self.renderer,
            sampler=self.sampler,
            reward_fn=self.reward_fn,
            sampling_config=self.sampling_config,
            group_size=1,
            batch_size=self.config.rollout_request_batch_size,
        )

        self.sampler.start()
        try:
            self.sampler.initialize_weight_sync()

            if self.start_step == 0:
                self._evaluate(evaluator, 0, log_rollouts=True)

            for step in range(self.start_step + 1, self.config.num_steps + 1):
                # 1. 随机采样一批训练题目。
                examples = self._sample_training_examples()
                prompts = self.renderer.render_batch(examples)

                # 2. 为每个 prompt 重复 group_size 次。
                (
                    repeated_prompts,
                    repeated_ground_truths,
                ) = self._prepare_repeated_batch(examples)

                expected_rollout_size = len(prompts) * self.config.group_size

                # 3. 确保 vLLM 使用当前最新的 policy。
                self.sampler.sync_policy(self.model)

                # 4. 生成 rollout。
                rollout_responses = (
                    self.sampler.generate_texts(
                        prompts=prompts,
                        group_size=self.config.group_size,
                        sampling_config=self.sampling_config,
                        batch_size=(
                            self.config.rollout_request_batch_size
                        ),
                    )
                )

                if len(rollout_responses) != expected_rollout_size:
                    raise RuntimeError(
                        "rollout 数量错误："
                        f"期望 {expected_rollout_size}，"
                        f"实际 {len(rollout_responses)}"
                    )

                if len(repeated_prompts) != len(
                    rollout_responses
                ):
                    raise RuntimeError(
                        "repeated_prompts 与 rollout_responses "
                        "长度不一致"
                    )

                if len(repeated_ground_truths) != len(
                    rollout_responses
                ):
                    raise RuntimeError(
                        "repeated_ground_truths 与 "
                        "rollout_responses 长度不一致"
                    )
                save_rollouts = (
                    step == 1 or step == self.config.num_steps
                    or step % self.config.rollout_log_interval == 0
                )
                if save_rollouts:
                    self._log_rollouts(step, "train", repeated_prompts,
                                       rollout_responses, repeated_ground_truths,
                                       max_examples=None)
                # 5. 计算旧策略在 rollout 上的 token log-probs。
                # 6. on-policy 一次更新；off-policy 将同一 rollout 拆成多个 minibatch 更新。
                mb = self.config.train_batch_size
                if mb is None:
                    mb = (
                        len(rollout_responses) // 32
                        if self.config.importance_reweighting_method != "none"
                        else len(rollout_responses)
                    )
                if mb <= 0 or len(rollout_responses) % mb != 0:
                    raise ValueError("train_batch_size 必须为正数并整除 rollout batch")
                if mb % self.config.group_size != 0:
                    raise ValueError("train_batch_size 必须包含完整的 response group")
                update_batches = [
                    (start, start + mb)
                    for start in range(0, len(rollout_responses), mb)
                ]

                old_log_probs_chunks = [None] * len(update_batches)
                if self.config.importance_reweighting_method != "none":
                    device = next(self.model.parameters()).device
                    with torch.no_grad():
                        for n, (start, end) in enumerate(update_batches):
                            tok = tokenize_prompt_and_output(
                                repeated_prompts[start:end],
                                rollout_responses[start:end], self.tokenizer,
                            )
                            ids = tok["input_ids"].to(device)
                            old_log_probs_chunks[n] = get_response_log_probs(
                                model=self.model, input_ids=ids,
                                labels=tok["labels"].to(device),
                                return_token_entropy=False,
                                attention_mask=(ids != self.tokenizer.pad_token_id),
                            )["log_probs"].detach()

                metrics_accum = []
                for n, (start, end) in enumerate(update_batches):
                    _, step_metrics = grpo_train_step_standard_on_policy(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        optimizer=self.optimizer,
                        gradient_accumulation_steps=(
                            self.config.gradient_accumulation_steps
                        ),
                        max_grad_norm=(
                            self.config.max_grad_norm
                        ),
                        reward_fn=self.reward_fn,
                        repeated_prompts=repeated_prompts[start:end],
                        rollout_responses=rollout_responses[start:end],
                        repeated_ground_truths=repeated_ground_truths[start:end],
                        group_size=self.config.group_size,
                        baseline=self.config.baseline,
                        advantage_eps=(
                            self.config.advantage_eps
                        ),
                        advantage_normalizer=(
                            self.config.advantage_normalizer
                        ),
                        loss_normalization=(
                            self.config.loss_normalization
                        ),
                        normalization_constant=self.config.normalization_constant,
                        importance_reweighting_method=self.config.importance_reweighting_method,
                        old_log_probs=old_log_probs_chunks[n],
                        cliprange=self.config.cliprange,
                    )
                    metrics_accum.append(step_metrics)
                train_metrics = {
                    k: sum(float(m[k]) for m in metrics_accum) / len(metrics_accum)
                    for k in metrics_accum[0]
                }

                if "clip_count" in train_metrics:
                    count = sum(float(m["clip_count"]) for m in metrics_accum)
                    total = sum(float(m["clip_total"]) for m in metrics_accum)
                    train_metrics["clip_fraction"] = count / max(total, 1)

                train_metrics["optimizer_updates"] = float(len(update_batches))

                # 6. 记录训练指标。
                # grpo_train_step 返回的 metadata 中已经带有
                # train/ 和 advantage/ 前缀；这里去掉前缀，避免
                # MetricsLogger 再加 split 前缀后出现 train/train/...。
                clean_train_metrics = {}
                for key, value in train_metrics.items():
                    if key.startswith("train/"):
                        key = key.removeprefix("train/")
                    elif key.startswith("advantage/"):
                        key = key.removeprefix("advantage/")
                    clean_train_metrics[key] = value

                self.logger.log_metrics(
                    step=step,
                    split="train",
                    metrics=clean_train_metrics,
                )

                
                # Save both baseline and final examples, even if log/eval intervals differ.
                if step % self.config.eval_interval == 0 or save_rollouts:
                    self._evaluate(evaluator, step, log_rollouts=save_rollouts)
                if (self.config.checkpoint_interval > 0 and
                    (step % self.config.checkpoint_interval == 0
                     or step == self.config.num_steps)):
                    self.save_checkpoint(step)

                print(
                    f"[step {step}/{self.config.num_steps}] "
                    f"loss={float(train_metrics['loss']):.4f} "
                    f"reward={float(clean_train_metrics['mean_reward']):.4f}",
                    flush=True,
                )

        finally:
            self.sampler.close()
            self.logger.close()

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cs336_alignment.data import GSM8KDataset
from cs336_alignment.prompting.templates import (
    PromptRenderer,
)
from cs336_alignment.rollout.sampler import (
    RolloutSampler,
    SamplingConfig,
)

"""
Evaluator 负责固定验证集评估，不做反向传播
验证集样本
    ↓
PromptRenderer
    ↓
RolloutSampler
    ↓
Reward function
    ↓
统计 validation metrics
"""
@dataclass
class EvaluationResult:
    total_reward: float
    format_reward: float
    answer_reward: float
    accuracy: float
    avg_response_length: float
    prompts: list[str]
    responses: list[str]
    ground_truths: list[str]

class GSM8KEvaluator:
    """在固定 GSM8K validation subset 上进行评估。
    通常更容易解释 validation accuracy，因为每个问题只生成一个回答
    """

    def __init__(
        self,
        dataset: GSM8KDataset,
        renderer: PromptRenderer,
        sampler: RolloutSampler,
        reward_fn: Callable[
            [str, str],
            dict[str, float],
        ],
        sampling_config: SamplingConfig,
        group_size: int = 1,
        batch_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.renderer = renderer
        self.sampler = sampler
        self.reward_fn = reward_fn
        self.sampling_config = sampling_config
        self.group_size = group_size
        self.batch_size = batch_size

    def evaluate(
        self,
        policy,
    ) -> EvaluationResult:
        # 验证前使用最新 policy 权重。
        self.sampler.sync_policy(policy)

        examples = list(self.dataset)

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
            for _ in range(self.group_size)
        ]

        repeated_ground_truths = [
            ground_truth
            for ground_truth in ground_truths
            for _ in range(self.group_size)
        ]

        responses = self.sampler.generate_texts(
            prompts=repeated_prompts,
            group_size=1,
            sampling_config=self.sampling_config,
            batch_size=self.batch_size,
        )

        rewards = [
            self.reward_fn(response, ground_truth)
            for response, ground_truth in zip(
                responses,
                repeated_ground_truths,
            )
        ]

        total_rewards = [
            float(result["reward"])
            for result in rewards
        ]

        format_rewards = [
            float(result["format_reward"])
            for result in rewards
        ]

        answer_rewards = [
            float(result["answer_reward"])
            for result in rewards
        ]

        accuracy = sum(
            reward == 1.0
            for reward in answer_rewards
        ) / len(answer_rewards)

        avg_response_length = sum(
            len(response)
            for response in responses
        ) / len(responses)

        return EvaluationResult(
            total_reward=sum(total_rewards)
            / len(total_rewards),
            format_reward=sum(format_rewards)
            / len(format_rewards),
            answer_reward=sum(answer_rewards)
            / len(answer_rewards),
            accuracy=accuracy,
            avg_response_length=avg_response_length,
            prompts=repeated_prompts,
            responses=responses,
            ground_truths=repeated_ground_truths,
        )

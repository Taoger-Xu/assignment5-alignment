from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


class RolloutRewardComputer:
    """
        批量计算 rollout rewards，并生成日志统计信息。
        response → raw reward
    """

    REQUIRED_KEYS = (
        "reward",
        "format_reward",
        "answer_reward",
    )

    # compute_rollout_rewards 不应该自己实现 GSM8K 答案解析，它只负责调用传入的 reward_fn
    # rollout_responses[i]是第 i 条模型生成结果,
    # repeated_ground_truths[i]是这条 response 对应的标准答案。

    def compute(
        self,
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
    ) -> tuple[Tensor, dict[str, float]]:
        if len(rollout_responses) != len(repeated_ground_truths):
            raise ValueError(
                "rollout_responses 和 repeated_ground_truths "
                "的长度必须相同"
            )

        if len(rollout_responses) == 0:
            raise ValueError("rollout batch 不能为空")

        raw_rewards: list[float] = []
        format_rewards: list[float] = []
        answer_rewards: list[float] = []

        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
        ):
            reward_result = reward_fn(response, ground_truth)
            missing_keys = [
                key
                for key in self.REQUIRED_KEYS
                if key not in reward_result
            ]
            if missing_keys:
                raise KeyError(
                    f"reward_fn 返回值缺少字段: {missing_keys}"
                )

            raw_rewards.append(float(reward_result["reward"]))
            format_rewards.append(
                float(reward_result["format_reward"])
            )
            answer_rewards.append(
                float(reward_result["answer_reward"])
            )
        
        raw_rewards_tensor = torch.tensor(
            raw_rewards,
            dtype=torch.float32,
        )
        metadata = {
            "mean_reward": sum(raw_rewards) / len(raw_rewards),
            "mean_format_reward": (
                sum(format_rewards) / len(format_rewards)
            ),
            "mean_answer_reward": (
                sum(answer_rewards) / len(answer_rewards)
            ),
        }
        
        return raw_rewards_tensor, metadata



def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    """计算每条 rollout 的原始 reward。"""
    computer = RolloutRewardComputer()

    return computer.compute(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )
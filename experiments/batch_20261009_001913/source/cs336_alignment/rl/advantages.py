from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


class GroupAdvantageEstimator:
    """在每个 response group 内计算 GRPO advantages。
        检查 reward batch；
        按 group_size 切分 reward；
        计算组内 baseline；
        计算组内归一化；
        返回 advantage；
        返回日志 metadata。
    """

    def compute(
        self,
        raw_rewards: Tensor,
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal[
            "std", "none", "mean"
        ] = "std",
    ) -> tuple[Tensor, dict[str, float]]:
        if raw_rewards.ndim != 1:
            raise ValueError(
                "raw_rewards 必须是一维张量，"
                f"当前形状为 {tuple(raw_rewards.shape)}"
            )
        if group_size <= 0:
            raise ValueError("group_size 必须大于 0")

        if raw_rewards.numel() == 0:
            raise ValueError("raw_rewards 不能为空")

        if raw_rewards.numel() % group_size != 0:
            raise ValueError(
                "raw_rewards 的长度必须能被 group_size 整除"
            )

        if advantage_eps < 0:
            raise ValueError("advantage_eps 不能为负数")

        if baseline not in {"mean", "none"}:
            raise ValueError(
                f"不支持的 baseline: {baseline}"
            )

        if advantage_normalizer not in {
            "std",
            "none",
            "mean",
        }:
            raise ValueError(
                f"不支持的 advantage_normalizer: "
                f"{advantage_normalizer}"
            )

        num_groups = raw_rewards.numel() // group_size
        grouped_rewards = raw_rewards.reshape(
            num_groups,
            group_size,
        )

        if baseline == "mean":
            group_baselines = grouped_rewards.mean(dim=1)
            grouped_advantages = (
                grouped_rewards
                - group_baselines.unsqueeze(1)
            )
        else:
            # 防止除0
            group_baselines = torch.zeros(
                num_groups,
                device=raw_rewards.device,
                dtype=raw_rewards.dtype,
            )
            grouped_advantages = grouped_rewards
        """
        减去 baseline 后，是否继续除以 std 或 mean
        advantage =
                    (r - group_mean)
                    / (group_std + eps)
        """
        if advantage_normalizer == "std":
            # 使用样本标准差，与作业测试的 snapshot 保持一致。
            if group_size == 1:
                group_normalizers = torch.ones(
                    num_groups,
                    device=raw_rewards.device,
                    dtype=raw_rewards.dtype,
                )
            else:
                # 对于 group normalization，通常更适合使用 group 内 reward 的总体标准差
                # 即unbias=False
                group_normalizers = grouped_rewards.std(
                    dim=1,
                    unbiased=True,
                )
            """
            NaN  → 0.0
            +∞   → 0.0
            -∞   → 0.0
            """
            group_normalizers = torch.nan_to_num(
                group_normalizers,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # 防止除0
            grouped_advantages = (
                grouped_advantages
                / (group_normalizers.unsqueeze(1) + advantage_eps)
            )
        elif advantage_normalizer == "mean":
            group_normalizers = grouped_rewards.mean(dim=1)
            group_normalizers = torch.nan_to_num(
                group_normalizers,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            grouped_advantages = (
                grouped_advantages
                / (group_normalizers.unsqueeze(1) + advantage_eps)
            )

        else:
            group_normalizers = torch.ones(
                num_groups,
                device=raw_rewards.device,
                dtype=raw_rewards.dtype,
            )
        advantages = grouped_advantages.reshape_as(raw_rewards)

        metadata = {
            "mean_reward": float(raw_rewards.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "mean_group_baseline": float(
                group_baselines.mean().item()
            ),
            "mean_group_normalizer": float(
                group_normalizers.mean().item()
            ),
        }

        return advantages, metadata



def compute_group_normalized_rewards_grpo(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal[
        "std", "none", "mean"
    ] = "std",
) -> tuple[Tensor, dict[str, float]]:
    estimator = GroupAdvantageEstimator()

    return estimator.compute(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
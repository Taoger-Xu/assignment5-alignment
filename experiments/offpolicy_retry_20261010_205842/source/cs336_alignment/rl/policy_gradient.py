from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


class PolicyGradientObjective:
    """计算逐 token 的 policy-gradient loss。
    advantages:       [batch_size]
    policy_log_probs: [batch_size, sequence_length]

    loss = -advantages[:, None] * policy_log_probs
    """

    def compute(
        self,
        raw_rewards_or_advantages: Tensor,
        policy_log_probs: Tensor,
        importance_reweighting_method: Literal[
            "none", "noclip", "grpo", "gspo"
        ] = "none",
        old_log_probs: Tensor | None = None,
        cliprange: float | None = None,
        response_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if importance_reweighting_method not in {"none", "noclip", "grpo", "gspo"}:
            raise ValueError(f"不支持的 importance_reweighting_method: {importance_reweighting_method}")
        if importance_reweighting_method != "none" and old_log_probs is None:
            raise ValueError("off-policy 方法必须提供 old_log_probs")
        if importance_reweighting_method in {"grpo", "gspo"} and cliprange is None:
            raise ValueError("clipped off-policy 方法必须提供 cliprange")

        if policy_log_probs.ndim != 2:
            raise ValueError(
                "policy_log_probs 必须是二维张量，"
                f"当前形状为 {tuple(policy_log_probs.shape)}"
            )

        if raw_rewards_or_advantages.ndim == 1:
            advantages = raw_rewards_or_advantages.unsqueeze(-1)
        elif raw_rewards_or_advantages.ndim == 2:
            if raw_rewards_or_advantages.shape[-1] != 1:
                raise ValueError(
                    "二维 rewards/advantages 的第二维必须为 1"
                )
            advantages = raw_rewards_or_advantages
        else:
            raise ValueError(
                "raw_rewards_or_advantages 必须是一维或二维张量"
            )

        if advantages.shape[0] != policy_log_probs.shape[0]:
            raise ValueError(
                "rewards/advantages 的 batch size 与 "
                "policy_log_probs 不一致"
            )

        if old_log_probs is not None and old_log_probs.shape != policy_log_probs.shape:
            raise ValueError("old_log_probs 与 policy_log_probs 的形状必须一致")

        # Compute ratios in float32 and remove ignored tokens BEFORE exp.
        # Prompt/padding ratios can overflow even though their loss is masked out.
        policy_log_probs = policy_log_probs.float()
        advantages = advantages.float()
        if old_log_probs is not None:
            log_ratio = policy_log_probs - old_log_probs.float()
            if response_mask is not None:
                log_ratio = torch.where(response_mask.bool(), log_ratio, 0.0)

        if importance_reweighting_method == "none":
            per_token_objective = advantages * policy_log_probs
        elif importance_reweighting_method == "gspo":
            if response_mask is None:
                raise ValueError("GSPO 必须提供 response_mask")
            if response_mask.shape != policy_log_probs.shape:
                raise ValueError("response_mask 与 policy_log_probs 的形状必须一致")
            token_count = response_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            sequence_ratio = torch.exp(
                log_ratio.sum(dim=-1, keepdim=True)
                / token_count
            )
            clipped_ratio = torch.clamp(sequence_ratio, 1.0 - cliprange, 1.0 + cliprange)
            per_token_objective = torch.minimum(
                advantages * sequence_ratio,
                advantages * clipped_ratio,
            ).expand_as(policy_log_probs)
        else:
            ratio = torch.exp(log_ratio)
            if importance_reweighting_method == "noclip":
                per_token_objective = advantages * ratio
            elif importance_reweighting_method == "grpo":
                clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
                per_token_objective = torch.minimum(
                    advantages * ratio,
                    advantages * clipped_ratio,
                )

        # advantages: [batch_size, 1]
        # policy_log_probs: [batch_size, sequence_length]
        # 广播后得到逐 token loss。
        per_token_loss = -per_token_objective

        # detach() 会返回一个不再参与梯度计算的张量，数值保持不变
        metadata = {
            "mean_advantage": advantages.detach().mean(),
            "mean_policy_log_prob": (
                policy_log_probs.detach().mean()
            ),
            "mean_loss": per_token_loss.detach().mean(),
        }

        # Fraction of valid tokens (GRPO) or non-empty sequences (GSPO)
        # whose importance ratio lies outside the clipping interval.
        clip_count = policy_log_probs.new_zeros(())
        clip_total = policy_log_probs.new_zeros(())
        if importance_reweighting_method in {"grpo", "gspo"}:
            valid = (response_mask.bool() if response_mask is not None
                     else torch.ones_like(policy_log_probs, dtype=torch.bool))
            if importance_reweighting_method == "gspo":
                valid = valid.any(dim=-1, keepdim=True)
                clipping_ratio = sequence_ratio.detach()
            else:
                clipping_ratio = ratio.detach()
            outside = (clipping_ratio < 1.0 - cliprange) | (clipping_ratio > 1.0 + cliprange)
            clip_count = (outside & valid).sum().float()
            clip_total = valid.sum().float()
        metadata.update(
            clip_count=clip_count, clip_total=clip_total,
            clip_fraction=clip_count / clip_total.clamp_min(1),
        )
        return per_token_loss, metadata
    
class MicrobatchLossAggregator:
    """将逐 token loss 聚合成可反向传播的标量 loss。
    normalization_constant 通常用于 gradient accumulation 场景
    """

    def __call__(
        self,
        per_token_policy_gradient_loss: Tensor,
        mask: Tensor,
        loss_normalization: Literal[
            "sequence", "constant"
        ] = "sequence",
        normalization_constant: int | None = None,
    ) -> Tensor:
        return self.aggregate(
            per_token_policy_gradient_loss=per_token_policy_gradient_loss,
            mask=mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

    def aggregate(
        self,
        per_token_policy_gradient_loss: Tensor,
        mask: Tensor,
        loss_normalization: Literal[
            "sequence", "constant"
        ] = "sequence",
        normalization_constant: int | None = None,
    ) -> Tensor:
        if per_token_policy_gradient_loss.ndim != 2:
            raise ValueError(
                "per_token_policy_gradient_loss 必须是二维张量，"
                f"当前形状为 "
                f"{tuple(per_token_policy_gradient_loss.shape)}"
            )

        if mask.ndim != 2:
            raise ValueError(
                "mask 必须是二维张量，"
                f"当前形状为 {tuple(mask.shape)}"
            )

        if (
            per_token_policy_gradient_loss.shape
            != mask.shape
        ):
            raise ValueError(
                "loss 和 mask 的形状必须一致，"
                f"当前分别为 "
                f"{tuple(per_token_policy_gradient_loss.shape)} "
                f"和 {tuple(mask.shape)}"
            )

        if loss_normalization not in {
            "sequence",
            "constant",
        }:
            raise ValueError(
                f"不支持的 loss_normalization: "
                f"{loss_normalization}"
            )

        if loss_normalization == "constant":
            if normalization_constant is None:
                raise ValueError(
                    "loss_normalization='constant' 时，"
                    "必须提供 normalization_constant"
                )

            if normalization_constant <= 0:
                raise ValueError(
                    "normalization_constant 必须大于 0"
                )
        mask = mask.to(
            device=per_token_policy_gradient_loss.device,
            dtype=per_token_policy_gradient_loss.dtype,
        )

        masked_loss = (
            per_token_policy_gradient_loss * mask
        )

        if loss_normalization == "constant":
            return masked_loss.sum() / normalization_constant
        # 对每条序列单独计算平均 loss
        # 每条序列中有效 response token 的数量。
        tokens_per_sequence = mask.sum(dim=-1)

        # 防止空序列除以 0。
        safe_tokens_per_sequence = (
            tokens_per_sequence.clamp_min(1.0)
        )

        # 先对每条序列的有效 token 求平均。
        sequence_losses = (
            masked_loss.sum(dim=-1)
            / safe_tokens_per_sequence
        )

        # 再对 microbatch 中的序列求平均。
        return sequence_losses.mean()

            
def compute_policy_gradient_loss_on_policy(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    objective = PolicyGradientObjective()

    return objective.compute(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
        importance_reweighting_method=importance_reweighting_method,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        response_mask=response_mask,
    )

def aggregate_loss_across_microbatch_sequence(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    aggregator = MicrobatchLossAggregator()

    return aggregator.aggregate(
        per_token_policy_gradient_loss=(
            per_token_policy_gradient_loss
        ),
        mask=mask,
        loss_normalization=loss_normalization,
        normalization_constant=normalization_constant,
    )

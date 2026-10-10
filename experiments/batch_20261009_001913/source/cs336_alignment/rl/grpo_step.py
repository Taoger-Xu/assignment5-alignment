from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor


def grpo_train_step_standard_on_policy(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal[
        "std", "none", "mean"
    ] = "std",
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",
    normalization_constant: int | None = None,
    prune_zero_advantages: bool = True,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """执行一次标准 on-policy GRPO train step。"""

    from cs336_alignment.models.logprobs import (
        get_response_log_probs,
    )
    from cs336_alignment.prompting.tokenization import (
        tokenize_prompt_and_output,
    )
    from cs336_alignment.rewards.reward_pipeline import (
        compute_rollout_rewards,
    )
    from cs336_alignment.rl.advantages import (
        compute_group_normalized_rewards_grpo,
    )
    from cs336_alignment.rl.policy_gradient import (
        aggregate_loss_across_microbatch_sequence,
        compute_policy_gradient_loss_on_policy,
    )

    batch_size = len(rollout_responses)

    if len(repeated_prompts) != batch_size:
        raise ValueError(
            "repeated_prompts 与 rollout_responses 长度不一致"
        )

    if len(repeated_ground_truths) != batch_size:
        raise ValueError(
            "repeated_ground_truths 与 rollout_responses "
            "长度不一致"
        )

    if batch_size == 0:
        raise ValueError("rollout batch 不能为空")

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps 必须大于 0"
        )

    if gradient_accumulation_steps > batch_size:
        raise ValueError(
            "gradient_accumulation_steps 不能大于 batch size"
        )
    # 1. 计算每条 rollout 的原始 reward。
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    # 2. 在每个 group 内计算 advantage。
    advantages, advantage_metadata = (
        compute_group_normalized_rewards_grpo(
            raw_rewards=raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
    )

    # 3. 对 prompt/output 进行 tokenization。
    tokenized = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )

    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    # 利用rollout的数据在gpu上进行训练
    model_device = next(model.parameters()).device

    input_ids = input_ids.to(model_device)
    labels = labels.to(model_device)
    response_mask = response_mask.to(model_device)
    advantages = advantages.to(model_device)

    # rollout负责采集
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(model_device)

    # 4. 清除上一次 train step 的梯度。
    optimizer.zero_grad()

     # 使用 torch.chunk 保证 microbatch 的大小之和等于完整 batch。
     # 防止显存溢出
    original_chunks = torch.chunk(torch.arange(batch_size), gradient_accumulation_steps)
    microbatch_size = original_chunks[0].numel()
    active_indices = (
        torch.nonzero(advantages.detach().cpu() != 0, as_tuple=True)[0]
        if prune_zero_advantages else torch.arange(batch_size)
    )
    index_chunks = active_indices.split(microbatch_size) if active_indices.numel() else ()
    # Preserve Adam momentum/step semantics even when the entire batch has zero advantage.
    if not active_indices.numel():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)

    advantage_chunks = [advantages[idx] for idx in index_chunks]

    if old_log_probs is not None:
        old_log_prob_chunks = [old_log_probs[idx] for idx in index_chunks]
    else:
        old_log_prob_chunks = [None] * len(index_chunks)

    total_loss = torch.zeros(
        (),
        device=model_device,
        dtype=torch.float32,
    )

    total_entropy = torch.zeros(
        (),
        device=model_device,
        dtype=torch.float32,
    )

    total_response_tokens = torch.zeros(
        (),
        device=model_device,
        dtype=torch.float32,
    )

    total_clip_count = torch.zeros((), device=model_device)
    total_clip_total = torch.zeros((), device=model_device)
    processed_sequences = 0

    # microbatch训练
    for idx, advantages_mb, old_log_probs_mb in zip(index_chunks, advantage_chunks, old_log_prob_chunks):
        prompts_mb = [repeated_prompts[int(i)] for i in idx]
        responses_mb = [rollout_responses[int(i)] for i in idx]
        tokenized_mb = tokenize_prompt_and_output(prompts_mb, responses_mb, tokenizer)
        input_ids_mb = tokenized_mb["input_ids"].to(model_device)
        labels_mb = tokenized_mb["labels"].to(model_device)
        response_mask_mb = tokenized_mb["response_mask"].to(model_device)
        microbatch_size = input_ids_mb.shape[0]

        # 5. 计算当前之前rollout的 token log probabilities 和 entropy。
        log_prob_output = get_response_log_probs(
            model=model,
            input_ids=input_ids_mb,
            labels=labels_mb,
            return_token_entropy=True,
            attention_mask=(input_ids_mb != tokenizer.pad_token_id),
        )
        policy_log_probs_mb = log_prob_output["log_probs"]
        token_entropy_mb = log_prob_output["token_entropy"]

        # old_log_probs 可能按完整 prompt+response 序列提供，而
        # get_response_log_probs 只返回 response token；对齐到 response 尾部。
        if old_log_probs_mb is not None and old_log_probs_mb.shape[1] != policy_log_probs_mb.shape[1]:
            if old_log_probs_mb.shape[1] < policy_log_probs_mb.shape[1]:
                raise ValueError("old_log_probs 的 token 长度不足以覆盖 response")
            old_log_probs_mb = old_log_probs_mb[:, :policy_log_probs_mb.shape[1]]

         # 6. 计算逐 token policy-gradient loss。
        per_token_loss_mb, objective_metadata = (
            compute_policy_gradient_loss_on_policy(
                raw_rewards_or_advantages=advantages_mb,
                policy_log_probs=policy_log_probs_mb,
                importance_reweighting_method=(
                    importance_reweighting_method
                ),
                old_log_probs=old_log_probs_mb,
                cliprange=cliprange,
                response_mask=response_mask_mb,
            )
        )

        total_clip_count += objective_metadata["clip_count"].detach()
        total_clip_total += objective_metadata["clip_total"].detach()

        # 7. 聚合当前 microbatch 的 token loss。
        microbatch_loss = (
            aggregate_loss_across_microbatch_sequence(
                per_token_policy_gradient_loss=(
                    per_token_loss_mb
                ),
                mask=response_mask_mb,
                loss_normalization=loss_normalization,
                normalization_constant=(
                    normalization_constant
                ),
            )
        )

        if loss_normalization == "sequence":
            # sequence normalization 时，每个 microbatch 的平均
            # loss 需要按照它包含的序列数重新加权。
            loss_weight = microbatch_size / batch_size
        else:
            # constant normalization 时，聚合函数已经使用了
            # 全局 normalization constant，不再按序列数缩放。
            loss_weight = 1.0

        weighted_microbatch_loss = (
            microbatch_loss * loss_weight
        )

         # 8. 累积梯度。
        weighted_microbatch_loss.backward()

        total_loss = total_loss + (
            weighted_microbatch_loss.detach()
        )

        # 9. 只统计 response token 的 entropy。
        entropy_masked = (
            token_entropy_mb * response_mask_mb.to(
                token_entropy_mb.dtype
            )
        )

        total_entropy = total_entropy + (
            entropy_masked.detach().sum()
        )

        total_response_tokens = total_response_tokens + (
            response_mask_mb.detach().sum()
        )

        processed_sequences += microbatch_size

    # 10. 在 optimizer.step() 之前裁剪完整 batch 的累积梯度。
    if max_grad_norm is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    else:
        squared_norm = torch.zeros(
            (),
            device=model_device,
            dtype=torch.float32,
        )

        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + (
                    parameter.grad.detach()
                    .float()
                    .pow(2)
                    .sum()
                )

        gradient_norm = squared_norm.sqrt()


     # 11. 累积梯度完成后，只更新一次参数。
    optimizer.step()

    # 12. 清理梯度。
    optimizer.zero_grad()

    if total_response_tokens.item() > 0:
        mean_token_entropy = (
            total_entropy / total_response_tokens
        )
    else:
        mean_token_entropy = torch.zeros(
            (),
            device=model_device,
            dtype=torch.float32,
        )

    metadata: dict[str, Tensor | float] = {
        "clip_count": total_clip_count.detach(),
        "clip_total": total_clip_total.detach(),
        "clip_fraction": (total_clip_count / total_clip_total.clamp_min(1)).detach(),
        "loss": total_loss.detach(),
        "gradient_norm": (
            gradient_norm.detach()
            if isinstance(gradient_norm, Tensor)
            else float(gradient_norm)
        ),
        "token_entropy": mean_token_entropy.detach(),
        "num_response_tokens": (
            total_response_tokens.detach()
        ),
        "num_sequences": float(batch_size),
        "num_active_sequences": float(processed_sequences),
        "entropy_num_sequences": float(processed_sequences),
    }

    for key, value in reward_metadata.items():
        metadata[f"train/{key}"] = value

    for key, value in advantage_metadata.items():
        metadata[f"advantage/{key}"] = value

    return total_loss.detach(), metadata

from __future__ import annotations

import torch
from torch import Tensor


class ResponseLogProbCalculator:
    """计算 causal language model 每个位置的 token log probability 和 entropy。"""

    """
    其中：labels[:, t]表示模型在看到：input_ids[:, :t+1]之后，需要预测的目标 token。
    """
    def __call__(
        self,
        model: torch.nn.Module,
        input_ids: Tensor,
        labels: Tensor,
        return_token_entropy: bool = False,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.compute(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=return_token_entropy,
            attention_mask=attention_mask,
        )

    def compute(
        self,
        model: torch.nn.Module,
        input_ids: Tensor,
        labels: Tensor,
        return_token_entropy: bool = False,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids 必须是二维张量，当前形状为 {input_ids.shape}"
            )

        if labels.ndim != 2:
            raise ValueError(
                f"labels 必须是二维张量，当前形状为 {labels.shape}"
            )

        if input_ids.shape != labels.shape:
            raise ValueError(
                "input_ids 和 labels 的形状必须相同，"
                f"当前分别为 {input_ids.shape} 和 {labels.shape}"
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                f"input_ids 必须使用 torch.long，当前为 {input_ids.dtype}"
            )

        if labels.dtype != torch.long:
            raise TypeError(
                f"labels 必须使用 torch.long，当前为 {labels.dtype}"
            )
        # 因为这个任务是给已经生成好的 token 序列打分，不是让模型继续生成文本。
        # model(input_ids=input_ids) 做的是一次 forward
        model_kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask 与 input_ids 的形状必须相同")
            model_kwargs["attention_mask"] = attention_mask
        outputs = model(**model_kwargs)

        # logits.shape = [batch_size, sequence_length, vocab_size]
        logits = outputs.logits

        if logits.ndim != 3:
            raise ValueError(
                f"模型 logits 必须是三维张量，当前形状为 {logits.shape}"
            )

        batch_size, sequence_length = input_ids.shape

        if logits.shape[0] != batch_size:
            raise ValueError("logits 的 batch size 与 input_ids 不一致")

        if logits.shape[1] != sequence_length:
            raise ValueError("logits 的序列长度与 input_ids 不一致")

        # logits: [batch_size, sequence_length, vocab_size]
        log_probs_all = torch.log_softmax(logits, dim=-1)

        vocab_size = log_probs_all.shape[-1]

        if torch.any(labels < 0) or torch.any(labels >= vocab_size):
            raise ValueError("labels 中存在超出词表范围的 token id")
        # 将 labels 扩展为 [batch_size, sequence_length, 1]，
        # 以便在词表维度上取出目标 token 的 log probability。
        target_indices = labels.unsqueeze(-1)

        # selected_log_probs: [batch_size, sequence_length, 1]
        selected_log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=target_indices,
        )

         # log_probs: [batch_size, sequence_length]
        log_probs = selected_log_probs.squeeze(-1)

        result: dict[str, Tensor] = {
            "log_probs": log_probs,
        }

        # H(p) = -Σ_v p(v) log p(v),entropy 是模型在每个位置对整个词表分布的不确定性
        if return_token_entropy:
            probabilities = log_probs_all.exp()
            token_entropy = -torch.sum(
                probabilities * log_probs_all,
                dim=-1
            )
            result["token_entropy"] = token_entropy
        return result



# 它本身通常需要返回整个序列每个位置的 log probability，后续通过response_mask只计算生成的作为loss
def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
    attention_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """计算每个目标 token 的条件 log probability，可选计算 entropy。"""
    calculator = ResponseLogProbCalculator()

    return calculator.compute(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
        attention_mask=attention_mask,
    )
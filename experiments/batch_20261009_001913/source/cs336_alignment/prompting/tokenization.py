from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


class PromptOutputTokenizer:
    """将 prompt/output 转换为右侧 padding 的因果语言模型训练输入。

    返回的 response_mask 与 labels 对齐，仅标记 response token。
    本组件负责 tokenization，不负责模型计算或设备迁移。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.padding_side != "right":
            raise ValueError(
                "PromptOutputTokenizer 只支持右侧 padding，"
                f"当前 padding_side={tokenizer.padding_side!r}。"
            )

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id 未设置，"
                "请先为 tokenizer 配置 pad token。"
            )

        self.tokenizer = tokenizer

    def __call__(
        self,
        prompt_strs: list[str],
        output_strs: list[str],
    ) -> dict[str, torch.Tensor]:
        return self.tokenize(prompt_strs, output_strs)

    def tokenize(
        self,
        prompt_strs: list[str],
        output_strs: list[str],
    ) -> dict[str, torch.Tensor]:
        """分别编码、拼接、padding，并构造 next-token labels。"""
        if len(prompt_strs) != len(output_strs):
            raise ValueError(
                "prompt_strs 与 output_strs 的长度必须相同。"
            )

        if not prompt_strs:
            raise ValueError("输入 batch 不能为空。")
         # 必须分别编码，避免字符串拼接改变边界处的分词结果。
        prompt_ids = self._encode_batch(prompt_strs)
        output_ids = self._encode_batch(output_strs)

        # 拼接
        sequences: list[list[int]] = []
        response_masks: list[list[int]] = []

        for index, (prompt, output) in enumerate(
            zip(prompt_ids, output_ids)
        ):
            if not prompt:
                raise ValueError(
                    f"第 {index} 条 prompt 编码后为空，"
                    "无法为第一个 response token 提供上下文。"
                )

            sequences.append(prompt + output)

            # 先构造与完整 token 序列对齐的 mask。
            response_masks.append(
                [0] * len(prompt) + [1] * len(output)
            )
        # 批量 padding

        padded_ids, padded_mask = self._pad_batch(
            sequences,
            response_masks,
        )

        # Causal LM shift
        # - input_ids 去掉完整序列最后一个位置。
        # - labels 去掉完整序列第一个位置。
        # - response mask 也要按照 labels 的位置进行对齐
        # 每个 input token 的目标是其后一个 token。
        # mask 跟随 labels 切片，不能跟随 input_ids 切片。
        return {
            "input_ids": padded_ids[:, :-1].contiguous(),
            "labels": padded_ids[:, 1:].contiguous(),
            "response_mask": padded_mask[:, 1:].contiguous(),
        }


    def _encode_batch(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        """编码字符串，不自动添加 BOS/EOS，不做 padding 或截断。"""
        """
        关闭会导致 prompt/output 各自重复添加的 special tokens
        避免出现
        prompt → [BOS] prompt_tokens [EOS]
        output → [BOS] output_tokens [EOS]
        """
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        return encoded["input_ids"]
    
    def _pad_batch(
        self,
        sequences: list[list[int]],
        response_masks: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """右侧补齐完整序列，padding 对应的 response mask 为 0。"""
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id 未设置，"
                "请在初始化 tokenizer 时明确指定 padding token。"
            )

        batch_size = len(sequences)
        max_length = max(len(sequence) for sequence in sequences)

        padded_ids = torch.full(
            (batch_size, max_length),
            fill_value=pad_token_id,
            dtype=torch.long,
        )

        padded_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )

        # row = 当前样本在 batch 中的行号
        # sequence = 当前样本的 token id
        # mask = 当前样本的 response mask
        # sequence = [10, 11, 20, 21]
        # mask     = [ 0,  0,  1,  1]
        # row      = 0
        for row, (sequence, mask) in enumerate(
            zip(sequences, response_masks)
        ):
            length = len(sequence)

            padded_ids[row, :length] = torch.tensor(
                sequence,
                dtype=torch.long,
            )
            padded_mask[row, :length] = torch.tensor(
                mask,
                dtype=torch.long,
            )
        # 真实序列：       [10, 11, 20, 21]
        # response mask：  [ 0,  0,  1,  1]

        # padding 后：

        # padded_ids：     [10, 11, 20, 21, PAD, PAD]
        # padded_mask：    [ 0,  0,  1,  1,   0,   0]
        return padded_ids, padded_mask

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """提供函数接口，方便训练代码和测试 adapter 调用。"""
    processor = PromptOutputTokenizer(tokenizer)
    return processor(prompt_strs, output_strs)
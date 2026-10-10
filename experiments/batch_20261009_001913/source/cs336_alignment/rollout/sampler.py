from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cs336_alignment.vllm_utils import (
    VLLMCompletion,
    VLLMServer,
)

"""
启动 vLLM
初始化权重同步
同步 Hugging Face policy
生成 rollout
提取 response 文本
关闭 vLLM
"""

@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: tuple[str, ...] = ("</answer>",)
    include_stop_str_in_output: bool = True
    seed: int = 0

    def to_dict(self, group_size: int) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "include_stop_str_in_output": (
                self.include_stop_str_in_output
            ),
            "n": group_size,
            "seed": self.seed,
        }

class RolloutSampler:
    """封装 vLLM 服务和 Hugging Face 权重同步。"""

    def __init__(
        self,
        model_id: str,
        inference_gpu: int,
        policy_device: str,
        port: int = 8000,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.inference_gpu = inference_gpu
        self.policy_device = policy_device
        self.port = port
        self.seed = seed

        self.server = VLLMServer(
            model_id=model_id,
            gpu=inference_gpu,
            port=port,
            seed=seed,
            gpu_memory_utilization=(
                gpu_memory_utilization
            ),
        )

        self._started = False
        self._weight_sync_initialized = False

    def start(self) -> None:
        if self._started:
            return

        self.server.start()
        self._started = True

    def initialize_weight_sync(self) -> None:
        if not self._started:
            raise RuntimeError(
                "必须先调用 start() 启动 vLLM"
            )

        if self._weight_sync_initialized:
            return

        self.server.init_weight_sync(
            policy_device=self.policy_device
        )

        self._weight_sync_initialized = True

    def sync_policy(
        self,
        policy: torch.nn.Module,
    ) -> None:
        if not self._started:
            raise RuntimeError(
                "必须先调用 start() 启动 vLLM"
            )

        if not self._weight_sync_initialized:
            raise RuntimeError(
                "必须先初始化权重同步"
            )

        self.server.sync_policy_weights(policy)

    def generate(
        self,
        prompts: list[str],
        group_size: int,
        sampling_config: SamplingConfig,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]:
        if not prompts:
            raise ValueError("prompts 不能为空")

        if group_size <= 0:
            raise ValueError(
                "group_size 必须大于 0"
            )

        if not self._started:
            raise RuntimeError(
                "必须先调用 start() 启动 vLLM"
            )

        params = sampling_config.to_dict(
            group_size=group_size
        )

        return self.server.generate_completions(
            prompts=prompts,
            sampling_params=params, #每一个prompt产生group_size个回答
            batch_size=batch_size,
        )

    def generate_texts(
        self,
        prompts: list[str],
        group_size: int,
        sampling_config: SamplingConfig,
        batch_size: int | None = None,
    ) -> list[str]:
        completions = self.generate(
            prompts=prompts,
            group_size=group_size,
            sampling_config=sampling_config,
            batch_size=batch_size,
        )

        return [
            completion.text
            for completion in completions
        ]

    def close(self) -> None:
        if self._started:
            self.server.stop()
            self._started = False

    def __enter__(self) -> "RolloutSampler":
        self.start()
        self.initialize_weight_sync()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
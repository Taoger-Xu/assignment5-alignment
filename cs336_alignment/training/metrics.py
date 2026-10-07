from __future__ import annotations

import json
from pathlib import Path
from typing import Any

"""
训练指标
{
  "step": 10,
  "split": "train",
  "loss": 0.42,
  "gradient_norm": 0.91,
  "token_entropy": 2.31,
  "total_reward": 0.18,
  "format_reward": 0.25
}

验证指标
{
  "step": 10,
  "split": "val",
  "total_reward": 0.23,
  "format_reward": 0.41,
  "answer_reward": 0.23,
  "accuracy": 0.23,
  "avg_response_length": 126.4
}

rollout示例
{
  "step": 10,
  "split": "val",
  "prompt": "...",
  "response": "...",
  "ground_truth": "72",
  "reward": 1.0,
  "format_reward": 1.0,
  "answer_reward": 1.0
}
"""
class MetricsLogger:
    """记录训练指标、验证指标和 rollout 样例。"""

    def __init__(
        self,
        output_dir: str | Path,
        use_wandb: bool = False,
        wandb_run: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics_path = (
            self.output_dir / "metrics.jsonl"
        )
        self.rollouts_path = (
            self.output_dir / "rollouts.jsonl"
        )

        self.use_wandb = use_wandb
        self.wandb_run = wandb_run

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """将 Tensor、NumPy scalar 等转换为 JSON 类型。"""
        if hasattr(value, "detach"):
            value = value.detach()

        if hasattr(value, "cpu"):
            value = value.cpu()

        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass

        if isinstance(value, dict):
            return {
                str(key): MetricsLogger._to_jsonable(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                MetricsLogger._to_jsonable(item)
                for item in value
            ]

        if isinstance(value, Path):
            return str(value)

        return value

    def log_metrics(
        self,
        step: int,
        split: str,
        metrics: dict[str, Any],
    ) -> None:
        record = {
            "step": int(step),
            "split": split,
        }

        for key, value in metrics.items():
            record[str(key)] = self._to_jsonable(value)

        with self.metrics_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        if self.use_wandb and self.wandb_run is not None:
            wandb_metrics = {
                f"{split}/{key}": self._to_jsonable(value)
                for key, value in metrics.items()
            }

            self.wandb_run.log(
                wandb_metrics,
                step=step,
            )


    def log_rollouts(
        self,
        step: int,
        split: str,
        prompts: list[str],
        responses: list[str],
        ground_truths: list[str],
        rewards: list[dict[str, Any]],
        max_examples: int | None = None,
    ) -> None:
        if not (
            len(prompts)
            == len(responses)
            == len(ground_truths)
            == len(rewards)
        ):
            raise ValueError(
                "prompts、responses、ground_truths 和 rewards "
                "的长度必须一致"
            )

        limit = len(prompts)

        if max_examples is not None:
            limit = min(limit, max_examples)

        with self.rollouts_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            for index in range(limit):
                record = {
                    "step": int(step),
                    "split": split,
                    "index": index,
                    "prompt": prompts[index],
                    "response": responses[index],
                    "ground_truth": ground_truths[index],
                    "reward": rewards[index],
                }

                record = self._to_jsonable(record)

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def close(self) -> None:
        """预留给后续 wandb 或其他日志后端的清理。"""
        if self.use_wandb and self.wandb_run is not None:
            finish = getattr(
                self.wandb_run,
                "finish",
                None,
            )

            if finish is not None:
                finish()

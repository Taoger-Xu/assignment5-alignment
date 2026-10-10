from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


"""
    {
        "question": "...",
        "answer": "... #### 72"
    }
"""
@dataclass(frozen=True)
class GSM8KExample:
    """一条 GSM8K 样本。"""

    question: str
    answer: str
    ground_truth: str
    index: int


class GSM8KDataset(Dataset[GSM8KExample]):
    """PyTorch Dataset 形式的 GSM8K 数据集。"""

    def __init__(
        self,
        path: str | Path,
        limit: int | None = None,
    ) -> None:
        super().__init__()

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"GSM8K 文件不存在：{self.path}"
            )

        self.examples = self._load_examples(
            limit=limit,
        )

        if not self.examples:
            raise ValueError(
                f"GSM8K 数据集为空：{self.path}"
            )

    def _load_examples(
        self,
        limit: int | None = None,
    ) -> list[GSM8KExample]:
        examples: list[GSM8KExample] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_index, line in enumerate(file):
                if limit is not None and len(examples) >= limit:
                    break

                line = line.strip()

                if not line:
                    continue

                record: dict[str, Any] = json.loads(line)

                if "question" not in record:
                    raise KeyError(
                        f"第 {line_index} 行缺少 question 字段"
                    )

                if "answer" not in record:
                    raise KeyError(
                        f"第 {line_index} 行缺少 answer 字段"
                    )

                question = str(record["question"])
                answer = str(record["answer"])

                ground_truth = self._extract_ground_truth(
                    answer
                )

                examples.append(
                    GSM8KExample(
                        question=question,
                        answer=answer,
                        ground_truth=ground_truth,
                        index=line_index,
                    )
                )

        return examples

    @staticmethod
    def _extract_ground_truth(answer: str) -> str:
        """提取 #### 后面的最终答案。"""
        match = re.search(
            r"####\s*(.+?)\s*$",
            answer,
            flags=re.DOTALL,
        )

        if match is None:
            raise ValueError(
                "answer 中不存在有效的 #### 最终答案标记："
                f"{answer!r}"
            )

        return match.group(1).strip()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GSM8KExample:
        if index < 0:
            index += len(self.examples)

        if index < 0 or index >= len(self.examples):
            raise IndexError(
                f"数据索引越界：{index}"
            )

        return self.examples[index]
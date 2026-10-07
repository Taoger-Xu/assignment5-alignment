from __future__ import annotations

from pathlib import Path
from typing import Literal

from cs336_alignment.data.gsm8k import GSM8KExample


PromptType = Literal[
    "question_only",
    "r1_zero",
    "r1_zero_three_shot",
]

"""
prompt 模板和 tokenization
GSM8KExample.question
    ↓
选择 prompt 模板
    ↓
填充 {question}
    ↓
返回模型可接受的 prompt 字符串
"""
class PromptRenderer:
    """根据 prompt 类型，将 GSM8K 样本渲染为模型输入。"""

    def __init__(
        self,
        prompt_type: PromptType,
        prompt_dir: str | Path = (
            "cs336_alignment/prompts"
        ),
    ) -> None:
        self.prompt_type = prompt_type
        self.prompt_dir = Path(prompt_dir)

        self.template_path = self._resolve_template_path()
        self.template = self._load_template()

        if "{question}" not in self.template:
            raise ValueError(
                f"模板中缺少 '{{question}}' 占位符："
                f"{self.template_path}"
            )

    def _resolve_template_path(self) -> Path:
        if self.prompt_type == "question_only":
            filename = "question_only.prompt"
        elif self.prompt_type == "r1_zero":
            filename = "r1_zero.prompt"
        elif self.prompt_type == "r1_zero_three_shot":
            filename = (
                "r1_zero_three_shot_gsm8k.prompt"
            )
        else:
            raise ValueError(
                f"不支持的 prompt 类型：{self.prompt_type}"
            )

        path = self.prompt_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt 模板不存在：{path}"
            )

        return path

    def _load_template(self) -> str:
        return self.template_path.read_text(
            encoding="utf-8"
        ).strip()

    def render_question(self, question: str) -> str:
        if not question.strip():
            raise ValueError("question 不能为空")

        return self.template.replace(
            "{question}",
            question,
        )

    def render(self, example: GSM8KExample) -> str:
        return self.render_question(
            example.question
        )

    def render_batch(
        self,
        examples: list[GSM8KExample],
    ) -> list[str]:
        return [
            self.render(example)
            for example in examples
        ]
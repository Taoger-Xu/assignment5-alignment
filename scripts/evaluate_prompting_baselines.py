from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import VLLMServer


@dataclass(frozen=True)
class GSM8KExample:
    """One GSM8K problem, its extracted gold answer, and reference rationale."""

    question: str
    ground_truth: str
    reference_answer: str


@dataclass(frozen=True)
class PromptConfig:
    """Configuration connecting a prompt template to its reward function."""

    name: str
    template_path: str
    reward_fn: Callable
    stop: list[str] | None


class GSM8KDataset:
    """Load GSM8K JSONL examples without exposing reference answers to the model."""

    def __init__(self, path: str | Path, max_examples: int | None = None) -> None:
        self.path = Path(path)
        self.max_examples = max_examples

    def load(self) -> list[GSM8KExample]:
        examples: list[GSM8KExample] = []

        with self.path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                item = json.loads(line)
                answer_text = item["answer"]
                if "####" not in answer_text:
                    raise ValueError("GSM8K answer does not contain ####")

                # Only the text after #### is used as the grading target.
                ground_truth = answer_text.split("####", 1)[1].strip()
                examples.append(
                    GSM8KExample(
                        question=item["question"],
                        ground_truth=ground_truth,
                        reference_answer=answer_text,
                    )
                )

                if self.max_examples is not None and len(examples) >= self.max_examples:
                    break

        if not examples:
            raise ValueError(f"No GSM8K examples found in {self.path}")
        return examples


class ResultSummarizer:
    """Classify per-example rewards and compute aggregate metrics."""

    @staticmethod
    def classify(metrics: dict[str, float]) -> str:
        format_reward = metrics["format_reward"]
        answer_reward = metrics["answer_reward"]

        if format_reward == 1.0 and answer_reward == 1.0:
            return "category_1_correct_and_formatted"
        if format_reward == 1.0 and answer_reward == 0.0:
            return "category_2_formatted_but_wrong"
        if format_reward == 0.0 and answer_reward == 0.0:
            return "category_3_unformatted_and_wrong"
        return "other"

    @classmethod
    def summarize(cls, results: list[dict]) -> dict:
        if not results:
            raise ValueError("Cannot summarize an empty result list")

        category_counts = Counter(result["category"] for result in results)
        n = len(results)

        return {
            "num_examples": n,
            "category_counts": dict(category_counts),
            "mean_format_reward": sum(
                result["metrics"]["format_reward"] for result in results
            ) / n,
            "mean_answer_reward": sum(
                result["metrics"]["answer_reward"] for result in results
            ) / n,
            "mean_reward": sum(
                result["metrics"]["reward"] for result in results
            ) / n,
        }


class PromptingEvaluator:
    """Run all prompt baselines against one vLLM server."""

    def __init__(
        self,
        server: VLLMServer,
        examples: list[GSM8KExample],
        output_dir: str | Path,
        batch_size: int,
    ) -> None:
        self.server = server
        self.examples = examples
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_template(path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def _format_prompts(self, template: str) -> list[str]:
        return [
            template.replace("{question}", example.question)
            for example in self.examples
        ]

    def _sampling_params(self, config: PromptConfig) -> dict:
        params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "n": 1,
            "seed": self.server.seed,
        }
        if config.stop is not None:
            params["stop"] = config.stop
            params["include_stop_str_in_output"] = True
        return params

    def evaluate(self, config: PromptConfig) -> dict:
        template = self._load_template(config.template_path)
        prompts = self._format_prompts(template)
        completions = self.server.generate_completions(
            prompts=prompts,
            sampling_params=self._sampling_params(config),
            batch_size=self.batch_size,
        )

        if len(completions) != len(self.examples):
            raise RuntimeError(
                f"Expected {len(self.examples)} completions, "
                f"got {len(completions)}"
            )

        results = []
        for example, prompt, completion in zip(
            self.examples, prompts, completions
        ):
            metrics = config.reward_fn(
                completion.text,
                example.ground_truth,
            )
            results.append(
                {
                    "prompt_name": config.name,
                    "question": example.question,
                    "ground_truth": example.ground_truth,
                    "reference_answer": example.reference_answer,
                    "prompt": prompt,
                    "response": completion.text,
                    "finish_reason": completion.finish_reason,
                    "metrics": metrics,
                    "category": ResultSummarizer.classify(metrics),
                }
            )

        result_path = self.output_dir / f"{config.name}.jsonl"
        with result_path.open("w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

        summary = ResultSummarizer.summarize(results)
        summary.update(
            {
                "prompt_name": config.name,
                "result_path": str(result_path),
            }
        )
        return summary

    def evaluate_all(self, configs: list[PromptConfig]) -> list[dict]:
        summaries = []
        self.server.start()
        try:
            for config in configs:
                summary = self.evaluate(config)
                summaries.append(summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2))
        finally:
            self.server.stop()
        return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/jk/work/models/OLMo-2-0425-1B")
    parser.add_argument("--data-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", default="results/prompting_baselines")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def build_prompt_configs() -> list[PromptConfig]:
    return [
        PromptConfig(
            name="question_only",
            template_path="cs336_alignment/prompts/question_only.prompt",
            reward_fn=question_only_reward_fn,
            stop=None,
        ),
        PromptConfig(
            name="r1_zero",
            template_path="cs336_alignment/prompts/r1_zero.prompt",
            reward_fn=r1_zero_reward_fn,
            stop=["</answer>"],
        ),
        PromptConfig(
            name="r1_zero_three_shot",
            template_path=(
                "cs336_alignment/prompts/"
                "r1_zero_three_shot_gsm8k.prompt"
            ),
            reward_fn=r1_zero_reward_fn,
            stop=["</answer>"],
        ),
    ]


def main() -> None:
    args = parse_args()
    examples = GSM8KDataset(
        path=args.data_path,
        max_examples=args.max_examples,
    ).load()

    server = VLLMServer(
        model_id=args.model_path,
        gpu=args.gpu,
        port=args.port,
        seed=args.seed,
    )
    evaluator = PromptingEvaluator(
        server=server,
        examples=examples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    summaries = evaluator.evaluate_all(build_prompt_configs())

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

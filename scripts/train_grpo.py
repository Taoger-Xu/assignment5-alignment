from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.data import GSM8KDataset
from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.prompting.templates import PromptRenderer
from cs336_alignment.rollout.sampler import RolloutSampler, SamplingConfig
from cs336_alignment.training.metrics import MetricsLogger
from cs336_alignment.training.trainer import GRPOTrainer, GRPOTrainerConfig


def parse_args() -> argparse.Namespace:
    # 1. 命令行参数解析：模型、数据、训练超参数、GPU 和日志配置。
    parser = argparse.ArgumentParser(description="Train OLMo with on-policy GRPO")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--validation-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--train-limit", type=int, default=6400)
    parser.add_argument("--validation-limit", type=int, default=1024)
    parser.add_argument("--prompt-type", choices=["r1_zero", "question_only", "r1_zero_three_shot"], default="r1_zero")
    parser.add_argument("--output-dir", default="experiments/grpo")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--baseline", choices=["mean", "none"], default="mean")
    parser.add_argument("--advantage-normalizer", choices=["std", "none", "mean"], default="std")
    parser.add_argument("--loss-normalization", choices=["sequence", "constant"], default="sequence")
    parser.add_argument("--importance-reweighting-method", choices=["none", "noclip", "grpo", "gspo"], default="none")
    parser.add_argument("--cliprange", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--rollout-log-interval", type=int, default=40)
    parser.add_argument("--rollout-request-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--inference-gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    # 2. 随机种子设置：保证数据采样、PyTorch 和 CUDA 的随机性可复现。
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path: str, device: str):
    # 3. tokenizer 和模型加载。
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 4. 自动设置 pad_token；部分基础模型没有单独的 padding token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=device,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else "eager",
    )
    model.train()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # 5. GSM8K 训练集与验证集加载。
    train_dataset = GSM8KDataset(args.train_path, limit=args.train_limit)
    validation_dataset = GSM8KDataset(args.validation_path, limit=args.validation_limit)

    # 6. prompt 支持：r1_zero、question_only、r1_zero_three_shot。
    renderer = PromptRenderer(args.prompt_type)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.policy_device)

    # 8. AdamW 优化器。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)

    # 7. 根据 prompt 类型选择对应 reward function。
    stop = ("</answer>",) if args.prompt_type != "question_only" else ()
    reward_fn = (
        question_only_reward_fn
        if args.prompt_type == "question_only"
        else r1_zero_reward_fn
    )
    sampling_config = SamplingConfig(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        stop=stop, include_stop_str_in_output=bool(stop), seed=args.seed,
    )

    # 9. vLLM RolloutSampler：负责启动推理服务、同步 policy 权重和生成回答。
    sampler = RolloutSampler(
        model_id=args.model_path, inference_gpu=args.inference_gpu,
        policy_device=args.policy_device, port=args.port, seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 11. 可选 WandB：未传 --wandb-project 时只写本地 JSONL 日志。
    wandb_run = None
    if args.wandb_project is not None:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    # 10. MetricsLogger：记录训练/验证指标以及 rollout 样例。
    logger = MetricsLogger(Path(args.output_dir), use_wandb=wandb_run is not None, wandb_run=wandb_run)
    config = GRPOTrainerConfig(
        train_rollout_batch_size=args.rollout_batch_size, group_size=args.group_size,
        num_steps=args.num_steps, gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm, eval_interval=args.eval_interval,
        rollout_log_interval=args.rollout_log_interval, validation_limit=args.validation_limit,
        rollout_request_batch_size=args.rollout_request_batch_size,
        baseline=args.baseline, advantage_eps=args.advantage_eps,
        advantage_normalizer=args.advantage_normalizer,
        loss_normalization=args.loss_normalization,
        importance_reweighting_method=args.importance_reweighting_method,
        cliprange=args.cliprange,
        train_batch_size=args.train_batch_size,
        normalization_constant=(args.rollout_batch_size * args.max_tokens
                                 if args.loss_normalization == "constant" else None),
    )
    # 12. GRPOTrainer 初始化和启动完整训练循环。
    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, optimizer=optimizer,
        train_dataset=train_dataset, validation_dataset=validation_dataset,
        renderer=renderer, sampler=sampler, reward_fn=reward_fn,
        sampling_config=sampling_config, logger=logger, config=config, seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()

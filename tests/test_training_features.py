import copy
import random

import pytest
import torch
from transformers import AutoModelForCausalLM

from cs336_alignment.checkpoint import save_training_checkpoint, load_training_checkpoint
from cs336_alignment.rl.grpo_step import grpo_train_step_standard_on_policy
from cs336_alignment.rl.policy_gradient import compute_policy_gradient_loss_on_policy


@pytest.mark.parametrize("method,expected", [("grpo", 2 / 3), ("gspo", 1 / 2)])
def test_clip_fraction_ignores_padding(method, expected):
    ratios = torch.tensor([[1.5, 1.5, 100.0], [0.7, 100.0, 100.0], [100.0, 100.0, 100.0]])
    ratios[1, 0] = 1.0
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    _, metadata = compute_policy_gradient_loss_on_policy(
        torch.ones(3), ratios.log(), importance_reweighting_method=method,
        old_log_probs=torch.zeros_like(ratios), cliprange=0.2, response_mask=mask,
    )
    assert metadata["clip_fraction"].item() == pytest.approx(expected)


def test_checkpoint_restores_model_optimizer_and_rng(tmp_path, tiny_train_model, tokenizer):
    model = tiny_train_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.tensor([[3, 4]])).logits.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    rng = random.Random(42)
    checkpoint = save_training_checkpoint(
        tmp_path / "step_000001", model, tokenizer, optimizer, 1, rng, {}, {},
    )
    expected_weights = copy.deepcopy(model.state_dict())
    expected_random = (rng.random(), random.random(), torch.rand(3))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    assert load_training_checkpoint(checkpoint, model, optimizer, rng) == 1
    assert rng.random() == expected_random[0]
    assert random.random() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key])
    assert all(state["step"] == 1 for state in optimizer.state.values())
    exported = AutoModelForCausalLM.from_pretrained(checkpoint)
    for key, value in exported.state_dict().items():
        torch.testing.assert_close(value, expected_weights[key])


@pytest.mark.parametrize("baseline,normalization,all_zero", [
    ("none", "constant", False), ("none", "sequence", False),
    ("mean", "sequence", False), ("none", "constant", True),
])
def test_pruning_preserves_adam_update(tiny_train_model, tokenizer, baseline, normalization, all_zero):
    models = [copy.deepcopy(tiny_train_model) for _ in range(2)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in models]
    # Populate momentum so an all-zero batch must still preserve optimizer semantics.
    for model, optimizer in zip(models, optimizers):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    responses = ["world", "test", "a test", "another test"]
    active = 0 if all_zero else (1 if baseline == "none" else 2)
    losses = []
    forwarded = []
    gradients = []
    for model, optimizer, prune in zip(models, optimizers, [False, True]):
        count = []
        saved_step = optimizer.step
        def capture_step():
            gradients.append([p.grad.detach().clone() if p.grad is not None else None
                              for p in model.parameters()])
            return saved_step()
        optimizer.step = capture_step
        hook = model.register_forward_pre_hook(lambda module, args, kwargs: count.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
        loss, metadata = grpo_train_step_standard_on_policy(
            model, tokenizer, optimizer, 2, 1.0,
            lambda response, truth: {
                "reward": float(not all_zero and response == "world"),
                "format_reward": 1.0,
                "answer_reward": float(not all_zero and response == "world"),
            }, ["Hello", "Hello", "This is", "This is"], responses, ["42"] * 4,
            group_size=2, baseline=baseline, loss_normalization=normalization,
            normalization_constant=32 if normalization == "constant" else None,
            prune_zero_advantages=prune,
        )
        hook.remove()
        losses.append(loss)
        forwarded.append(sum(count))
    assert forwarded == [4, active]
    torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-5)
    for a, b in zip(gradients[0], gradients[1]):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_trainer_resume_matches_uninterrupted_and_logs_baseline(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cs336_alignment.training import trainer as module
    from cs336_alignment.training.metrics import MetricsLogger
    from cs336_alignment.rollout.sampler import SamplingConfig

    evaluation = SimpleNamespace(total_reward=1.0, format_reward=1.0, answer_reward=1.0,
                                 accuracy=1.0, avg_response_length=5.0,
                                 prompts=["Hello"], responses=["world"], ground_truths=["42"])
    monkeypatch.setattr(module, "GSM8KEvaluator", lambda **kwargs: SimpleNamespace(evaluate=lambda model: evaluation))

    def update(**kwargs):
        optimizer = kwargs["optimizer"]
        optimizer.zero_grad()
        loss = kwargs["model"].weight.sum() * float(kwargs["repeated_ground_truths"][0])
        loss.backward()
        optimizer.step()
        return loss.detach(), {"loss": float(loss.detach()), "train/mean_reward": 1.0}

    monkeypatch.setattr(module, "grpo_train_step_standard_on_policy", update)
    initial = torch.nn.Linear(1, 1, bias=False)

    def build(output, steps):
        model = copy.deepcopy(initial)
        sampler = Mock()
        sampler.generate_texts.return_value = ["world"] * 4
        examples = [SimpleNamespace(ground_truth=str(i + 1)) for i in range(4)]
        renderer = SimpleNamespace(render_batch=lambda examples: ["Hello"] * len(examples))
        return module.GRPOTrainer(
            model=model, tokenizer=None,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            train_dataset=examples, validation_dataset=examples, renderer=renderer,
            sampler=sampler, reward_fn=lambda response, truth: {
                "reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0},
            sampling_config=SamplingConfig(), logger=MetricsLogger(output),
            config=module.GRPOTrainerConfig(num_steps=steps, train_rollout_batch_size=4,
                group_size=2, gradient_accumulation_steps=1,
                checkpoint_interval=1, rollout_log_interval=1), seed=7,
        )

    full = build(tmp_path / "full", 2)
    full.train()
    first = build(tmp_path / "resumed", 1)
    first.train()
    resumed = build(tmp_path / "resumed", 2)
    resumed.load_checkpoint(str(tmp_path / "resumed/checkpoints/step_000001"))
    resumed.train()
    torch.testing.assert_close(full.model.weight, resumed.model.weight, atol=0, rtol=0)
    records = [json.loads(line) for line in resumed.logger.metrics_path.read_text().splitlines()]
    assert [(r["split"], r["step"]) for r in records] == [
        ("val", 0), ("train", 1), ("val", 1), ("train", 2), ("val", 2)]
    rollouts = [json.loads(line) for line in resumed.logger.rollouts_path.read_text().splitlines()]
    assert len([r for r in rollouts if r["split"] == "train"]) == 8
    assert [r["step"] for r in rollouts if r["split"] == "val"] == [0, 1, 2]

@pytest.mark.parametrize('method', ['noclip', 'grpo', 'gspo'])
def test_ignored_importance_ratios_cannot_overflow(method):
    logp = torch.tensor([[0.0, 1000.0]], requires_grad=True)
    loss, _ = compute_policy_gradient_loss_on_policy(
        torch.ones(1), logp, importance_reweighting_method=method,
        old_log_probs=torch.zeros_like(logp), cliprange=0.2,
        response_mask=torch.tensor([[1, 0]]),
    )
    (loss * torch.tensor([[1, 0]])).sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(logp.grad).all()
    assert logp.grad[0, 1] == 0

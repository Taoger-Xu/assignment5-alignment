from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.training import trainer as trainer_module
from cs336_alignment.training.trainer import GRPOTrainer, GRPOTrainerConfig
from cs336_alignment.rollout.sampler import SamplingConfig


@pytest.mark.parametrize("batch_size,accumulation,expected_updates", [(None, 32, 1), (8, 1, 32)])
def test_none_reweighting_respects_training_batch(monkeypatch, batch_size, accumulation, expected_updates):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    dataset = [SimpleNamespace(ground_truth=str(i)) for i in range(32)]
    renderer = Mock()
    renderer.render_batch.side_effect = lambda examples: [x.ground_truth for x in examples]
    responses = [str(i) for i in range(256)]
    sampler = Mock()
    sampler.generate_texts.return_value = responses
    evaluator = Mock()
    evaluator.evaluate.return_value = SimpleNamespace(
        total_reward=0.0, format_reward=0.0, answer_reward=0.0,
        accuracy=0.0, avg_response_length=1.0,
        prompts=[], responses=[], ground_truths=[],
    )
    monkeypatch.setattr(trainer_module, "GSM8KEvaluator", Mock(return_value=evaluator))
    old_log_probs = Mock(side_effect=AssertionError("naive must not compute old log probs"))
    monkeypatch.setattr(trainer_module, "get_response_log_probs", old_log_probs)
    observed = []

    def train_step(**kwargs):
        observed.extend(kwargs["rollout_responses"])
        assert len(kwargs["rollout_responses"]) == (batch_size or 256)
        assert kwargs["gradient_accumulation_steps"] == accumulation
        assert kwargs["old_log_probs"] is None
        assert kwargs["importance_reweighting_method"] == "none"
        for start in range(0, len(kwargs["repeated_prompts"]), 8):
            assert len(set(kwargs["repeated_prompts"][start:start + 8])) == 1
        optimizer.zero_grad()
        model.weight.sum().backward()
        optimizer.step()
        return torch.tensor(0.0), {"loss": 0.0, "train/mean_reward": 0.0}

    monkeypatch.setattr(trainer_module, "grpo_train_step_standard_on_policy", train_step)
    trainer = GRPOTrainer(
        model=model, tokenizer=None, optimizer=optimizer,
        train_dataset=dataset, validation_dataset=dataset,
        renderer=renderer, sampler=sampler, reward_fn=Mock(),
        sampling_config=SamplingConfig(), logger=Mock(),
        config=GRPOTrainerConfig(
            num_steps=1, train_batch_size=batch_size, checkpoint_interval=0,
            gradient_accumulation_steps=accumulation,
        ),
    )
    initial_weight = model.weight.detach().clone()
    trainer.train()
    assert optimizer.step.call_count == expected_updates
    assert observed == responses
    assert torch.allclose(model.weight, initial_weight - 0.1 * expected_updates, atol=1e-5)
    old_log_probs.assert_not_called()
    sampler.generate_texts.assert_called_once()
    sampler.close.assert_called_once()

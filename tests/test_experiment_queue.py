import json
from pathlib import Path

import pytest

from scripts.run_experiment_queue import Queue


def test_queue_checks_each_job_and_allocates_unique_ports(tmp_path, monkeypatch):
    commands = []

    class Process:
        pid = 12345

        def __init__(self, command, **kwargs):
            commands.append(command)
            options = dict(zip(command[3::2], command[4::2]))
            output = Path(options["--output-dir"])
            output.joinpath("metrics.jsonl").write_text(
                json.dumps({"step": 0, "split": "val", "total_reward": 0.0}) + "\n" +
                json.dumps({"step": 1, "split": "val", "total_reward": 0.5}) + "\n")
            output.joinpath("rollouts.jsonl").write_text(json.dumps({"split": "train"}) + "\n")
            checkpoint = output / "checkpoints/step_000001"
            checkpoint.mkdir(parents=True)
            checkpoint.joinpath("training_state.pt").touch()

        def poll(self):
            return 0

    monkeypatch.setattr("scripts.run_experiment_queue.subprocess.Popen", Process)
    queue = Queue(tmp_path)
    queue.run_stage("smoke", [(f"job{i}", {}) for i in range(8)], smoke=True)
    assert (tmp_path / "smoke/.complete").exists()
    assert all(job["state"] == "complete" for job in queue.status["jobs"].values())
    for start in [0, 4]:
        ports = [command[command.index("--port") + 1] for command in commands[start:start + 4]]
        assert ports == ["8500", "8501", "8502", "8503"]
    assert queue.env["HF_HUB_OFFLINE"] == "1"
    assert queue.env["WANDB_MODE"] == "disabled"


def test_failed_process_continues_next_wave(tmp_path, monkeypatch):
    launched = []

    class Failed:
        pid = 12345

        def __init__(self, command, **kwargs):
            launched.append(command)

        def poll(self):
            return 1

    monkeypatch.setattr("scripts.run_experiment_queue.subprocess.Popen", Failed)
    queue = Queue(tmp_path)
    queue.run_stage("smoke", [(f"job{i}", {}) for i in range(8)], smoke=True)
    assert len(launched) == 8
    assert (tmp_path / "smoke/.finished_with_errors").exists()
    assert not (tmp_path / "smoke/.complete").exists()
    assert all(job["state"] == "failed" for job in queue.status["jobs"].values())

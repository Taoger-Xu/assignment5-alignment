"""Offline experiment supervisor: four independent training/inference GPU pairs."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"
MODEL = Path("/home/jk/work/models/OLMo-2-0425-1B")
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
VARIANTS = {
    "GRPO_constant": {"loss-normalization": "constant"},
    "Dr_GRPO": {"loss-normalization": "constant", "advantage-normalizer": "none"},
    "RFT": {"loss-normalization": "constant", "advantage-normalizer": "none", "baseline": "none"},
    "MaxRL": {"loss-normalization": "constant", "advantage-normalizer": "mean"},
}
OFFPOLICY = {
    "offpolicy_naive": {"importance-reweighting-method": "none"},
    "offpolicy_noclip": {"importance-reweighting-method": "noclip"},
    "offpolicy_clip": {"importance-reweighting-method": "grpo", "cliprange": 0.2},
    "offpolicy_gspo": {"importance-reweighting-method": "gspo", "cliprange": 0.0003},
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


class Queue:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / "queue.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = os.environ.copy()
        self.env.update(
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            WANDB_MODE="disabled", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
            PYTHONUNBUFFERED="1", VIRTUAL_ENV=str(REPO / ".venv"),
            PATH=str(REPO / ".venv/bin") + ":" + self.env.get("PATH", ""),
            OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
            VLLM_HOST_IP="127.0.0.1", NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo",
        )
        status_path = root / "status.json"
        self.status = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
        self.status.update(state="starting", pid=os.getpid(), root=str(root))
        self.status.pop("error", None)
        self.persist()

    def persist(self):
        self.status["updated"] = datetime.now().astimezone().isoformat()
        write_json(self.root / "status.json", self.status)

    def preflight(self):
        for path in [PYTHON, MODEL / "config.json", MODEL / "tokenizer.json",
                     REPO / "data/gsm8k/train.jsonl", REPO / "data/gsm8k/test.jsonl"]:
            if not path.is_file():
                raise FileNotFoundError(path)
        index = json.loads((MODEL / "model.safetensors.index.json").read_text())
        for filename in set(index["weight_map"].values()):
            if not (MODEL / filename).is_file():
                raise FileNotFoundError(MODEL / filename)
        subprocess.run([str(PYTHON), "-c", "import torch, flash_attn, vllm, wandb; "
                        "assert torch.cuda.is_available() and torch.cuda.device_count() >= 8"],
                       cwd=REPO, env=self.env, check=True)
        write_json(self.root / "environment.json", {
            "python": str(PYTHON), "model": str(MODEL), "gpu_pairs": PAIRS,
            "offline": True,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        })
        (self.root / "source.patch").write_text(subprocess.check_output(
            ["git", "diff"], cwd=REPO, text=True))

    def validate(self, output, steps):
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        for record in records:
            for value in record.values():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise RuntimeError(f"Non-finite metric: {output}: {record}")
        final = [r for r in records if r["split"] == "val" and r["step"] == steps]
        if not final or not any(r["split"] == "val" and r["step"] == 0 for r in records):
            raise RuntimeError(f"Missing baseline/final validation: {output}")
        checkpoint = output / "checkpoints" / f"step_{steps:06d}"
        if not (checkpoint / "training_state.pt").exists():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")
        rollouts = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
        if not any(r["split"] == "train" for r in rollouts):
            raise RuntimeError(f"Missing training rollouts: {output}")
        return final[-1]

    def run_stage(self, name, jobs, smoke=False):
        self.status.update(state="running", stage=name)
        self.persist()
        print(f"[{datetime.now().isoformat()}] stage={name}, jobs={len(jobs)}", flush=True)
        for offset in range(0, len(jobs), 4):
            running = []
            for slot, (label, options) in enumerate(jobs[offset:offset + 4]):
                output = self.root / name / label
                steps = 1 if smoke else int(options.get("num-steps", 200))
                if (output / ".complete").exists():
                    self.validate(output, steps)
                    continue
                if (output / "metrics.jsonl").exists():
                    raise RuntimeError(f"Incomplete output exists; inspect before rerunning: {output}")
                output.mkdir(parents=True, exist_ok=True)
                policy, inference = PAIRS[slot]
                args = {
                    "model-path": str(MODEL), "train-path": "data/gsm8k/train.jsonl",
                    "validation-path": "data/gsm8k/test.jsonl", "train-limit": 6400,
                    "validation-limit": 8 if smoke else 1024, "num-steps": steps,
                    "rollout-batch-size": 256, "group-size": 8,
                    "gradient-accumulation-steps": 32, "prompt-type": "r1_zero",
                    "learning-rate": 1e-5, "max-tokens": 512,
                    "policy-device": f"cuda:{policy}", "inference-gpu": inference,
                    "port": 8500 + slot, "gpu-memory-utilization": 0.8,
                    "eval-interval": 10, "rollout-log-interval": 40,
                    "checkpoint-interval": 1 if smoke else 50, "seed": 0,
                    "output-dir": str(output),
                }
                args.update(options)
                command = [str(PYTHON), "-u", "scripts/train_grpo.py"]
                for key, value in args.items():
                    command.extend([f"--{key}", str(value)])
                write_json(output / "launch.json", {"command": command, "offline": True})
                log = (output / "startup.log").open("a")
                process = subprocess.Popen(command, cwd=REPO, env=self.env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                job_id = f"{name}/{label}"
                self.status["jobs"][job_id] = {
                    "state": "running", "pid": process.pid, "gpu_pair": [policy, inference],
                    "port": args["port"], "output": str(output), "started": time.time(),
                }
                running.append((job_id, process, log, output, steps))
                print(f"launch {job_id}: GPU {policy}/{inference}, port {args['port']}", flush=True)
            self.persist()
            errors = []
            while running:
                for item in list(running):
                    job_id, process, log, output, steps = item
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    running.remove(item)
                    state = self.status["jobs"][job_id]
                    state.update(exit_code=code, finished=time.time())
                    try:
                        if code != 0:
                            raise RuntimeError(f"exit code {code}; inspect {output / 'startup.log'}")
                        state["final_validation"] = self.validate(output, steps)
                        (output / ".complete").touch()
                        state["state"] = "complete"
                    except Exception as exc:
                        state.update(state="failed", error=str(exc))
                        errors.append(f"{job_id}: {exc}")
                    print(f"{job_id}: {state['state']}", flush=True)
                    self.persist()
                if running:
                    time.sleep(5)
            if errors:
                raise RuntimeError("\n".join(errors))
        (self.root / name / ".complete").touch()

    def run(self):
        self.preflight()
        smoke = [("standard", {}), ("lr_low", {"learning-rate": "5e-6"}),
                 ("lr_high", {"learning-rate": "2e-5"}),
                 ("question_only", {"prompt-type": "question_only"}),
                 ("three_shot", {"prompt-type": "r1_zero_three_shot"})]
        smoke += list(VARIANTS.items())
        smoke += [(name, dict(options, **{"train-batch-size": 8, "gradient-accumulation-steps": 1}))
                  for name, options in OFFPOLICY.items()]
        smoke += [(f"standard_seed{seed}", {"seed": seed}) for seed in [1, 2, 3]]
        self.run_stage("smoke", smoke, smoke=True)
        if not (self.root / "skip_debug").exists():
            self.run_stage("grpo_debug", [("seed0", {"num-steps": 50})])
        self.run_stage("grpo_standard", [(f"seed{seed}", {"seed": seed}) for seed in range(4)])
        self.run_stage("lr_sweep", [(f"lr_{lr}_seed{seed}", {"learning-rate": lr, "seed": seed})
                                   for lr in ["5e-6", "2e-5"] for seed in range(4)])
        rewards = {}
        for lr in ["5e-6", "1e-5", "2e-5"]:
            directories = [self.root / "grpo_standard" / f"seed{s}" if lr == "1e-5"
                           else self.root / "lr_sweep" / f"lr_{lr}_seed{s}" for s in range(4)]
            rewards[lr] = sum(self.validate(directory, 200)["total_reward"] for directory in directories) / 4
        best_lr = max(rewards, key=rewards.get)
        write_json(self.root / "learning_rate_selection.json", {"mean_final_rewards": rewards, "selected": best_lr})
        self.run_stage("prompt_ablation", [
            (f"{prompt}_seed{seed}", {"prompt-type": prompt, "seed": seed, "learning-rate": best_lr})
            for prompt in ["question_only", "r1_zero_three_shot"] for seed in range(4)])
        self.run_stage("grpo_variants", [
            (f"{method}_seed{seed}", dict(options, seed=seed, **{"learning-rate": best_lr}))
            for method, options in VARIANTS.items() for seed in range(4)])
        self.run_stage("grpo_offpolicy", [
            (f"{method}_seed{seed}", dict(options, seed=seed, **{
                "learning-rate": best_lr, "train-batch-size": 8, "gradient-accumulation-steps": 1}))
            for method, options in OFFPOLICY.items() for seed in range(4)])
        self.status.update(state="complete", stage="done")
        self.persist()
        (self.root / ".complete").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    queue = Queue(args.root.resolve())
    try:
        queue.run()
    except Exception as exc:
        queue.status.update(state="failed", error=str(exc))
        queue.persist()
        print(f"QUEUE FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()

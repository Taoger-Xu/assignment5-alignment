# Offline experiment queue

- Supervisor: `scripts/run_experiment_queue.py`.
- Progress: `status.json`; supervisor log: `queue.log`.
- Order: smoke (16 one-step checks), debug skipped (already completed), standard (4 seeds × 200), LR sweep (2 additional LRs × 4 seeds), prompt (8 runs), on-policy variants (16 runs), off-policy (16 runs).
- The default LR 1e-5 reuses the four standard runs. `learning_rate_selection.json` selects the highest mean final validation reward for subsequent stages.
- Each job directory contains startup.log, launch.json, config.json, metrics.jsonl, rollouts.jsonl and checkpoints. A .complete marker is written only after a zero exit status and log/checkpoint checks.
- All training uses local model/data and the existing .venv, with Hugging Face offline and WandB disabled. No uv synchronization runs.
- The supervisor is detached from the chat and shell. Disconnecting the network does not stop it; shutting down/rebooting the computer does.
- Failure stops the next wave/stage; inspect status.json and the failed job startup.log. This queue does not automatically retry a failed run.
- Ports 8500–8503 are reused only after the previous wave exits. Keep these GPU pairs and ports free of other experiments.
- source/ records the code snapshot; smoke and formal outputs are separated. Do not reuse job output directories for fresh training.

- Completed experiment checkpoints are automatically removed by cleanup_completed_checkpoints.py. Metrics, rollouts and logs remain. Standard/LR final checkpoints are retained until learning-rate selection finishes.

- Failure policy updated: record failed jobs and continue next wave/stage. Job timeout: 4 hours; timed-out trainer and descendants are killed. Final status distinguishes finished_with_errors. Offline supervisor can adopt existing trainers after restart.

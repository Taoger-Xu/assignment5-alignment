"""Retry failed off-policy jobs in separate directories, preserving original evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment_queue import Queue, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    q = Queue(args.root.resolve())
    try:
        q.preflight()
        state = json.loads((args.source / 'status.json').read_text())
        jobs = []
        excluded = {'model-path', 'train-path', 'validation-path', 'validation-limit',
                    'num-steps', 'policy-device', 'inference-gpu', 'port', 'output-dir',
                    'checkpoint-interval', 'checkpoint-dir', 'resume-from'}
        for name, job in state['jobs'].items():
            if name.startswith('grpo_offpolicy/') and job['state'] == 'failed':
                command = json.loads((Path(job['output']) / 'launch.json').read_text())['command']
                options = {k.removeprefix('--'): v for k, v in zip(command[3::2], command[4::2])
                           if k.removeprefix('--') not in excluded}
                jobs.append((name.split('/')[-1], options))
        write_json(q.root / 'retry_plan.json', {'source': str(args.source), 'jobs': jobs,
                    'changes': ['float32 log probabilities and ratios', 'mask before exp',
                                'skip non-finite optimizer updates with explicit metric']})
        q.run_stage('smoke', jobs, smoke=True)
        passed = [(label, opts) for label, opts in jobs
                  if q.status['jobs'][f'smoke/{label}']['state'] == 'complete']
        q.run_stage('grpo_offpolicy', passed)
        failed = [k for k,v in q.status['jobs'].items() if v['state'] == 'failed']
        q.status.update(state='finished_with_errors' if failed else 'complete',
                        stage='done', failed_jobs=failed)
        q.persist()
        (q.root / '.complete').touch()
    except Exception as exc:
        q.status.update(state='failed', error=str(exc))
        q.persist()
        raise


if __name__ == '__main__':
    main()

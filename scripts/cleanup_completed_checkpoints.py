"""Remove checkpoints only after the experiment supervisor validates completion."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time


def cleanup(root):
    for marker in root.glob('*/*/.complete'):
        output = marker.parent
        checkpoints = output / 'checkpoints'
        if not checkpoints.is_dir():
            continue
        # The already-running supervisor checks the final checkpoint again when
        # selecting a learning rate. Retain it until that selection is recorded.
        if output.parent.name in {'grpo_standard', 'lr_sweep'} and not (
                root / 'learning_rate_selection.json').exists():
            for checkpoint in checkpoints.glob('step_*'):
                if checkpoint.name != 'step_000200' and checkpoint.is_dir():
                    shutil.rmtree(checkpoint)
                    print(f'Deleted intermediate {checkpoint}', flush=True)
            continue
        # The supervisor creates .complete after checking metrics and checkpoint.
        size = sum(p.stat().st_size for p in checkpoints.rglob('*') if p.is_file())
        record = output / 'checkpoints_deleted.json'
        temporary = record.with_suffix('.tmp')
        temporary.write_text(json.dumps({'authorized': True, 'bytes': size,
                                         'started': time.time()}, indent=2))
        temporary.replace(record)
        shutil.rmtree(checkpoints)
        print(f'Deleted {checkpoints}: {size / 2**30:.2f} GiB', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'checkpoint_cleanup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            cleanup(root)
            if (root / '.complete').exists():
                break
            time.sleep(15)


if __name__ == '__main__':
    main()

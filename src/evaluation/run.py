"""One unchanged official evaluation with immutable input/output hash evidence."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from src.common import ROOT, sha256, timestamp, write_json
from src.release.check import check_files
from scripts.setup_inputs import verify_starter


def run_official(submission, out, *, seed=0, n_per_class=1000, batch_size=500,
                 data_root=ROOT / 'data', device='cuda', metric_cache=ROOT / 'models/evaluator'):
    submission, out = Path(submission).resolve(), Path(out).resolve()
    if out.exists() or out.with_suffix('.provenance.json').exists():
        raise FileExistsError('Original evaluation evidence is immutable; use a new output path')
    verify_starter()
    official_before = {str(path.relative_to(ROOT / 'starter_kit')).replace('\\', '/'): sha256(path)
                       for path in (ROOT / 'starter_kit').rglob('*')
                       if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    before = check_files(submission)['file_sha256']
    command = [sys.executable, str(ROOT / 'starter_kit/evaluate.py'), '--submission', str(submission),
        '--grading', '--n-per-class', str(n_per_class), '--seed', str(seed), '--batch-size', str(batch_size),
        '--device', device, '--data-root', str(Path(data_root).resolve()), '--out', str(out)]
    child_env = dict(os.environ) | {'TORCH_HOME': str(Path(metric_cache).resolve()),
                                  'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
    out.parent.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    with out.with_suffix('.stdout.log').open('w', encoding='utf-8') as log:
        process = subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    after = check_files(submission)['file_sha256']
    official_after = {name: sha256(ROOT / 'starter_kit' / name) for name in official_before}
    if after != before or official_after != official_before:
        raise ValueError('Inference or official evaluator files changed during evaluation')
    record = {'command': command, 'started': started, 'finished': timestamp(), 'returncode': process.returncode,
              'submission_file_sha256': before, 'seed': seed, 'n_per_class': n_per_class,
              'batch_size': batch_size, 'grading': True, 'device': device,
              'result_sha256': sha256(out) if out.exists() else None}
    record['official_source_sha256'] = official_before
    write_json(out.with_suffix('.provenance.json'), record)
    if process.returncode:
        raise RuntimeError(f'Official evaluator failed; inspect {out.with_suffix(".stdout.log")}')
    result = json.loads(out.read_text())
    if result['metrics']['memorization_rate'] is None:
        raise ValueError('Official run omitted memorization; preserved output is not complete release evidence')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission', default='submission', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--n-per-class', default=1000, type=int)
    p.add_argument('--batch-size', default=500, type=int)
    p.add_argument('--data-root', default=ROOT / 'data', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--metric-cache', default=ROOT / 'models/evaluator', type=Path)
    args = p.parse_args()
    result = run_official(args.submission, args.out, seed=args.seed, n_per_class=args.n_per_class,
                          batch_size=args.batch_size, data_root=args.data_root, device=args.device,
                          metric_cache=args.metric_cache)
    print(result['metrics'])


if __name__ == '__main__':
    main()

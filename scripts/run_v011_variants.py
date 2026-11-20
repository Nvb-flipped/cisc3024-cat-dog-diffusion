"""Config-only variants trained on the frozen v011 manifest (same recipe otherwise).

  python -m scripts.run_v011_variants train NAME
  python -m scripts.run_v011_variants screen NAME --updates U --sampler S [--seeds ...]
Variants change only model width or EMA decay, never code, so all frozen plans stay valid.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.run_v010_ext import SAMPLERS
from scripts.run_v011 import HOLDOUT, MANIFEST, PARENT_CONFIG, parity
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.release.export import export_checkpoint
from src.training.optimize import evaluate

VARIANTS = {
    'v011_wide_flip': dict(model={'channels': [48, 96, 144, 192]}, training={}, updates=100000),
    'v011_ema9995_flip': dict(model={}, training={'ema_decay': 0.9995}, updates=100000),
}


def config(name):
    variant = VARIANTS[name]
    settings = read(PARENT_CONFIG)
    settings['model'] = settings['model'] | variant['model']
    settings['training'] = settings['training'] | variant['training'] | {'updates': variant['updates']}
    path = Path('configs/runs') / f'{name}.json'
    if not path.exists():
        write_json(path, settings)
    if read(path) != settings:
        raise ValueError(f'Frozen config changed: {path}')
    return path, settings


def plan(name):
    cfg, settings = config(name)
    path = Path('results/plans') / f'{name}.json'
    spec = dict(inputs={str(p): sha256(p) for p in (MANIFEST, HOLDOUT, MANIFEST.with_suffix('.provenance.json'),
                PARENT_CONFIG, cfg, *map(Path, ('src/training/train.py', 'src/training/step.py',
                'src/training/data.py', 'submission/model.py', 'submission/diffusion.py')))},
                variant=VARIANTS[name], comparison='v011_flip (same manifest/recipe; only the listed config change)')
    if not path.exists():
        write_json(path, spec | {'created': timestamp()})
    frozen = read(path)
    frozen.pop('created', None)
    if frozen != spec:
        raise ValueError(f'Frozen plan changed: {path}')
    return cfg, settings


def train(name):
    parity()
    cfg, settings = plan(name)
    verify_starter()
    run = Path('checkpoints') / f'{name}_seed17'
    run.mkdir(parents=True, exist_ok=True)
    digest = sha256(MANIFEST)
    for partial in run.glob('*.tmp'):
        partial.unlink()
    for orphan in run.glob('update_*.pt'):
        if not orphan.with_suffix('.json').exists():
            (run / 'orphaned').mkdir(exist_ok=True)
            orphan.replace(run / 'orphaned' / f'{orphan.name}.{timestamp().replace(":", "")}')
    saved = sorted(run.glob('update_*.json'))
    if not saved and (run / 'run.json').exists():
        quarantine = run / 'orphaned' / timestamp().replace(':', '')
        quarantine.mkdir(parents=True)
        for leftover in [*run.glob('*.json'), *run.glob('*.csv')]:
            leftover.replace(quarantine / leftover.name)
    updates = settings['training']['updates']
    if (run / 'completed.json').exists() and read(run / 'completed.json')['last_update'] >= updates:
        return print(read(run / 'completed.json'))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(cfg), '--manifest', str(MANIFEST),
               '--validation-manifest', str(HOLDOUT), '--seed', '17', '--updates', str(updates), '--out', str(run)]
    if saved:
        command += ['--resume', str(verified_checkpoint(run, read(saved[-1])['update'], digest))]
    subprocess.run(command, check=True)


def screen(name, updates, sampler, seeds):
    plan(name)
    run, digest = Path('checkpoints') / f'{name}_seed17', sha256(MANIFEST)
    for update in updates:
        checkpoint = verified_checkpoint(run, update, digest)
        export = Path('private_build') / name / f'u{update:06d}_{sampler}'
        if not export.exists():
            export_checkpoint(checkpoint, MANIFEST, export, **SAMPLERS[sampler])
        results = Path('results/experiments') / name / f'u{update:06d}_{sampler}'
        if not (results / 'interface.json').exists():
            write_json(results / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
        runtime = {k: SAMPLERS[sampler][k] for k in ('weights', 'steps', 'guidance')}
        result = evaluate(checkpoint, MANIFEST, export, results, seeds=seeds, **runtime)
        for row in result['evidence']:
            m = row['metrics']
            print({'update': update, 'sampler': sampler, 'seed': row['seed'], 'marks': row['marks']['total'],
                   'accuracy': round(m['accuracy'], 4), 'fid': round(m['fid'], 2), 'recall': round(m['recall'], 4),
                   'precision': round(m['precision'], 4), 'memorization': m['memorization_rate']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'screen'))
    parser.add_argument('name', choices=sorted(VARIANTS))
    parser.add_argument('--updates', type=int, nargs='+')
    parser.add_argument('--sampler', default='g275', choices=sorted(SAMPLERS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    train(args.name) if args.phase == 'train' else screen(args.name, args.updates, args.sampler, args.seeds)


if __name__ == '__main__':
    main()

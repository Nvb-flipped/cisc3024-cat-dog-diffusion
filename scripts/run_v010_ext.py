"""Continuations of frozen v010 runs and sampler retuning (separate from run_v010.py, whose own
hash is part of its frozen plans).

  train NAME                   resume the parent's verified checkpoint into checkpoints/NAME_seed17
                               with the identical config (only the update budget changes)
  screen NAME --updates U --sampler S [--seeds ...]
                               export EMA checkpoints with sampler S and run the official evaluator
Checkpoints every 1k updates; every phase is resumable on the shared GPU.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.run_v010 import HOLDOUT, MANIFEST, SAMPLERS as BASE_SAMPLERS, plan_for
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.release.export import export_checkpoint
from src.training.optimize import evaluate

EXTENSIONS = {
    'v010_flip_ext': dict(parent='v010_attn16_flip', parent_update=80000, updates=160000),
    'v010_wide_flip_ext': dict(parent='v010_wide_flip', parent_update=80000, updates=160000),
}
SAMPLERS = dict(BASE_SAMPLERS)
# Names carry no dots: export folders are imported as Python packages by the interface check.
for guidance in (2.0, 2.25, 2.5, 2.75, 3.0):
    SAMPLERS[f'g{round(guidance * 100):03d}'] = BASE_SAMPLERS['v009'] | {'guidance': guidance}
for strength in (1.5, 1.0):
    SAMPLERS[f'g250_freeu{round(strength * 10):02d}'] = BASE_SAMPLERS['v009'] | {'guidance': 2.5, 'freeu_strength': strength}
# Visual-quality variants around the best seed-0 setting (guidance 3.0): FreeU off/1.5, 100 steps.
SAMPLERS['g300_nofreeu'] = BASE_SAMPLERS['v009'] | {'guidance': 3.0, 'freeu': False, 'freeu_strength': None}
SAMPLERS['g300_freeu15'] = BASE_SAMPLERS['v009'] | {'guidance': 3.0, 'freeu_strength': 1.5}
SAMPLERS['g300_s100'] = BASE_SAMPLERS['v009'] | {'guidance': 3.0, 'steps': 100}
# Guidance limited to an interval (Kynkaanniemi et al., 2024): no guidance at the highest noise levels,
# where it mostly collapses diversity (recall).
SAMPLERS['g300_gmax850'] = BASE_SAMPLERS['v009'] | {'guidance': 3.0, 'guidance_max_timestep': 850}
SAMPLERS['g300_gmax700'] = BASE_SAMPLERS['v009'] | {'guidance': 3.0, 'guidance_max_timestep': 700}
SAMPLERS['g350_gmax700'] = BASE_SAMPLERS['v009'] | {'guidance': 3.5, 'guidance_max_timestep': 700}


def run_dir(name):
    return Path('checkpoints') / f'{name}_seed17'


def parent_info(name):
    """Return (run dir, digest, config path) for a frozen v010 variant or an extension."""
    if name in EXTENSIONS:
        ext = EXTENSIONS[name]
        _, _, _, _ = plan_for(ext['parent'])
        return run_dir(name), sha256(MANIFEST), Path('configs/runs') / f"{ext['parent']}.json"
    plan_for(name)
    return run_dir(name), sha256(MANIFEST), Path('configs/runs') / f'{name}.json'


def train(name):
    ext = EXTENSIONS[name]
    out, digest, config = parent_info(name)
    parent_checkpoint = verified_checkpoint(run_dir(ext['parent']), ext['parent_update'], digest)
    plan = Path('results/plans') / f'{name}.json'
    spec = dict(parent=ext['parent'], parent_checkpoint=str(parent_checkpoint),
                parent_checkpoint_sha256=sha256(parent_checkpoint), config=str(config),
                config_sha256=sha256(config), manifest_sha256=digest, holdout_sha256=sha256(HOLDOUT),
                updates=ext['updates'], rule='Same recipe and seed; only the update budget grows. '
                'Screen EMA checkpoints on official seed 0; memorization must stay well below 5%.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    frozen = read(plan)
    frozen.pop('created', None)
    if frozen != spec:
        raise ValueError(f'Frozen extension plan changed: {plan}')
    verify_starter()
    for partial in out.glob('*.tmp'):
        partial.unlink()
    for orphan in out.glob('update_*.pt'):
        if not orphan.with_suffix('.json').exists():
            (out / 'orphaned').mkdir(exist_ok=True)
            orphan.replace(out / 'orphaned' / f'{orphan.name}.{timestamp().replace(":", "")}')
    saved = sorted(out.glob('update_*.json'))
    if (out / 'completed.json').exists() and read(out / 'completed.json')['last_update'] >= ext['updates']:
        print({'completed': read(out / 'completed.json')})
        return
    resume = verified_checkpoint(out, read(saved[-1])['update'], digest) if saved else parent_checkpoint
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config), '--manifest', str(MANIFEST),
               '--validation-manifest', str(HOLDOUT), '--seed', '17', '--updates', str(ext['updates']),
               '--out', str(out), '--resume', str(resume)]
    subprocess.run(command, check=True)


def screen(name, updates, sampler, seeds):
    out, digest, _ = parent_info(name)
    for update in updates:
        checkpoint = verified_checkpoint(out, update, digest)
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
                   'valid': row['marks']['valid'], 'accuracy': round(m['accuracy'], 4), 'fid': round(m['fid'], 2),
                   'recall': round(m['recall'], 4), 'precision': round(m['precision'], 4),
                   'memorization': m['memorization_rate']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'screen'))
    parser.add_argument('name')
    parser.add_argument('--updates', type=int, nargs='+')
    parser.add_argument('--sampler', default='v009', choices=sorted(SAMPLERS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    if args.phase == 'train':
        train(args.name)
    else:
        screen(args.name, args.updates, args.sampler, args.seeds)


if __name__ == '__main__':
    main()

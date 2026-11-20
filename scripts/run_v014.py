"""v014: personal v010 rows + every AI-found row approved by agent visual review (full ranked review, re-checks, re-crops, one-head re-check), trained with the exact v010 + mirroring recipe.

  parity                      build the v014 cache and verify it against an official rebuild
  train                       fresh seed-17 training (resumable, 1k-update checkpoints)
  screen --updates U --sampler S [--seeds ...]
                              official evaluations of EMA checkpoints
Only the manifest (and the update budget) differs from v010_attn16_flip, so the comparison
isolates the effect of the AI-curated rows.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.run_v010_ext import SAMPLERS
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.release.export import export_checkpoint
from src.training.optimize import evaluate

NAME = 'v014_flip'
MANIFEST = Path('manifests/v014.csv')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
PARENT_CONFIG = Path('configs/runs/v010_attn16_flip.json')
UPDATES = 160000
RUN = Path('checkpoints') / f'{NAME}_seed17'
PARITY = Path('results/curation/v014_cache_parity.json')


def config():
    settings = read(PARENT_CONFIG)
    settings['training'] = settings['training'] | {'updates': UPDATES}
    path = Path('configs/runs') / f'{NAME}.json'
    if not path.exists():
        write_json(path, settings)
    if read(path) != settings:
        raise ValueError(f'Frozen config changed: {path}')
    return path


def parity():
    if not PARITY.exists():
        from src.training.data import load_manifest_data
        for path in (MANIFEST, HOLDOUT):
            load_manifest_data(path)
        subprocess.run([sys.executable, '-m', 'scripts.verify_manifest_cache', str(MANIFEST), str(HOLDOUT),
                        '--out', str(PARITY)], check=True)
    if not all(r['pixels_equal'] and r['labels_equal'] for r in read(PARITY)['checks']):
        raise ValueError('v014 cache differs from the official rebuild')


def plan():
    cfg = config()
    path = Path('results/plans') / f'{NAME}.json'
    spec = dict(inputs={str(p): sha256(p) for p in (Path(__file__).relative_to(ROOT), MANIFEST, HOLDOUT,
                MANIFEST.with_suffix('.provenance.json'), PARITY, PARENT_CONFIG, cfg,
                Path('results/fullpool_20261119/selection_v014.json'),
                *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                            'src/training/lineage.py', 'submission/model.py', 'submission/diffusion.py')))},
                updates=UPDATES, comparison='v010_attn16_flip (same config except manifest and update budget)',
                rule='Screen EMA checkpoints with the v009 sampler and guidance 3.0 on official seed 0; '
                     'compare with v010 64k on FID/recall/accuracy/memorization and visual quality; '
                     'multi-seed before any selection.')
    if not path.exists():
        write_json(path, spec | {'created': timestamp()})
    frozen = read(path)
    frozen.pop('created', None)
    if frozen != spec:
        raise ValueError(f'Frozen plan changed: {path}')
    return cfg


def train():
    parity()
    cfg = plan()
    verify_starter()
    digest = sha256(MANIFEST)
    RUN.mkdir(parents=True, exist_ok=True)
    for partial in RUN.glob('*.tmp'):
        partial.unlink()
    for orphan in RUN.glob('update_*.pt'):
        if not orphan.with_suffix('.json').exists():
            (RUN / 'orphaned').mkdir(exist_ok=True)
            orphan.replace(RUN / 'orphaned' / f'{orphan.name}.{timestamp().replace(":", "")}')
    saved = sorted(RUN.glob('update_*.json'))
    if not saved and (RUN / 'run.json').exists():
        quarantine = RUN / 'orphaned' / timestamp().replace(':', '')
        quarantine.mkdir(parents=True)
        for leftover in [*RUN.glob('*.json'), *RUN.glob('*.csv')]:
            leftover.replace(quarantine / leftover.name)
    if (RUN / 'completed.json').exists() and read(RUN / 'completed.json')['last_update'] >= UPDATES:
        return print(read(RUN / 'completed.json'))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(cfg), '--manifest', str(MANIFEST),
               '--validation-manifest', str(HOLDOUT), '--seed', '17', '--updates', str(UPDATES), '--out', str(RUN)]
    if saved:
        command += ['--resume', str(verified_checkpoint(RUN, read(saved[-1])['update'], digest))]
    subprocess.run(command, check=True)


def screen(updates, sampler, seeds):
    plan()
    digest = sha256(MANIFEST)
    for update in updates:
        checkpoint = verified_checkpoint(RUN, update, digest)
        export = Path('private_build') / NAME / f'u{update:06d}_{sampler}'
        if not export.exists():
            export_checkpoint(checkpoint, MANIFEST, export, **SAMPLERS[sampler])
        results = Path('results/experiments') / NAME / f'u{update:06d}_{sampler}'
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
    parser.add_argument('phase', choices=('parity', 'train', 'screen'))
    parser.add_argument('--updates', type=int, nargs='+')
    parser.add_argument('--sampler', default='g300', choices=sorted(SAMPLERS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    if args.phase == 'parity':
        parity()
    elif args.phase == 'train':
        train()
    else:
        screen(args.updates, args.sampler, args.seeds)


if __name__ == '__main__':
    main()

"""Run the exact 32k comparison of the personally reviewed v008 expansion."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import rows, verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved

NAME = 'v008_coverage32k'
MANIFEST = Path('manifests/v008.csv')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
PARENT = Path('manifests/v007.csv')
PLAN = Path('results/plans') / f'{NAME}.json'
RUN = Path('checkpoints') / f'{NAME}_seed17'
OUT = Path('results/experiments') / NAME
EXPORT = Path('private_build') / NAME
REFERENCE = Path('results/experiments/v007_class_annealing010/selection.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate', 'confirm'))
    args = parser.parse_args()
    verify_starter()
    parent, expanded = rows(PARENT), rows(MANIFEST)
    manifest_hash, holdout_hash = sha256(MANIFEST), sha256(HOLDOUT)
    if not parent < expanded or holdout_hash != sha256(PARENT.with_suffix('.validation.csv')):
        raise ValueError('The expanded manifest must preserve all parent rows and the exact holdout')
    cache = Path('results/curation/v008_cache_parity.json')
    checks = read(cache)['checks']
    if [(x['manifest_sha256'], x['rows']) for x in checks] != [
            (manifest_hash, len(expanded)), (holdout_hash, len(rows(HOLDOUT)))
    ] or not all(x['pixels_equal'] and x['labels_equal'] for x in checks):
        raise ValueError('Expected verified official crop/cache parity')
    reference = read(REFERENCE)['candidate']
    parent_run = Path(reference['checkpoint']).parent / 'run.json'
    config = Path('configs/runs') / f'{NAME}.json'
    parent_config = read(parent_run)['config']
    parent_config['training']['updates'] = 32000
    if not config.exists():
        write_json(config, parent_config)
    if read(config) != parent_config:
        raise ValueError('Expected the unchanged v007 training recipe')
    reference_config = read(Path(reference['exported']) / 'inference.json')
    settings = dict(weights='ema', steps=50, guidance=3.25, pag_scale=.15,
                    score_sharpness=1.03, score_sigma=.5, class_noise_max=.10)
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config),
               '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
               '--updates', '32000', '--out', str(RUN)]
    paths = (Path(__file__), PARENT, MANIFEST, HOLDOUT, MANIFEST.with_suffix('.provenance.json'),
             cache, REFERENCE, parent_run, config, *map(Path, (
                 'src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                 'src/training/validation.py', 'src/training/lineage.py', 'submission/model.py',
                 'submission/diffusion.py', 'submission/generate.py', 'src/release/export.py',
                 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(p): sha256(p) for p in paths}, command=command, sampler=settings,
                added_crops=len(expanded) - len(parent), seeds=[0, 101, 202], n_per_class=1000)
    if not PLAN.exists():
        write_json(PLAN, spec)
    if read(PLAN) != spec:
        raise ValueError('Frozen coverage comparison inputs changed')
    complete = RUN / 'completed.json'
    if args.phase == 'train' and not complete.exists():
        receipts = sorted(RUN.glob('update_*.json'))
        if receipts:
            previous = verified_checkpoint(RUN, read(receipts[-1])['update'], manifest_hash)
            command += ['--resume', str(previous)]
        subprocess.run(command, check=True)
    receipt = read(complete)
    initial_path, = RUN.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (receipt['last_update'], receipt['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            32000, manifest_hash, None, 17, sha256(config), holdout_hash):
        raise ValueError('Training lineage differs from the frozen comparison')
    checkpoint = verified_checkpoint(RUN, 32000, manifest_hash)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if args.phase == 'train':
        return
    if not EXPORT.exists():
        export_checkpoint(checkpoint, MANIFEST, EXPORT, **settings)
    if read(EXPORT / 'inference.json') != reference_config:
        raise ValueError('Expected the exact v007 sampler')
    if not (OUT / 'interface.json').exists():
        write_json(OUT / 'interface.json', verify_interface(EXPORT, EXPORT) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    runtime = {k: settings[k] for k in ('weights', 'steps', 'guidance')}
    screen = evaluate(checkpoint, MANIFEST, EXPORT, OUT, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(OUT / 'screen.json', dict(plan_sha256=sha256(PLAN), candidate=screen, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and args.phase == 'confirm':
        selection = evaluate(checkpoint, MANIFEST, EXPORT, OUT, seeds=spec['seeds'], **runtime)
        write_json(OUT / 'selection.json', dict(screen_sha256=sha256(OUT / 'screen.json'),
                   candidate=selection, improves_reference=quality_improved(reference, selection)))
        print([item['marks'] for item in selection['evidence']], flush=True)


if __name__ == '__main__':
    main()

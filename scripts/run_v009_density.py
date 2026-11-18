"""Matched v009 comparison that changes only training-crop sampling probabilities."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.diagnostics import overfit
from src.training.optimize import evaluate, quality_improved

NAME = 'v009_appearance32k'
DIAGNOSTIC = Path('results/diagnostics') / NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('diagnostic', 'train', 'evaluate', 'confirm'))
    phase = parser.parse_args().phase
    if phase == 'diagnostic':
        overfit(out=DIAGNOSTIC, appearance_sampling=True)
        return
    manifest = Path('manifests/v009.csv')
    holdout = manifest.with_suffix('.validation.csv')
    parent_plan = Path('results/plans/v009_coverage32k.json')
    parent_screen = Path('results/experiments/v009_coverage32k/screen.json')
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    selection_path = parent_screen.with_name('selection.json')
    if selection_path.exists() and read(selection_path)['improves_reference']:
        reference_path = selection_path
    reference = read(reference_path)['candidate']
    old_plan = read(parent_plan)
    command = old_plan['command'].copy()
    config = Path('configs/runs') / f'{NAME}.json'
    settings = read(Path(command[command.index('--config') + 1]))
    settings['training']['appearance_sampling'] = True
    if not config.exists():
        write_json(config, settings)
    if read(config) != settings:
        raise ValueError('The comparison may only change source sampling probabilities')
    run = Path('checkpoints') / f'{NAME}_seed17'
    for key, value in (('--config', config), ('--out', run)):
        command[command.index(key) + 1] = str(value)
    command[0] = sys.executable
    diagnostic_path = DIAGNOSTIC / 'result.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic['appearance_sampling'], diagnostic['updates'], diagnostic['passed'],
            diagnostic['diagnostic_only']) != (True, 2000, True, True):
        raise ValueError('Expected a passing fixed 2k diagnostic')
    if any(sha256(p) != digest for p, digest in diagnostic['appearance_sampling_sources'].items()):
        raise ValueError('Sources changed since the fixed diagnostic')
    weights_path = Path('results/diagnostics/v009_appearance_weights_v1.json')
    weights = read(weights_path)
    digest = sha256(manifest)
    if weights['manifest_sha256'] != digest or weights['source_code_sha256'] != sha256('src/training/density_sampling.py'):
        raise ValueError('Manifest or code differs from the pool-only weight probe')
    paths = (Path(__file__), parent_plan, parent_screen, reference_path, manifest, holdout,
             manifest.with_suffix('.provenance.json'), config, diagnostic_path, weights_path,
             *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/density_sampling.py',
                         'src/training/data.py', 'src/training/validation.py', 'submission/model.py',
                         'submission/diffusion.py', 'src/release/export.py', 'src/evaluation/run.py')))
    plan = Path('results/plans') / f'{NAME}.json'
    spec = dict(inputs={str(p): sha256(p) for p in paths}, command=command, sampler=old_plan['sampler'],
                seeds=[0, 101, 202], n_per_class=1000,
                rule='Keep the exact targets and species mass. Increase exposure to locally uncommon '
                     'crop appearances using capped within-species weights.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen appearance comparison inputs changed')
    verify_starter()
    completion = run / 'completed.json'
    if phase == 'train' and not completion.exists():
        receipts = sorted(run.glob('update_*.json'))
        if receipts:
            prior = verified_checkpoint(run, read(receipts[-1])['update'], digest)
            command += ['--resume', str(prior)]
        subprocess.run(command, check=True)
    receipt = read(completion)
    initial_path, = run.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (receipt['last_update'], receipt['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256'],
            initial['appearance_sampling_sha256']) != (
            32000, digest, None, 17, sha256(config), sha256(holdout), weights['weights_sha256']):
        raise ValueError('Unexpected training lineage or sampling weights')
    checkpoint = verified_checkpoint(run, 32000, digest)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if phase == 'train':
        return
    out = Path('results/experiments') / NAME
    export = Path('private_build') / NAME
    control = read(parent_screen)['candidate']
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json') != read(Path(control['exported']) / 'inference.json'):
        raise ValueError('The exact v009 sampler is required')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    matched = validated_result(control['evidence'][0]['path'], 0, control['exported'])
    runtime = {k: spec['sampler'][k] for k in ('weights', 'steps', 'guidance')}
    result = evaluate(checkpoint, manifest, export, out, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, result)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=result, advances=advances,
               improves_matched=quality_improved({'score': score_results([matched])}, result)))
    print({'seed': 0, 'marks': result['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and phase == 'confirm':
        selection = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selection,
                   improves_reference=quality_improved(reference, selection)))
        print([item['marks'] for item in selection['evidence']], flush=True)


if __name__ == '__main__':
    main()

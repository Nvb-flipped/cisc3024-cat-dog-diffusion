"""Compare 34k and 32k input perturbation checkpoints with the same sampler."""

import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v007_perturbation_34k'
    manifest = Path('manifests/v007.csv')
    holdout = manifest.with_suffix('.validation.csv')
    config = Path('configs/runs/v007_input_perturbation.json')
    run = Path('checkpoints/v007_input_perturbation_seed17')
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    control_path = Path('results/experiments/v007_input_perturbation/comparison.json')
    reference_path = Path('results/experiments/v007_32k_temporal103/selection.json')
    control = read(control_path)['screened']
    best = read(reference_path)['candidate']
    manifest_hash = sha256(manifest)
    parent = verified_checkpoint(run, 32000, manifest_hash)
    if sha256(parent) != control['checkpoint_sha256']:
        raise ValueError('The extension must start from the evaluated 32k control')
    target_update = 34000
    target = run / f'update_{target_update:06d}.pt'
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config),
               '--manifest', str(manifest), '--validation-manifest', str(holdout), '--seed', '17',
               '--out', str(run), '--resume', str(parent), '--updates', str(target_update)]
    paths = (Path(__file__), manifest, manifest.with_suffix('.provenance.json'), holdout, config, parent,
             control_path, reference_path, Path('src/training/train.py'), Path('src/training/step.py'),
             Path('submission/model.py'), Path('submission/diffusion.py'), Path('src/release/export.py'))
    preflight = plan.with_name(f'{name}_preflight_failed.json')
    if preflight.exists():
        paths += (preflight, export / 'model.py', export / 'diffusion.py')
    spec = dict(inputs={str(path): sha256(path) for path in paths}, command=command,
                sampler=sampler | {'pag_scale': .15}, seed=0, n_per_class=1000,
                rule='Resume for 2,000 updates from the exact reviewed 32k control. Its 3.35% copy rate '
                     'provides margin to test whether the later fit improves fidelity or coverage.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen extension inputs changed')
    verify_starter()
    if not target.exists():
        subprocess.run(command, check=True)
    checkpoint = verified_checkpoint(run, target_update, manifest_hash)
    completion = read(run / 'completed.json')
    segment = read(run / f"segment_{completion['segment_id']}.json")
    if (completion['last_update'], completion['manifest_sha256'], segment['parent_checkpoint_sha256']) != (
            target_update, manifest_hash, sha256(parent)):
        raise ValueError('Extension receipts do not bind to the frozen control')
    print({'training_completed': target_update, 'seconds': completion['segment_seconds'],
           'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json')['sampling']['pag_scale'] != .15:
        raise ValueError('Sampler differs from the matched control')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    matched = validated_result(control['evidence'][0]['path'], 0, control['exported'])
    anchor = validated_result(best['evidence'][0]['path'], 0, best['exported'])
    screen = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances,
               improves_matched=quality_improved({'score': score_results([matched])}, screen)))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)


if __name__ == '__main__':
    main()

"""Test one fresh v007 fit with broader exposure to low-SNR timesteps."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import read_config, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.diagnostics import overfit
from src.training.optimize import evaluate, quality_improved

NAME = 'v007_timestep_focus065'
FOCUS = .65
UPDATES = 32000
DIAGNOSTIC = Path(f'results/diagnostics/{NAME}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('diagnostic', 'train', 'evaluate', 'confirm'))
    args = parser.parse_args()
    if args.phase == 'diagnostic':
        overfit(out=DIAGNOSTIC, timestep_sampling_focus=FOCUS)
        return

    manifest = Path('manifests/v007.csv')
    holdout = manifest.with_suffix('.validation.csv')
    run = Path('checkpoints') / f'{NAME}_seed17'
    out = Path('results/experiments') / NAME
    export = Path('private_build') / NAME
    plan = Path('results/plans') / f'{NAME}.json'
    config_path = Path('configs/runs') / f'{NAME}.json'
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    reference = read(reference_path)['candidate']
    anchor_run = Path(reference['checkpoint']).parent / 'run.json'
    config = read_config(anchor_run)['config']
    config['training'].update(updates=UPDATES, timestep_sampling_focus=FOCUS)
    diagnostic_path = DIAGNOSTIC / 'result.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic['timestep_sampling_focus'], diagnostic['updates'], diagnostic['passed'],
            diagnostic['diagnostic_only']) != (FOCUS, 2000, True, True):
        raise ValueError('The exact timestep distribution needs a passing fixed diagnostic')
    if any(sha256(path) != digest for path, digest in diagnostic['timestep_sampling_sources'].items()):
        raise ValueError('Training or diagnostic source changed')
    if not config_path.exists():
        write_json(config_path, config)
    if read(config_path) != config:
        raise ValueError('Only the timestep distribution may change')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5, class_noise_max=.10)
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config_path),
               '--manifest', str(manifest), '--validation-manifest', str(holdout), '--seed', '17',
               '--updates', str(UPDATES), '--out', str(run)]
    paths = (Path(__file__), manifest, manifest.with_suffix('.provenance.json'), holdout, config_path,
             reference_path, anchor_run, diagnostic_path, Path('results/curation/v007_cache_parity.json'),
             *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                         'submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, command=command,
                sampler=sampler | options, seed=0, n_per_class=1000,
                source='https://arxiv.org/abs/2601.21348',
                rule='Use a discrete Gaussian with center .65 and width .22 for 75% of timestep probability. '
                     'The remaining 25% is uniform. Keep exact targets, full timestep support, all v007 rows, '
                     'scratch initialization, the seed, update count and the annealing 0.10 sampler fixed.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen timestep comparison inputs changed')
    verify_starter()
    manifest_hash = sha256(manifest)
    completion_path = run / 'completed.json'
    if args.phase == 'train' and not completion_path.exists():
        receipts = sorted(run.glob('update_*.json'))
        if receipts:
            last = verified_checkpoint(run, read(receipts[-1])['update'], manifest_hash)
            command += ['--resume', str(last)]
        subprocess.run(command, check=True)
    completed = read(completion_path)
    initial_path, = run.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (completed['last_update'], completed['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            UPDATES, manifest_hash, None, 17, sha256(config_path), sha256(holdout)):
        raise ValueError('Training receipts do not match the frozen comparison')
    checkpoint = verified_checkpoint(run, UPDATES, manifest_hash)
    print({'training_completed': UPDATES, 'checkpoint_sha256': sha256(checkpoint), 'receipt': completed}, flush=True)
    if args.phase == 'train':
        return
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json') != read(Path(reference['exported']) / 'inference.json'):
        raise ValueError('This comparison requires exactly the same sampler')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screen = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', {'plan_sha256': sha256(plan), 'candidate': screen, 'advances': advances})
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if args.phase == 'confirm' and advances:
        selected = evaluate(checkpoint, manifest, export, out, seeds=[0, 101, 202], **sampler)
        write_json(out / 'selection.json', {'screen_sha256': sha256(out / 'screen.json'), 'candidate': selected,
                                           'improves_reference': quality_improved(reference, selected)})
        print({'confirmation_marks': [row['marks'] for row in selected['evidence']]}, flush=True)


if __name__ == '__main__':
    main()

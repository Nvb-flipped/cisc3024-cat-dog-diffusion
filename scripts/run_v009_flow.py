"""Screen one matched-data linear flow-matching recipe from scratch."""

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
from src.training.optimize import evaluate, quality_improved

NAME = 'v009_flow32k'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate', 'confirm'))
    phase = parser.parse_args().phase
    manifest = Path('manifests/v009.csv')
    holdout = manifest.with_suffix('.validation.csv')
    control_plan = Path('results/plans/v009_coverage32k.json')
    control_screen = Path('results/experiments/v009_coverage32k/screen.json')
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    reference = read(reference_path)['candidate']
    recipe = read(control_plan)
    diagnostic_path = Path('results/diagnostics') / NAME / 'result.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic['prediction_type'], diagnostic['updates'], diagnostic['passed'],
            diagnostic['diagnostic_only']) != ('flow', 2000, True, True):
        raise ValueError('Expected a passing fixed 2k flow diagnostic')
    if any(sha256(p) != digest for p, digest in diagnostic['flow_sources'].items()):
        raise ValueError('Flow sources changed since the fixed diagnostic')
    command = recipe['command'].copy()
    base = Path(command[command.index('--config') + 1])
    settings = read(base)
    settings['training']['prediction_type'] = 'flow'
    settings['sampling']['solver'] = 'flow_heun'
    config = Path('configs/runs') / f'{NAME}.json'
    if not config.exists():
        write_json(config, settings)
    if read(config) != settings:
        raise ValueError('Expected the fixed flow objective and unchanged model, data and optimizer')
    run = Path('checkpoints') / f'{NAME}_seed17'
    for key, value in (('--config', config), ('--out', run)):
        command[command.index(key) + 1] = str(value)
    command[0] = sys.executable
    paths = (Path(__file__), control_plan, control_screen, reference_path, manifest, holdout,
             manifest.with_suffix('.provenance.json'), base, config, diagnostic_path,
             Path('results/curation/v009_cache_parity.json'),
             *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                         'src/training/validation.py', 'submission/model.py', 'submission/diffusion.py',
                         'submission/flow.py', 'submission/generate.py', 'src/release/export.py',
                         'src/release/check.py', 'src/evaluation/run.py')))
    plan = Path('results/plans') / f'{NAME}.json'
    runtime = dict(weights='ema', steps=50, guidance=2.5)
    spec = dict(inputs={str(p): sha256(p) for p in paths}, command=command, sampler=runtime | {'solver': 'flow_heun'},
                seeds=[0, 101, 202], n_per_class=1000, source='https://arxiv.org/abs/2210.02747',
                rule='One linear flow objective and its deterministic Heun sampler, with the same v009 exact '
                     'targets, model architecture, scratch seed and 32k update budget.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen flow comparison inputs changed')
    verify_starter()
    digest = sha256(manifest)
    completion = run / 'completed.json'
    if phase == 'train' and not completion.exists():
        saved = sorted(run.glob('update_*.json'))
        if saved:
            checkpoint = verified_checkpoint(run, read(saved[-1])['update'], digest)
            command += ['--resume', str(checkpoint)]
        subprocess.run(command, check=True)
    receipt = read(completion)
    first, = run.glob('segment_000000_*.json')
    initial = read(first)
    if (receipt['last_update'], receipt['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            32000, digest, None, 17, sha256(config), sha256(holdout)):
        raise ValueError('Flow training lineage differs from the frozen scratch recipe')
    checkpoint = verified_checkpoint(run, 32000, digest)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if phase == 'train':
        return
    export, out = Path('private_build') / NAME, Path('results/experiments') / NAME
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    config = read(export / 'inference.json')['sampling']
    if (config['prediction_type'], config['solver']) != ('flow', 'flow_heun'):
        raise ValueError('The frozen flow needs the full Heun trajectory')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'flow_heun'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and phase == 'confirm':
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                   candidate=selected, improves_reference=quality_improved(reference, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

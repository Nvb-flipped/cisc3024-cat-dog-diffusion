"""Compare a 16x16 encoder attention block with uniform v009 training."""

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

NAME = 'v009_attention16_32k'
DIAGNOSTIC = Path('results/diagnostics') / NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('diagnostic', 'train', 'evaluate', 'confirm'))
    phase = parser.parse_args().phase
    if phase == 'diagnostic':
        overfit(out=DIAGNOSTIC, encoder_attention16=True)
        return
    manifest = Path('manifests/v009.csv')
    holdout = manifest.with_suffix('.validation.csv')
    original_plan = Path('results/plans/v009_coverage32k.json')
    original_result = Path('results/experiments/v009_coverage32k/screen.json')
    ref_path = Path('results/experiments/v007_freeu_cfg310/selection.json')
    reference, matched = read(ref_path)['candidate'], read(original_result)['candidate']
    recipe = read(original_plan)
    diagnostic_path = DIAGNOSTIC / 'result.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic['model'].get('encoder_attention16'), diagnostic['updates'], diagnostic['passed'],
            diagnostic['diagnostic_only']) != (True, 2000, True, True):
        raise ValueError('Expected a passing fixed 2k attention diagnostic')
    if any(sha256(p) != digest for p, digest in diagnostic['attention_sources'].items()):
        raise ValueError('Training sources changed since the diagnostic')
    command = recipe['command'].copy()
    base_config = Path(command[command.index('--config') + 1])
    settings = read(base_config)
    settings['model']['encoder_attention16'] = True
    config = Path('configs/runs') / f'{NAME}.json'
    if not config.exists():
        write_json(config, settings)
    if read(config) != settings:
        raise ValueError('Budget and settings must match the uniform v009 recipe')
    run = Path('checkpoints') / f'{NAME}_seed17'
    for key, value in (('--config', config), ('--out', run)):
        command[command.index(key) + 1] = str(value)
    command[0] = sys.executable
    paths = (Path(__file__), original_plan, original_result, ref_path, manifest, holdout,
             manifest.with_suffix('.provenance.json'), base_config, config, diagnostic_path,
             Path('results/curation/v009_cache_parity.json'),
             *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                         'src/training/validation.py', 'submission/model.py', 'submission/diffusion.py',
                         'submission/generate.py', 'src/release/export.py', 'src/evaluation/run.py')))
    plan = Path('results/plans') / f'{NAME}.json'
    spec = dict(inputs={str(p): sha256(p) for p in paths}, command=command, sampler=recipe['sampler'],
                seeds=[0, 101, 202], n_per_class=1000, source='https://github.com/openai/improved-diffusion',
                rule='Add one 16x16 encoder attention block. Keep the exact pool and control recipe.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen attention comparison inputs changed')
    verify_starter()
    digest = sha256(manifest)
    completion = run / 'completed.json'
    if phase == 'train' and not completion.exists():
        saved = sorted(run.glob('update_*.json'))
        if saved:
            last = verified_checkpoint(run, read(saved[-1])['update'], digest)
            command += ['--resume', str(last)]
        subprocess.run(command, check=True)
    receipt = read(completion)
    first, = run.glob('segment_000000_*.json')
    initial = read(first)
    if (receipt['last_update'], receipt['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            32000, digest, None, 17, sha256(config), sha256(holdout)):
        raise ValueError('The attention run does not match the frozen scratch recipe')
    checkpoint = verified_checkpoint(run, 32000, digest)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if phase == 'train':
        return
    export, out = Path('private_build') / NAME, Path('results/experiments') / NAME
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json')['sampling'] != read(Path(matched['exported']) / 'inference.json')['sampling']:
        raise ValueError('Expected the same sampler as uniform v009')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    control = validated_result(matched['evidence'][0]['path'], 0, matched['exported'])
    runtime = {k: spec['sampler'][k] for k in ('weights', 'steps', 'guidance')}
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen,
               advances=advances, improves_matched=quality_improved({'score': score_results([control])}, screen)))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and phase == 'confirm':
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                   candidate=selected, improves_reference=quality_improved(reference, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

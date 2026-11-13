"""Train and evaluate one bounded EDM challenger against the v002 reference."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.setup_inputs import verify_starter
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


MANIFEST = Path('manifests/v002.csv')
HOLDOUT = Path('manifests/v002.validation.csv')
REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
DIAGNOSTIC = Path('results/diagnostics/edm_v002_shared/result.json')
BASE_CONFIG = Path('configs/runs/86c3c843eb/capacity_32_seed17_u030000.yaml')
CONFIG = Path('configs/runs/63bc3fce35/edm_32_seed17_u030000.yaml')
PLAN = Path('results/plans/edm_32_seed17_63bc3fce35.json')
RUN = Path('checkpoints/edm_32_seed17_u030000_63bc3fce35')
OUTPUT = Path('results/experiments/edm_32_seed17_v002')
EXPORTS = Path('private_build/edm_32_seed17_v002')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    if PLAN.exists():
        raise FileExistsError('Preserve the existing EDM plan')
    diagnostic = read(DIAGNOSTIC)
    if (diagnostic['manifest_sha256'], diagnostic['updates'], diagnostic['passed'],
            diagnostic['source_sha256']['submission/edm.py']) != (
            sha256(MANIFEST), 2000, True, sha256('submission/edm.py')):
        raise ValueError('The fixed EDM integration diagnostic must pass on this implementation')
    reference = read_config(BASE_CONFIG)
    reference['training'].update(prediction_type='edm', updates=30000)
    reference['sampling'].update(steps=26, guidance=2.5)
    if reference != read_config(CONFIG):
        raise ValueError('EDM config changes settings outside the declared recipe')
    sources = [MANIFEST, HOLDOUT, MANIFEST.with_suffix('.provenance.json'), REFERENCE, DIAGNOSTIC,
               CONFIG, BASE_CONFIG, Path(__file__).relative_to(ROOT)]
    sources += [Path(name) for name in ('src/training/train.py', 'src/training/step.py',
                'src/training/data.py', 'src/training/validation.py', 'src/training/optimize.py',
                'submission/model.py', 'submission/edm.py', 'submission/diffusion.py',
                'submission/generate.py', 'src/release/export.py', 'src/release/check.py',
                'src/evaluation/run.py', 'src/evaluation/select.py', 'src/evaluation/sweep.py')]
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
               '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT),
               '--seed', '17', '--out', str(RUN)]
    write_json(PLAN, {'created': timestamp(), 'inputs': {str(path): sha256(path) for path in sources},
                      'command': command, 'paper': 'https://arxiv.org/abs/2206.00364',
                      'hypothesis': 'EDM noise preconditioning and deterministic Heun sampling improve '
                                    'FID and recall at a valid memorization rate.',
                      'sampler': {'weights': 'ema', 'steps': 26, 'guidance': 2.5},
                      'sampler_comparison': '26 Heun levels yield 51 denoiser evaluations per '
                                            'guidance branch versus 50 for reference DDIM.',
                      'updates': [24000, 28000, 30000], 'selection_seeds': [0, 101, 202],
                      'confirmation_seeds': [303, 404], 'n_per_class': 1000,
                      'rule': 'Advance at most one valid seed-0 improvement over the v002 30k '
                              'reference. Advance it to confirmation only if it improves the '
                              'three-seed reference by the established tier or margin rule.'})


def verified_checkpoint(update):
    path = RUN / f'update_{update:06d}.pt'
    receipt = read(path.with_suffix('.json'))
    if (sha256(path), update, sha256(MANIFEST)) != (
            receipt['sha256'], receipt['update'], receipt['manifest_sha256']):
        raise ValueError(f'EDM checkpoint receipt differs: {path}')
    return path


def execute(plan):
    verify_starter()
    for name, digest in plan['inputs'].items():
        if sha256(name) != digest:
            raise ValueError(f'EDM comparison input changed: {name}')
    if not (RUN / 'completed.json').exists():
        command = plan['command']
        if (RUN / 'run.json').exists():
            saved = sorted(RUN.glob('update_*.json'))
            if not saved:
                raise ValueError('Interrupted run has no checkpoint receipt')
            latest = verified_checkpoint(read(saved[-1])['update'])
            command = command + ['--resume', str(latest)]
        print(json.dumps({'event': 'training_started', 'command': command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    finished = read(RUN / 'completed.json')
    initial = next(read(path) for path in RUN.glob('segment_000000_*.json'))
    if (finished['last_update'], finished['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            30000, sha256(MANIFEST), None, 17, sha256(CONFIG), sha256(HOLDOUT)):
        raise ValueError('Training receipts do not match the matched fresh EDM run')
    print(json.dumps({'event': 'training_completed', **finished}), flush=True)

    reference = read(REFERENCE)
    reference_seed = reference['evidence'][0]
    anchor = validated_result(reference_seed['path'], 0, reference['exported'])
    baseline = {'score': score_results([anchor])}
    candidates = []
    for update in plan['updates']:
        name = f'update_{update:06d}'
        candidate = evaluate(verified_checkpoint(update), MANIFEST, EXPORTS / name, OUTPUT / name,
                             n_per_class=plan['n_per_class'], **plan['sampler'])
        candidates.append(candidate)
        print(json.dumps({'event': 'official_evaluation_completed', 'update': update,
                          'evidence': candidate['evidence']}), flush=True)
    top = max((row for row in candidates if quality_improved(baseline, row)),
              key=lambda row: row['score'], default=None)
    write_json(OUTPUT / 'screening.json', {'plan_sha256': sha256(PLAN), 'reference': reference_seed,
                                         'screened': candidates, 'promoted': top})
    selected = None
    if top:
        selected = evaluate(top['checkpoint'], MANIFEST, top['exported'],
                            OUTPUT / f"update_{top['update']:06d}", seeds=plan['selection_seeds'],
                            n_per_class=plan['n_per_class'], **plan['sampler'])
        print(json.dumps({'event': 'official_selection_completed',
                          'evidence': selected['evidence']}), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUTPUT / 'comparison.json', {'plan_sha256': sha256(PLAN), 'candidate': selected,
                                           'improves_reference': improved})
    print(json.dumps({'event': 'comparison_completed', 'improves_reference': improved}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        execute(read(PLAN))

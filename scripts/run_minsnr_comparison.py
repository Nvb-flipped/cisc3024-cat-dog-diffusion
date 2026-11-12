"""Run one matched v002 training experiment with a Min-SNR epsilon loss."""

import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.setup_inputs import verify_starter
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved

MANIFEST = Path('manifests/v002.csv')
HOLDOUT = Path('manifests/v002.validation.csv')
REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
DIAGNOSTIC = Path('results/diagnostics/minsnr_overfit_v002/result.json')
BASE_CONFIG = Path('configs/runs/86c3c843eb/capacity_32_seed17_u030000.yaml')
CONFIG = Path('configs/runs/63bc3fce35/minsnr5_32_seed17_u030000.yaml')
PLAN = Path('results/plans/minsnr5_32_seed17_63bc3fce35.json')
RUN = Path('checkpoints/minsnr5_32_seed17_u030000_63bc3fce35')
OUTPUT = Path('results/experiments/minsnr5_32_seed17_v002')
EXPORTS = Path('private_build/minsnr5_32_seed17_v002')
SAMPLER = {'weights': 'ema', 'steps': 50, 'guidance': 2.5}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_diagnostic():
    record = read(DIAGNOSTIC)
    if (record['updates'], record['prediction_type'], record['min_snr_gamma'], record['passed']) != (
        2000, 'epsilon', 5, True
    ):
        raise ValueError('The fixed Min-SNR integration diagnostic did not pass')


def prepare():
    if PLAN.exists() or CONFIG.exists():
        raise FileExistsError('Preserve the existing Min-SNR plan and config')
    verify_diagnostic()
    config = read_config(BASE_CONFIG)
    config['training']['min_snr_gamma'] = 5
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG.open('x', encoding='utf-8', newline='\n') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    sources = (MANIFEST, HOLDOUT, MANIFEST.with_suffix('.provenance.json'), REFERENCE, DIAGNOSTIC,
               CONFIG, BASE_CONFIG, Path('src/training/step.py'), Path('src/training/train.py'),
               Path('submission/diffusion.py'), Path(__file__).relative_to(ROOT))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG), '--manifest',
               str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17', '--updates', '30000',
               '--out', str(RUN)]
    write_json(PLAN, {'created': timestamp(), 'inputs': {str(path): sha256(path) for path in sources},
                      'command': command, 'hypothesis': 'Bounded timestep weighting can improve distribution '
                      'coverage without the memorization observed with an unweighted velocity target.',
                      'paper': 'https://arxiv.org/abs/2303.09556', 'sampler': SAMPLER,
                      'updates': [20000, 24000, 28000, 30000], 'selection_seeds': [0, 101, 202],
                      'n_per_class': 1000, 'rule': 'Compare each predeclared checkpoint to the matched full '
                      'seed-0 reference. Advance at most one valid improvement to the two remaining seeds. '
                      'Select it only if it improves the full reference by the usual tier or margin rule.'})


def train(plan):
    if (RUN / 'completed.json').exists():
        return
    command = plan['command']
    if (RUN / 'run.json').exists():
        saved = sorted(path for path in RUN.glob('update_*.pt') if path.with_suffix('.json').exists())
        if not saved or sha256(saved[-1]) != read(saved[-1].with_suffix('.json'))['sha256']:
            raise ValueError('Interrupted run has no verified checkpoint')
        command = command + ['--resume', str(saved[-1])]
    print(json.dumps({'event': 'training_started', 'command': command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    verify_starter()
    plan = read(PLAN)
    for name, digest in plan['inputs'].items():
        if sha256(name) != digest:
            raise ValueError(f'Min-SNR comparison input changed: {name}')
    verify_diagnostic()
    train(plan)
    finished = read(RUN / 'completed.json')
    initial = next(read(path) for path in RUN.glob('segment_000000_*.json'))
    if (finished['last_update'], finished['manifest_sha256'], initial['parent_checkpoint_sha256'],
        initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
        30000, sha256(MANIFEST), None, 17, sha256(CONFIG), sha256(HOLDOUT)
    ):
        raise ValueError('Training receipts do not match the matched fresh Min-SNR run')
    print(json.dumps({'event': 'training_completed', **finished}), flush=True)
    reference = read(REFERENCE)
    reference_seed = reference['evidence'][0]
    anchor = validated_result(Path(reference_seed['path']), 0, reference['exported'])
    baseline = {'score': score_results([anchor])}
    screened = []
    for update in plan['updates']:
        name = f'update_{update:06d}'
        checkpoint = RUN / f'{name}.pt'
        if sha256(checkpoint) != read(checkpoint.with_suffix('.json'))['sha256']:
            raise ValueError(f'Checkpoint receipt differs: {checkpoint}')
        screened.append(evaluate(checkpoint, MANIFEST, EXPORTS / name, OUTPUT / name,
                                 n_per_class=plan['n_per_class'], **plan['sampler']))
    top = max((row for row in screened if quality_improved(baseline, row)),
              key=lambda row: row['score'], default=None)
    write_json(OUTPUT / 'screening.json', {'plan_sha256': sha256(PLAN), 'reference': reference_seed,
                                         'screened': screened, 'promoted': top})
    selected = None
    if top:
        selected = evaluate(top['checkpoint'], MANIFEST, top['exported'], OUTPUT / f"update_{top['update']:06d}",
                            seeds=plan['selection_seeds'], n_per_class=plan['n_per_class'], **plan['sampler'])
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUTPUT / 'comparison.json', {'plan_sha256': sha256(PLAN), 'candidate': selected,
                                           'improves_reference': improved})
    print(json.dumps({'event': 'evaluation_completed', 'improves_reference': improved}), flush=True)


if __name__ == '__main__':
    main()

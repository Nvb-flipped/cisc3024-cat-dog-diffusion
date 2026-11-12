"""Run the predeclared v002 velocity-target comparison after its reference."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, write_json
from src.training.optimize import evaluate, meets_target

PLAN = Path('results/plans/velocity_32_seed17_63bc3fce35.json')
REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
DIAGNOSTIC = Path('results/diagnostics/velocity_overfit_v002/result.json')
RUN = Path('checkpoints/velocity_32_seed17_u030000_63bc3fce35')
EXPORT = Path('private_build/velocity_32_seed17_u030000_63bc3fce35')
RESULTS = Path('results/experiments/velocity_32_seed17_v002')
MANIFEST = Path('manifests/v002.csv')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def train(command):
    if (RUN / 'completed.json').exists():
        return
    if (RUN / 'run.json').exists():
        checkpoints = sorted(path for path in RUN.glob('update_*.pt')
                             if path.with_suffix('.json').is_file())
        if not checkpoints:
            raise ValueError('Interrupted run has no receipted checkpoint')
        latest = checkpoints[-1]
        if read(latest.with_suffix('.json'))['sha256'] != sha256(latest):
            raise ValueError('Interrupted checkpoint digest changed')
        command = command + ['--resume', str(latest)]
    print(json.dumps({'event': 'training_started', 'command': command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    verify_starter()
    plan, reference, diagnostic = map(read, (PLAN, REFERENCE, DIAGNOSTIC))
    if meets_target(reference):
        raise ValueError('The prespecified reference already meets the engineering target')
    if not diagnostic['passed'] or diagnostic['updates'] != 2000 or diagnostic['prediction_type'] != 'velocity':
        raise ValueError('The fixed velocity-target diagnostic did not pass')
    for name, expected in plan['inputs'].items():
        if sha256(name) != expected:
            raise ValueError(f'Comparison input changed: {name}')
    for name, hashes in plan['staged_changes'].items():
        if sha256(name) != hashes['staged_sha256']:
            raise ValueError(f'Velocity comparison source differs from the staged plan: {name}')
    train([sys.executable, *plan['command'][1:]])
    finished = read(RUN / 'completed.json')
    checkpoint = RUN / 'update_030000.pt'
    receipt = read(checkpoint.with_suffix('.json'))
    segments = [read(path) for path in RUN.glob('segment_*.json')]
    initial = next(row for row in segments if row['initial_update'] == 0)
    if (finished['last_update'], finished['manifest_sha256'], receipt['sha256'],
        initial['parent_checkpoint_sha256'], initial['seed'],
        initial['config']['training']['prediction_type']) != (
        30000, sha256(MANIFEST), sha256(checkpoint), None, 17, 'velocity'
    ):
        raise ValueError('Velocity checkpoint did not complete the matched fresh training run')
    print(json.dumps({'event': 'training_completed', **finished}), flush=True)
    protocol = plan['evaluation']
    sampler = {key: plan['sampler'][key] for key in ('weights', 'steps', 'guidance')}
    result = evaluate(checkpoint, MANIFEST, EXPORT, RESULTS, seeds=protocol['generation_seeds'],
                      n_per_class=protocol['n_per_class'], **sampler)
    improves = result['score'] is not None and (
        reference['score'] is None or tuple(result['score']) > tuple(reference['score']))
    write_json(RESULTS / 'comparison.json', result | {'plan_sha256': sha256(PLAN),
        'reference_sha256': sha256(REFERENCE), 'diagnostic_sha256': sha256(DIAGNOSTIC),
        'improves_official_selection_rank': improves})
    print(json.dumps({'event': 'evaluation_completed', 'score': result['score'],
                      'improves_official_selection_rank': improves}), flush=True)


if __name__ == '__main__':
    main()

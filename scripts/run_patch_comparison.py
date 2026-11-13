"""Run the matched v002 4×4 and 8×8 pixel patch comparison."""

import argparse
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
from submission.model import ConditionalUNet


MANIFEST = Path('manifests/v002.csv')
HOLDOUT = Path('manifests/v002.validation.csv')
REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
BASE_CONFIG = Path('configs/runs/86c3c843eb/capacity_32_seed17_u030000.yaml')
PLAN = Path('results/plans/patch_grid_v002.json')
OUTPUT = Path('results/experiments/patch_grid_v002')
SAMPLER = {'weights': 'ema', 'steps': 50, 'guidance': 2.5}
UPDATES = (24000, 30000)
SEEDS = (0, 101, 202)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    if PLAN.exists():
        raise FileExistsError(f'Preserve the existing experiment plan: {PLAN}')
    model_source = Path('submission/model.py')
    source_hash = sha256(model_source)
    diagnostics = {size: Path(f'results/diagnostics/patch{size}_detail_overfit/result.json')
                   for size in (4, 8)}
    for size, path in diagnostics.items():
        row = read(path)
        if (row['model_source_sha256'], row['model']['patch_size'], row['model']['patch_detail'],
                row['updates'], row['passed']) != (source_hash, size, True, 2000, True):
            raise ValueError(f'Patch {size} diagnostic does not validate this implementation')
    inputs = [MANIFEST, HOLDOUT, MANIFEST.with_suffix('.provenance.json'), REFERENCE, BASE_CONFIG,
              Path(__file__).relative_to(ROOT), model_source, *diagnostics.values()]
    inputs += [Path(name) for name in ('src/training/train.py', 'src/training/step.py',
               'src/training/validation.py', 'src/training/data.py', 'src/training/optimize.py',
               'submission/generate.py', 'submission/diffusion.py', 'src/release/export.py',
               'src/evaluation/run.py', 'src/evaluation/select.py', 'src/evaluation/sweep.py')]
    reference = read_config(BASE_CONFIG)
    baseline_parameters = sum(weight.numel() for weight in ConditionalUNet(**reference['model']).parameters())
    variants = []
    for size in diagnostics:
        name = f'patch{size}_detail'
        config_path = Path('configs/runs/63bc3fce35') / f'{name}_seed17_u030000.yaml'
        config = read_config(BASE_CONFIG)
        config['model'].update(patch_size=size, patch_detail=True)
        config['training']['updates'] = UPDATES[-1]
        config['sampling'].update(steps=SAMPLER['steps'], guidance=SAMPLER['guidance'])
        parameters = sum(weight.numel() for weight in ConditionalUNet(**config['model']).parameters())
        if abs(parameters / baseline_parameters - 1) > .04:
            raise ValueError(f'Patch {size} exceeds the declared 4% parameter tolerance')
        encoded = yaml.safe_dump(config, sort_keys=False)
        if config_path.exists() and config_path.read_text(encoding='utf-8') != encoded:
            raise ValueError(f'Existing patch config differs: {config_path}')
        config_path.write_text(encoded, encoding='utf-8', newline='\n')
        inputs.append(config_path)
        directory = Path('checkpoints') / f'{name}_seed17_u030000_63bc3fce35'
        variants.append({'name': name, 'parameters': parameters, 'patch_size': size,
                         'checkpoint_directory': str(directory), 'config': str(config_path),
                         'command': [sys.executable, '-u', '-m', 'src.training.train',
                                     '--config', str(config_path), '--manifest', str(MANIFEST),
                                     '--validation-manifest', str(HOLDOUT), '--seed', '17',
                                     '--out', str(directory)]})
    write_json(PLAN, {'created': timestamp(), 'inputs': {str(path): sha256(path) for path in inputs},
                      'variants': variants, 'baseline_parameters': baseline_parameters,
                      'maximum_relative_parameter_difference': .04,
                      'sampler': SAMPLER, 'updates': UPDATES, 'selection_seeds': SEEDS,
                      'n_per_class': 1000,
                      'scope': 'Whole-image epsilon diffusion with lossless patch packing and '
                               'a shared conditioned pixel-resolution detail branch.',
                      'rule': 'Evaluate all four checkpoints at seed 0. Advance at most one valid '
                              'improvement over the v002 30k anchor by tier or normalized margin. '
                              'Confirm an improvement against the reference across all three seeds.'})


def verified_checkpoint(directory, update):
    path = directory / f'update_{update:06d}.pt'
    row = read(path.with_suffix('.json'))
    if (sha256(path), update, sha256(MANIFEST)) != (row['sha256'], row['update'], row['manifest_sha256']):
        raise ValueError(f'Checkpoint receipt differs: {path}')
    return path


def train(variant, updates):
    directory = Path(variant['checkpoint_directory'])
    if not (directory / 'completed.json').exists():
        command = variant['command']
        if (directory / 'run.json').exists():
            latest = max(directory.glob('update_*.json'), default=None)
            if latest is None:
                raise ValueError(f'Interrupted run has no verified checkpoint: {directory}')
            command = command + ['--resume', str(verified_checkpoint(directory, read(latest)['update']))]
        print(json.dumps({'event': 'training_started', 'variant': variant['name'], 'command': command}),
              flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    completed = read(directory / 'completed.json')
    initial = read(next(directory.glob('segment_000000_*.json')))
    if (completed['last_update'], completed['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['parameters'], initial['config_sha256'], initial['validation']['sha256']) != (
            updates, sha256(MANIFEST), None, 17, variant['parameters'], sha256(variant['config']), sha256(HOLDOUT)):
        raise ValueError(f'Training receipts do not match the matched patch run: {directory}')
    print(json.dumps({'event': 'training_completed', 'variant': variant['name'], **completed}), flush=True)
    return directory


def execute(plan):
    verify_starter()
    for name, digest in plan['inputs'].items():
        if sha256(name) != digest:
            raise ValueError(f'Patch comparison input changed: {name}')
    reference = read(REFERENCE)
    reference_seed = reference['evidence'][0]
    anchor = validated_result(reference_seed['path'], 0, reference['exported'])
    baseline = {'score': score_results([anchor])}
    candidates = []
    plan_hash = sha256(PLAN)
    for variant in plan['variants']:
        directory = train(variant, plan['updates'][-1])
        for update in plan['updates']:
            name = f"{variant['name']}_u{update:06d}"
            candidate = evaluate(verified_checkpoint(directory, update), MANIFEST,
                                 Path('private_build/patch_grid_v002') / name, OUTPUT / name,
                                 n_per_class=plan['n_per_class'], **plan['sampler'])
            candidates.append(candidate | {'name': name, 'variant': variant['name'],
                                           'patch_size': variant['patch_size']})
            write_json(OUTPUT / 'screening.json', {'plan_sha256': plan_hash,
                                                 'reference': reference_seed, 'screened': candidates})
            print(json.dumps({'event': 'official_evaluation_completed', 'name': name,
                              'evidence': candidate['evidence']}), flush=True)
    top = max((row for row in candidates if quality_improved(baseline, row)),
              key=lambda row: row['score'], default=None)
    selected = None
    if top:
        selected = evaluate(top['checkpoint'], MANIFEST, top['exported'], OUTPUT / top['name'],
                            seeds=plan['selection_seeds'], n_per_class=plan['n_per_class'], **plan['sampler'])
        print(json.dumps({'event': 'official_selection_completed', 'name': top['name'],
                          'evidence': selected['evidence']}), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUTPUT / 'comparison.json', {'plan_sha256': plan_hash, 'candidate': selected,
                                           'improves_reference': improved})
    print(json.dumps({'event': 'comparison_completed', 'improves_reference': improved}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        execute(read(PLAN))

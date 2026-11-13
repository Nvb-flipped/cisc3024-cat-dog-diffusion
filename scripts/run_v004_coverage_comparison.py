"""Measure the first reviewed v004 expansion with the v002 pixel U-Net recipe."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


OLD = Path('manifests/v002.csv')
NEW = Path('manifests/v004.csv')
HOLDOUT = NEW.with_suffix('.validation.csv')
REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
CONFIG = Path('configs/runs/86c3c843eb/capacity_32_seed17_u030000.yaml')
CACHE = Path('results/curation/v004_cache_parity.json')
PLAN = Path('results/plans/v004_coverage.json')
RUN = Path('checkpoints/capacity_32_seed17_u030000_200d0f7e3f')
EXPORTS = Path('private_build/v004_coverage')
OUTPUT = Path('results/experiments/v004_coverage')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return {tuple(row.values()) for row in csv.DictReader(stream)}


def prepare():
    if PLAN.exists():
        raise FileExistsError(f'Preserve the declared experiment: {PLAN}')
    old_rows, new_rows = manifest_rows(OLD), manifest_rows(NEW)
    if len(old_rows) != 1197 or len(new_rows) != 1221 or not old_rows < new_rows or \
            sha256(HOLDOUT) != sha256(OLD.with_suffix('.validation.csv')):
        raise ValueError('v004 must retain the exact v002 training rows and holdout')
    checks = read(CACHE)['checks']
    if [row['manifest_sha256'] for row in checks] != [sha256(NEW), sha256(HOLDOUT)] or \
            not all(row['pixels_equal'] and row['labels_equal'] for row in checks):
        raise ValueError('v004 cache differs from the official reconstruction')
    sources = [OLD, OLD.with_suffix('.validation.csv'), NEW, HOLDOUT, NEW.with_suffix('.provenance.json'),
               REFERENCE, CONFIG, CACHE, Path(__file__).relative_to(ROOT)]
    sources += [Path(name) for name in ('src/training/train.py', 'src/training/step.py',
                'src/training/data.py', 'src/training/validation.py', 'src/training/optimize.py',
                'submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                'src/release/export.py', 'src/evaluation/run.py', 'src/evaluation/select.py',
                'src/evaluation/sweep.py')]
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
               '--manifest', str(NEW), '--validation-manifest', str(HOLDOUT), '--seed', '17',
               '--updates', '30000', '--out', str(RUN)]
    write_json(PLAN, {'created': timestamp(), 'inputs': {str(path): sha256(path) for path in sources},
                      'command': command, 'added_crops': len(new_rows - old_rows), 'updates': 30000,
                      'checkpoint_updates': [28000, 30000], 'selection_seeds': [0, 101, 202],
                      'sampler': {'weights': 'ema', 'steps': 50, 'guidance': 2.5},
                      'rule': '30k gives the primary matched estimate. Compare 28k to the same 30k anchor '
                              'for candidate selection only. Advance at most one valid seed-0 improvement '
                              'by the declared tier or margin rule, and require the three-seed improvement.'})


def checkpoint(update):
    path = RUN / f'update_{update:06d}.pt'
    receipt = read(path.with_suffix('.json'))
    if (sha256(path), update, sha256(NEW)) != (receipt['sha256'], receipt['update'], receipt['manifest_sha256']):
        raise ValueError(f'Checkpoint receipt differs: {path}')
    return path


def execute(plan):
    verify_starter()
    for name, expected in plan['inputs'].items():
        if sha256(name) != expected:
            raise ValueError(f'v004 comparison input changed: {name}')
    if not (RUN / 'completed.json').exists():
        command = plan['command']
        if (RUN / 'run.json').exists():
            latest = max(RUN.glob('update_*.json'), default=None)
            if latest is None:
                raise ValueError('Interrupted training has no verified checkpoint')
            command = command + ['--resume', str(checkpoint(read(latest)['update']))]
        print(json.dumps({'event': 'training_started', 'command': command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    completed = read(RUN / 'completed.json')
    initial = read(next(RUN.glob('segment_000000_*.json')))
    if (completed['last_update'], completed['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            plan['updates'], sha256(NEW), None, 17, sha256(CONFIG), sha256(HOLDOUT)):
        raise ValueError('Training receipts do not match the fresh v004 experiment')
    print(json.dumps({'event': 'training_completed', **completed}), flush=True)
    reference = read(REFERENCE)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    baseline = {'score': score_results([anchor])}
    plan_hash = sha256(PLAN)
    candidates = []
    for update in plan['checkpoint_updates']:
        name = f'update_{update:06d}'
        row = evaluate(checkpoint(update), NEW, EXPORTS / name, OUTPUT / name, **plan['sampler'])
        candidates.append(row)
        write_json(OUTPUT / 'screening.json', {'plan_sha256': plan_hash, 'screened': candidates})
        print(json.dumps({'event': 'official_evaluation_completed', 'update': update,
                          'evidence': row['evidence']}), flush=True)
    top = max((row for row in candidates if quality_improved(baseline, row)),
              key=lambda row: row['score'], default=None)
    candidate = None
    if top:
        name = f"update_{top['update']:06d}"
        candidate = evaluate(top['checkpoint'], NEW, top['exported'], OUTPUT / name,
                             seeds=plan['selection_seeds'], **plan['sampler'])
        print(json.dumps({'event': 'official_selection_completed', 'evidence': candidate['evidence']}),
              flush=True)
    improved = candidate is not None and quality_improved(reference, candidate)
    write_json(OUTPUT / 'comparison.json', {'plan_sha256': plan_hash, 'candidate': candidate,
                                           'improves_reference': improved})
    print(json.dumps({'event': 'comparison_completed', 'improves_reference': improved}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        execute(read(PLAN))

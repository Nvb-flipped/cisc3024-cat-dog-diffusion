"""Compare a frozen, additive pool expansion with the v002 pixel U-Net."""

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


REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
PARENT = Path('manifests/v002.csv')
CONFIG = Path('configs/runs/86c3c843eb/capacity_32_seed17_u030000.yaml')
UPDATES = (28000, 30000)
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return {tuple(row.values()) for row in csv.DictReader(stream)}


def prepare(plan_path, manifest, holdout, run):
    if plan_path.exists():
        raise FileExistsError(plan_path)
    parent_holdout = PARENT.with_suffix('.validation.csv')
    base, expanded = rows(PARENT), rows(manifest)
    if not base < expanded or sha256(holdout) != sha256(parent_holdout):
        raise ValueError('Coverage comparison must retain every exact parent crop and the same holdout')
    cache = Path(f'results/curation/{manifest.stem}_cache_parity.json')
    checks = read(cache)['checks']
    if [(row['manifest_sha256'], row['rows']) for row in checks] != [
            (sha256(manifest), len(expanded)), (sha256(holdout), len(rows(holdout)))
    ] or not all(row['pixels_equal'] and row['labels_equal'] for row in checks):
        raise ValueError('Cache verification differs from the official reconstruction')
    sources = [PARENT, parent_holdout, manifest, holdout, manifest.with_suffix('.provenance.json'),
               REFERENCE, CONFIG, cache, Path(__file__).relative_to(ROOT)]
    sources.extend(map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                              'src/training/validation.py', 'src/training/optimize.py', 'submission/model.py',
                              'submission/diffusion.py', 'submission/generate.py', 'src/release/export.py',
                              'src/evaluation/run.py', 'src/evaluation/select.py', 'src/evaluation/sweep.py')))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
               '--manifest', str(manifest), '--validation-manifest', str(holdout), '--seed', '17',
               '--updates', str(UPDATES[-1]), '--out', str(run)]
    write_json(plan_path, dict(created=timestamp(), inputs={str(path): sha256(path) for path in sources},
                               command=command, added_crops=len(expanded) - len(base),
                               checkpoint_updates=list(UPDATES), selection_seeds=[0, 101, 202], sampler=SAMPLER,
                               rule='30k is the matched estimate. 28k is a checkpoint screen. Advance at most one '
                                    'candidate by the existing hard gate and score or margin improvement rule.'))


def verified_checkpoint(run, update, manifest_hash):
    path = run / f'update_{update:06d}.pt'
    receipt = read(path.with_suffix('.json'))
    if (receipt['sha256'], receipt['update'], receipt['manifest_sha256']) != (sha256(path), update, manifest_hash):
        raise ValueError(f'Checkpoint does not match its receipt: {path}')
    return path


def execute(plan_path, manifest, holdout, run, exports, output):
    verify_starter()
    plan = read(plan_path)
    for name, expected in plan['inputs'].items():
        if sha256(name) != expected:
            raise ValueError(f'Comparison input changed: {name}')
    manifest_hash = sha256(manifest)
    completed_path = run / 'completed.json'
    if not completed_path.exists():
        command = plan['command']
        if (run / 'run.json').exists():
            receipts = sorted(run.glob('update_*.json'))
            if not receipts:
                raise ValueError('Interrupted training has no verified checkpoint')
            command = command + ['--resume', str(verified_checkpoint(run, read(receipts[-1])['update'], manifest_hash))]
        print(json.dumps(dict(event='training_started', command=command)), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    completed = read(completed_path)
    initial_path, = run.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (completed['last_update'], completed['manifest_sha256'], initial['parent_checkpoint_sha256'],
        initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            UPDATES[-1], manifest_hash, None, 17, sha256(CONFIG), sha256(holdout)):
        raise ValueError('Training receipts do not match the declared experiment')
    print(json.dumps(dict(event='training_completed', **completed)), flush=True)
    reference = read(REFERENCE)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    baseline = {'score': score_results([anchor])}
    plan_hash = sha256(plan_path)
    candidates = []
    for update in plan['checkpoint_updates']:
        name = f'update_{update:06d}'
        checkpoint = verified_checkpoint(run, update, manifest_hash)
        row = evaluate(checkpoint, manifest, exports / name, output / name, **plan['sampler'])
        candidates.append(row)
        write_json(output / 'screening.json', dict(plan_sha256=plan_hash, screened=candidates))
        print(json.dumps(dict(event='official_evaluation_completed', update=update, evidence=row['evidence'])), flush=True)
    top = max((row for row in candidates if quality_improved(baseline, row)), key=lambda row: row['score'], default=None)
    candidate = None
    if top:
        name = f"update_{top['update']:06d}"
        candidate = evaluate(top['checkpoint'], manifest, top['exported'], output / name,
                             seeds=plan['selection_seeds'], **plan['sampler'])
        print(json.dumps(dict(event='official_selection_completed', evidence=candidate['evidence'][1:])), flush=True)
    improved = candidate is not None and quality_improved(reference, candidate)
    write_json(output / 'comparison.json', dict(plan_sha256=plan_hash, candidate=candidate, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    manifest = args.manifest
    holdout = manifest.with_suffix('.validation.csv')
    name = f'{manifest.stem}_coverage'
    plan = Path(f'results/plans/{name}.json')
    run = Path(f'checkpoints/capacity_32_seed17_u030000_{sha256(manifest)[:10]}')
    if args.prepare:
        prepare(plan, manifest, holdout, run)
    else:
        execute(plan, manifest, holdout, run, Path('private_build') / name, Path('results/experiments') / name)


if __name__ == '__main__':
    main()

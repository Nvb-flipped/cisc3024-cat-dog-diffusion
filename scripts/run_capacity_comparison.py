"""Screen one intermediate-width pixel U-Net on the exact reviewed v006 pool."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import REFERENCE, PARENT, SAMPLER, read, rows, verified_checkpoint
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


CONTROL_RUN = Path('checkpoints/capacity_32_seed17_u030000_123aac91c9')


def prepare(plan_path, manifest, holdout, run, config, updates):
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
               REFERENCE, config, cache, Path(__file__).relative_to(ROOT), Path('scripts/run_coverage_comparison.py'),
               verified_checkpoint(CONTROL_RUN, updates[-1], sha256(manifest))]
    sources.extend(map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                              'src/training/validation.py', 'src/training/lineage.py', 'src/training/optimize.py',
                              'src/curation/pool_fingerprint.py', 'submission/model.py',
                              'submission/diffusion.py', 'submission/generate.py', 'src/release/export.py',
                              'src/evaluation/run.py', 'src/evaluation/select.py', 'src/evaluation/sweep.py')))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config),
               '--manifest', str(manifest), '--validation-manifest', str(holdout), '--seed', '17',
               '--updates', str(updates[-1]), '--out', str(run)]
    write_json(plan_path, dict(created=timestamp(), inputs={str(path): sha256(path) for path in sources},
                               command=command, added_crops=len(expanded) - len(base),
                               checkpoint_updates=list(updates), selection_seeds=[0, 101, 202], sampler=SAMPLER,
                               rule='Screen 22k, 24k and 26k and advance at most one checkpoint by the existing '
                                    'quality rule. Compare with the exact same-width-32 v006 fit at matched updates '
                                    'for the capacity effect and with the v002 export for promotion.'))


def execute(plan_path, manifest, holdout, run, exports, output):
    verify_starter()
    plan = read(plan_path)
    for name, expected in plan['inputs'].items():
        if sha256(name) != expected:
            raise ValueError(f'Comparison input changed: {name}')
    manifest_hash = sha256(manifest)
    completed_path = run / 'completed.json'
    if (output / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison receipt')
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
    command = plan['command']
    expected_config = Path(command[command.index('--config') + 1])
    if (completed['last_update'], completed['manifest_sha256'], initial['parent_checkpoint_sha256'],
        initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            plan['checkpoint_updates'][-1], manifest_hash, None, 17, sha256(expected_config), sha256(holdout)):
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
    control_name = 'width32_u026000'
    control = evaluate(verified_checkpoint(CONTROL_RUN, plan['checkpoint_updates'][-1], manifest_hash),
                       manifest, exports / control_name, output / control_name, **plan['sampler'])
    print(json.dumps(dict(event='official_control_completed', evidence=control['evidence'])), flush=True)
    top = max((row for row in candidates if quality_improved(baseline, row)), key=lambda row: row['score'], default=None)
    candidate = None
    if top:
        name = f"update_{top['update']:06d}"
        candidate = evaluate(top['checkpoint'], manifest, top['exported'], output / name,
                             seeds=plan['selection_seeds'], **plan['sampler'])
        print(json.dumps(dict(event='official_selection_completed', evidence=candidate['evidence'][1:])), flush=True)
    improved = candidate is not None and quality_improved(reference, candidate)
    write_json(output / 'comparison.json', dict(plan_sha256=plan_hash, matched_width32_control=control,
                                                candidate=candidate, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    manifest = Path('manifests/v006.csv')
    holdout = manifest.with_suffix('.validation.csv')
    name = f'{manifest.stem}_width40'
    plan = Path(f'results/plans/{name}.json')
    config = Path('configs/runs/v006_width40.yaml')
    updates = (22000, 24000, 26000)
    run = Path(f'checkpoints/capacity_40_seed17_u026000_{sha256(manifest)[:10]}')
    if args.prepare:
        prepare(plan, manifest, holdout, run, config, updates)
    else:
        execute(plan, manifest, holdout, run, Path('private_build') / name, Path('results/experiments') / name)


if __name__ == '__main__':
    main()

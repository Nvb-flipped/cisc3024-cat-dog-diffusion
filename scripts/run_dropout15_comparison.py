"""Test one moderate dropout increase against exact v006 and the leading sampler."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import SAMPLER, read, verified_checkpoint
from scripts.setup_inputs import verify_starter
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


MANIFEST = Path('manifests/v006.csv')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
CONFIG = Path('configs/runs/v006_dropout15.yaml')
CONTROL_CONFIG = Path('configs/runs/86c3c843eb/capacity_32_seed17_u030000.yaml')
CONTROL = Path('results/experiments/v006_coverage/screening.json')
REFERENCE = Path('results/experiments/rescale_followup/comparison.json')
PLAN = Path('results/plans/v006_dropout15.json')
RUN = Path('checkpoints/v006_dropout15_seed17')
OUT = Path('results/experiments/v006_dropout15')
EXPORTS = Path('private_build/v006_dropout15')
UPDATES = (28000, 30000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    comparison_config = read_config(CONFIG)
    comparison_config['model']['dropout'] = .1
    control_config = read_config(CONTROL_CONFIG)
    control_config['training']['updates'] = UPDATES[-1]
    if comparison_config != control_config:
        raise ValueError('Dropout and the recorded target update must be the only config differences')
    paths = (MANIFEST, MANIFEST.with_suffix('.provenance.json'), HOLDOUT, CONFIG, CONTROL_CONFIG, CONTROL,
             REFERENCE, Path(__file__), Path('results/curation/v006_cache_parity.json'),
             *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                         'src/training/lineage.py', 'src/training/validation.py', 'src/training/optimize.py',
                         'submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'scripts/run_coverage_comparison.py')))
    inputs = {str(path): sha256(path) for path in paths}
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
                   '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
                   '--updates', str(UPDATES[-1]), '--out', str(RUN)]
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, command=command, sampler=SAMPLER,
                              checkpoints=list(UPDATES), seeds=[0, 101, 202],
                              rule='Compare at matched updates with dropout 0.10. Advance only one valid '
                                   'candidate that improves the leading v002 rescale 0.25 result.'))
        return
    plan = read(PLAN)
    if (plan['inputs'], plan['sampler'], plan['checkpoints']) != (inputs, SAMPLER, list(UPDATES)):
        raise ValueError('Frozen dropout comparison inputs changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison')
    verify_starter()
    manifest_hash = sha256(MANIFEST)
    if not (RUN / 'completed.json').exists():
        command = plan['command']
        if (RUN / 'run.json').exists():
            receipts = sorted(RUN.glob('update_*.json'))
            if not receipts:
                raise ValueError('No verified checkpoint for this interrupted run')
            checkpoint = verified_checkpoint(RUN, read(receipts[-1])['update'], manifest_hash)
            command = command + ['--resume', str(checkpoint)]
        subprocess.run(command, cwd=ROOT, check=True)
    completion = read(RUN / 'completed.json')
    initial_path, = RUN.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (completion['last_update'], completion['manifest_sha256'], initial['parent_checkpoint_sha256'],
        initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            UPDATES[-1], manifest_hash, None, 17, sha256(CONFIG), sha256(HOLDOUT)):
        raise ValueError('Training lineage differs from the frozen comparison')
    print(json.dumps(dict(event='training_completed', **completion)), flush=True)
    reference = read(REFERENCE)['candidate']
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    controls = {row['update']: row for row in read(CONTROL)['screened']}
    trials = []
    for update in UPDATES:
        name = f'update_{update:06d}'
        control = controls[update]
        matched = validated_result(control['evidence'][0]['path'], 0, control['exported'])
        row = evaluate(verified_checkpoint(RUN, update, manifest_hash), MANIFEST,
                       EXPORTS / name, OUT / name, **SAMPLER)
        trials.append(row | dict(matched_dropout10=matched['metrics']))
        write_json(OUT / 'screening.json', dict(plan_sha256=sha256(PLAN), screened=trials))
        print(json.dumps(dict(event='official_evaluation_completed', update=update, evidence=row['evidence'])), flush=True)
    best = max((row for row in trials if quality_improved({'score': score_results([anchor])}, row)),
               key=lambda row: row['score'], default=None)
    selected = None
    if best:
        selected = evaluate(best['checkpoint'], MANIFEST, best['exported'], OUT / f"update_{best['update']:06d}",
                            seeds=plan['seeds'], **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

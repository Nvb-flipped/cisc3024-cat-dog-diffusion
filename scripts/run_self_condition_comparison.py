"""Compare epsilon self-conditioning with the matched v006 control."""

import argparse
from pathlib import Path
import json
import subprocess
import sys

from scripts.run_coverage_comparison import REFERENCE, SAMPLER, read, verified_checkpoint
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


NAME = 'v006_self_condition'
MANIFEST = Path('manifests/v006.csv')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
CONFIG = Path('configs/runs/v006_self_condition.yaml')
PLAN = Path(f'results/plans/{NAME}.json')
RUN = Path(f'checkpoints/{NAME}_seed17')
DIAGNOSTIC = Path('results/diagnostics/v006_self_condition/result.json')
CONTROL = Path('results/experiments/v006_coverage/screening.json')


def execute(plan, output):
    verify_starter()
    for name, digest in plan['inputs'].items():
        if sha256(name) != digest:
            raise ValueError(f'Frozen self-conditioning input changed: {name}')
    completion = RUN / 'completed.json'
    manifest_hash = sha256(MANIFEST)
    if not completion.exists():
        command = plan['command']
        if (RUN / 'run.json').exists():
            checkpoints = sorted(RUN.glob('update_*.json'))
            if not checkpoints:
                raise ValueError('Interrupted run has no verified checkpoint')
            prior = verified_checkpoint(RUN, read(checkpoints[-1])['update'], manifest_hash)
            command = command + ['--resume', str(prior)]
        subprocess.run(command, cwd=ROOT, check=True)
    finished = read(completion)
    initial_path, = RUN.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (finished['last_update'], finished['manifest_sha256'], initial['parent_checkpoint_sha256'],
        initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            30000, manifest_hash, None, 17, sha256(CONFIG), sha256(HOLDOUT)):
        raise ValueError('Training receipts differ from the frozen self-conditioning run')
    print(json.dumps(dict(event='training_completed', **finished)), flush=True)
    exports = Path('private_build') / NAME
    trials = []
    for update in plan['checkpoint_updates']:
        name = f'update_{update:06d}'
        trial = evaluate(verified_checkpoint(RUN, update, manifest_hash), MANIFEST,
                         exports / name, output / name, **plan['sampler'])
        trials.append(trial)
        write_json(output / 'screening.json', dict(plan_sha256=sha256(PLAN), screened=trials))
        print(json.dumps(dict(event='official_evaluation_completed', update=update, evidence=trial['evidence'])), flush=True)
    reference = read(REFERENCE)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    best = max((row for row in trials if quality_improved({'score': score_results([anchor])}, row)),
               key=lambda row: row['score'], default=None)
    selected = None
    if best:
        selected = evaluate(best['checkpoint'], MANIFEST, best['exported'], output / f"update_{best['update']:06d}",
                            seeds=plan['selection_seeds'], **plan['sampler'])
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(output / 'comparison.json', dict(plan_sha256=sha256(PLAN), candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    diagnostic = read(DIAGNOSTIC)
    if not diagnostic['passed'] or not diagnostic['diagnostic_only'] or \
            diagnostic['updates'] != 2000 or not diagnostic['model'].get('self_condition'):
        raise ValueError('A passing fixed 2,000-update self-conditioning diagnostic is required')
    if any(sha256(name) != digest for name, digest in diagnostic['self_condition_sources'].items()):
        raise ValueError('Self-conditioning code changed after its integration diagnostic')
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        paths = (MANIFEST, MANIFEST.with_suffix('.provenance.json'), HOLDOUT, CONFIG, DIAGNOSTIC, CONTROL,
                 REFERENCE, Path(__file__), Path('results/curation/v006_cache_parity.json'),
                 *map(Path, ('scripts/run_coverage_comparison.py', 'src/training/diagnostics.py',
                             'src/training/train.py', 'src/training/data.py', 'src/training/step.py',
                             'src/training/lineage.py', 'src/training/validation.py',
                             'submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                             'src/training/optimize.py', 'src/release/export.py', 'src/evaluation/run.py')))
        command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
                   '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
                   '--updates', '30000', '--out', str(RUN)]
        write_json(PLAN, dict(created=timestamp(), inputs={str(path): sha256(path) for path in paths},
                              command=command, sampler=SAMPLER, checkpoint_updates=[28000, 30000],
                              selection_seeds=[0, 101, 202],
                              rule='Compare with v006 at the same updates; report extra forward compute. '
                                   'Promote only a valid v002 improvement under the existing three-seed rule.',
                              source='https://arxiv.org/abs/2208.04202'))
        return
    output = Path('results/experiments') / NAME
    if (output / 'comparison.json').exists():
        raise FileExistsError('Preserve the existing comparison')
    execute(read(PLAN), output)


if __name__ == '__main__':
    main()

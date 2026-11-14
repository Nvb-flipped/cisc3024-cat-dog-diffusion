"""Train and screen one coarse-loss variant of the frozen v002 reference."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import read, verified_checkpoint
from scripts.run_v002_dpmpp import MANIFEST, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


NAME = 'v002_coarse_epsilon'
PLAN = Path(f'results/plans/{NAME}.json')
CONFIG = Path(f'configs/runs/{NAME}.yaml')
RUN = Path(f'checkpoints/{NAME}_seed17')
OUT = Path(f'results/experiments/{NAME}')
EXPORTS = Path(f'private_build/{NAME}')
DIAGNOSTIC = Path(f'results/diagnostics/{NAME}/result.json')
REF = Path('results/experiments/rescale_followup/comparison.json')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
UPDATES = [28000, 30000]
SEEDS = [0, 101, 202]
WEIGHT = .5
RESCALE = .25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REF)['candidate']
    control = Path(reference['checkpoint']).parent / 'run.json'
    config = read_config(CONFIG)
    if config['training'].pop('coarse_epsilon_weight') != WEIGHT or config != read(control)['config']:
        raise ValueError('Coarse epsilon loss must be the only training configuration change')
    diagnostic = read(DIAGNOSTIC)
    if (diagnostic['coarse_epsilon_weight'], diagnostic['updates'], diagnostic['passed'],
            diagnostic['diagnostic_only']) != (WEIGHT, 2000, True, True):
        raise ValueError('The exact fixed diagnostic must pass before training')
    if any(sha256(path) != digest for path, digest in diagnostic['coarse_epsilon_sources'].items()):
        raise ValueError('Loss implementation changed after the diagnostic')
    paths = (REF, control, MANIFEST, MANIFEST.with_suffix('.provenance.json'), HOLDOUT,
             CONFIG, DIAGNOSTIC, Path(__file__), *map(Path, (
             'src/training/train.py', 'src/training/step.py', 'src/training/diagnostics.py',
             'src/training/data.py', 'src/training/validation.py', 'src/training/lineage.py',
             'submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
             'src/release/export.py', 'src/training/optimize.py', 'src/evaluation/run.py')))
    inputs = {str(path): sha256(path) for path in paths}
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
               '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
               '--updates', str(UPDATES[-1]), '--out', str(RUN)]
    spec = dict(inputs=inputs, command=command, sampler=SAMPLER | {'guidance_rescale': RESCALE},
                checkpoint_updates=UPDATES, selection_seeds=SEEDS)
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, spec | dict(created=timestamp(), coarse_weight=WEIGHT,
                   source='https://proceedings.mlr.press/v202/hoogeboom23a.html',
                   rule='Single training change. Compare 30k to exact v002 rescale .25. Screen 28k too. '
                        'Advance at most one seed-0 improvement to seeds 101 and 202.'))
        return
    plan = read(PLAN)
    if any(plan[key] != value for key, value in spec.items()):
        raise ValueError('Frozen coarse epsilon comparison changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve completed evidence')
    verify_starter()
    manifest_hash = sha256(MANIFEST)
    completion = RUN / 'completed.json'
    if not completion.exists():
        if (RUN / 'run.json').exists():
            last = sorted(RUN.glob('update_*.json'))[-1]
            command += ['--resume', str(verified_checkpoint(RUN, read(last)['update'], manifest_hash))]
        subprocess.run(command, cwd=ROOT, check=True)
    finished = read(completion)
    initial_path, = RUN.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (finished['last_update'], finished['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            UPDATES[-1], manifest_hash, None, 17, sha256(CONFIG), sha256(HOLDOUT)):
        raise ValueError('Training receipts differ from the declared run')
    print(json.dumps(dict(event='training_completed', **finished)), flush=True)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    anchor_score = {'score': score_results([anchor])}
    trials = []
    for update in UPDATES:
        name = f'update_{update:06d}'
        checkpoint = verified_checkpoint(RUN, update, manifest_hash)
        export, folder = EXPORTS / name, OUT / name
        if not export.exists():
            export_checkpoint(checkpoint, MANIFEST, export, **plan['sampler'])
        if read(export / 'inference.json')['sampling'].get('guidance_rescale') != RESCALE:
            raise ValueError('Coarse epsilon export must use the same CFG rescale as its reference')
        receipt = folder / 'interface.json'
        if not receipt.exists():
            write_json(receipt, verify_interface(export, export) | {'solver': 'ddim', 'guidance_rescale': RESCALE})
        trial = evaluate(checkpoint, MANIFEST, export, folder, **SAMPLER)
        trials.append(trial)
        write_json(OUT / 'screening.json', dict(plan_sha256=sha256(PLAN), screened=trials))
        print(json.dumps(dict(event='official_evaluation_completed', update=update, evidence=trial['evidence'])), flush=True)
    best = max((trial for trial in trials if quality_improved(anchor_score, trial)),
               key=lambda trial: trial['score'], default=None)
    selected = None
    if best:
        selected = evaluate(best['checkpoint'], MANIFEST, best['exported'], OUT / f"update_{best['update']:06d}",
                            seeds=SEEDS, **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

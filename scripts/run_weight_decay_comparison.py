"""Fit one v002 AdamW weight-decay challenger with the same exact approved crops."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import read, verified_checkpoint
from scripts.run_v002_dpmpp import MANIFEST, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


NAME = 'v002_weight_decay003'
PLAN = Path(f'results/plans/{NAME}.json')
CONFIG = Path(f'configs/runs/{NAME}.json')
RUN = Path(f'checkpoints/{NAME}_seed17')
OUT = Path(f'results/experiments/{NAME}')
EXPORT = Path(f'private_build/{NAME}')
REF = Path('results/experiments/rescale_followup/comparison.json')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
RESCALE = .25
DECAY = .03
UPDATES = [28000, 30000]
SEEDS = [0, 101, 202]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REF)['candidate']
    control_path = Path(reference['checkpoint']).parent / 'run.json'
    config = read(control_path)['config']
    config['training']['weight_decay'] = DECAY
    if args.prepare:
        if PLAN.exists() or CONFIG.exists():
            raise FileExistsError('Preserve the existing comparison plan or config')
        write_json(CONFIG, config)
    if read(CONFIG) != config:
        raise ValueError('Weight decay must be the only training configuration change')
    paths = (REF, control_path, MANIFEST, MANIFEST.with_suffix('.provenance.json'), HOLDOUT,
             CONFIG, Path(__file__), *map(Path, (
             'src/training/train.py', 'src/training/step.py', 'src/training/data.py', 'src/training/lineage.py',
             'src/training/validation.py', 'src/curation/pool_fingerprint.py', 'submission/model.py',
             'submission/diffusion.py', 'submission/generate.py', 'src/release/export.py',
             'src/training/optimize.py', 'src/evaluation/run.py')))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(CONFIG),
               '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
               '--updates', str(UPDATES[-1]), '--out', str(RUN)]
    spec = dict(inputs={str(path): sha256(path) for path in paths}, command=command,
                sampler=SAMPLER | {'guidance_rescale': RESCALE}, checkpoint_updates=UPDATES, selection_seeds=SEEDS)
    if args.prepare:
        write_json(PLAN, spec | dict(created=timestamp(),
                   sources=['https://arxiv.org/abs/1711.05101',
                            'https://openaccess.thecvf.com/content/CVPR2023/supplemental/Bao_All_Are_Worth_CVPR_2023_supplemental.pdf',
                            'https://openaccess.thecvf.com/content/CVPR2025/papers/Sehwag_Stretching_Each_Dollar_Diffusion_Training_from_Scratch_on_a_Micro-Budget_CVPR_2025_paper.pdf'],
                   rule='One scratch fit, changing only AdamW weight decay from 0.0001 to 0.03. '
                        'The 30k result is the matched comparison with the current best v002 export. '
                        'Advance at most one valid seed-0 improvement from the two planned checkpoints.'))
        return
    plan = read(PLAN)
    if any(plan[key] != value for key, value in spec.items()):
        raise ValueError('Frozen weight-decay comparison changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve completed evidence')
    verify_starter()
    manifest_hash = sha256(MANIFEST)
    completion = RUN / 'completed.json'
    if not completion.exists():
        if (RUN / 'run.json').exists():
            latest = sorted(RUN.glob('update_*.json'))[-1]
            command += ['--resume', str(verified_checkpoint(RUN, read(latest)['update'], manifest_hash))]
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
        export, folder = EXPORT / name, OUT / name
        if not export.exists():
            export_checkpoint(checkpoint, MANIFEST, export, **plan['sampler'])
        if read(export / 'inference.json')['sampling'].get('guidance_rescale') != RESCALE:
            raise ValueError('Sampler does not match the v002 reference')
        if not (folder / 'interface.json').exists():
            write_json(folder / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
        trial = evaluate(checkpoint, MANIFEST, export, folder, **SAMPLER)
        trials.append(trial)
        write_json(OUT / 'screening.json', dict(plan_sha256=sha256(PLAN), screened=trials))
        print(json.dumps(dict(event='official_evaluation_completed', update=update, evidence=trial['evidence'])), flush=True)
    best = max((trial for trial in trials if quality_improved(anchor_score, trial)),
               key=lambda trial: trial['score'], default=None)
    selected = None
    if best:
        name = f"update_{best['update']:06d}"
        selected = evaluate(best['checkpoint'], MANIFEST, EXPORT / name, OUT / name, seeds=SEEDS, **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

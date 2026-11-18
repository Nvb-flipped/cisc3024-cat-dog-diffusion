"""Test a 2k continuation with the fixed v008 32k sampler and manifest."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.run_v008_coverage import HOLDOUT, MANIFEST, PLAN, REFERENCE, RUN
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate', 'confirm'))
    args = parser.parse_args()
    manifest_hash = sha256(MANIFEST)
    parent = verified_checkpoint(RUN, 32000, manifest_hash)
    name = 'v008_extension34k'
    run = Path('checkpoints') / f'{name}_seed17'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    control_path = Path('results/experiments/v008_coverage32k/screen.json')
    control = read(control_path)['candidate']
    reference = read(REFERENCE)['candidate']
    if control['checkpoint_sha256'] != sha256(parent):
        raise ValueError('Expected the official 32k control checkpoint')
    parent_plan = read(PLAN)
    if any(sha256(path) != digest for path, digest in parent_plan['inputs'].items()):
        raise ValueError('Parent recipe inputs changed')
    config = Path('configs/runs/v008_coverage32k.json')
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config),
               '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
               '--updates', '34000', '--resume', str(parent), '--out', str(run)]
    spec = dict(inputs={str(p): sha256(p) for p in (PLAN, control_path, REFERENCE, parent, Path(__file__))},
                sampler=parent_plan['sampler'], command=command, seed=0,
                rule='Use the v008 32k seed-0 copy-rate margin for one matched 2k continuation.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen continuation plan changed')
    verify_starter()
    complete = run / 'completed.json'
    if args.phase == 'train' and not complete.exists():
        subprocess.run(command, check=True)
    receipt = read(complete)
    segment = read(run / f"segment_{receipt['segment_id']}.json")
    if (receipt['last_update'], receipt['manifest_sha256'], segment['parent_checkpoint_sha256']) != (
            34000, manifest_hash, sha256(parent)):
        raise ValueError('Continuation receipts do not match the approved checkpoint')
    checkpoint = verified_checkpoint(run, 34000, manifest_hash)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if args.phase == 'train':
        return
    if not export.exists():
        export_checkpoint(checkpoint, MANIFEST, export, **spec['sampler'])
    if read(export / 'inference.json') != read(Path(control['exported']) / 'inference.json'):
        raise ValueError('Expected the same sampler as the 32k control')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    runtime = {k: spec['sampler'][k] for k in ('weights', 'steps', 'guidance')}
    screen = evaluate(checkpoint, MANIFEST, export, out, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and args.phase == 'confirm':
        selection = evaluate(checkpoint, MANIFEST, export, out, seeds=[0, 101, 202], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                   candidate=selection, improves_reference=quality_improved(reference, selection)))
        print([row['marks'] for row in selection['evidence']], flush=True)


if __name__ == '__main__':
    main()

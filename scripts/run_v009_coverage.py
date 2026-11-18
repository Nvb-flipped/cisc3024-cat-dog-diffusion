"""Run a matched 32k coverage comparison using the personally reviewed v009 rows."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import rows, verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.run_v008_coverage import MANIFEST as PARENT, PLAN as PARENT_PLAN, REFERENCE
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
    name = 'v009_coverage32k'
    manifest = Path('manifests/v009.csv')
    holdout = manifest.with_suffix('.validation.csv')
    cache = Path('results/curation/v009_cache_parity.json')
    run = Path('checkpoints') / f'{name}_seed17'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    control_path = Path('results/experiments/v008_coverage32k/screen.json')
    parent, expanded = rows(PARENT), rows(manifest)
    manifest_hash, holdout_hash = sha256(manifest), sha256(holdout)
    if not parent < expanded or holdout_hash != sha256(PARENT.with_suffix('.validation.csv')):
        raise ValueError('The comparison must retain the exact v008 rows and holdout')
    checks = read(cache)['checks']
    if ([(x['manifest_sha256'], x['rows']) for x in checks] !=
            [(manifest_hash, len(expanded)), (holdout_hash, len(rows(holdout)))]
            or not all(x['pixels_equal'] and x['labels_equal'] for x in checks)):
        raise ValueError('Expected exact official cache parity')
    old_plan = read(PARENT_PLAN)
    command = old_plan['command'].copy()
    for key, value in (('--manifest', manifest), ('--validation-manifest', holdout), ('--out', run)):
        command[command.index(key) + 1] = str(value)
    command[0] = sys.executable
    config = Path(command[command.index('--config') + 1])
    reused = [p for p in old_plan['inputs'] if Path(p).parts[0] in ('src', 'submission', 'configs')]
    if any(sha256(p) != old_plan['inputs'][p] for p in reused):
        raise ValueError('Matched v008 training or inference code changed')
    sources = (Path(__file__), PARENT_PLAN, PARENT, manifest, holdout,
               manifest.with_suffix('.provenance.json'), cache, REFERENCE, control_path, *reused)
    spec = dict(inputs={str(p): sha256(p) for p in sources}, command=command, sampler=old_plan['sampler'],
                added_crops=len(expanded) - len(parent), seeds=[0, 101, 202], n_per_class=1000)
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen v009 comparison inputs changed')
    verify_starter()
    complete = run / 'completed.json'
    if args.phase == 'train' and not complete.exists():
        receipts = sorted(run.glob('update_*.json'))
        if receipts:
            prior = verified_checkpoint(run, read(receipts[-1])['update'], manifest_hash)
            command += ['--resume', str(prior)]
        subprocess.run(command, check=True)
    receipt = read(complete)
    initial_path, = run.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (receipt['last_update'], receipt['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            32000, manifest_hash, None, 17, sha256(config), holdout_hash):
        raise ValueError('Lineage differs from the frozen scratch recipe')
    checkpoint = verified_checkpoint(run, 32000, manifest_hash)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if args.phase == 'train':
        return
    reference, control = read(REFERENCE)['candidate'], read(control_path)['candidate']
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json') != read(Path(control['exported']) / 'inference.json'):
        raise ValueError('Expected the exact v008 sampler')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    matched = validated_result(control['evidence'][0]['path'], 0, control['exported'])
    runtime = {k: spec['sampler'][k] for k in ('weights', 'steps', 'guidance')}
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances,
               improves_matched=quality_improved({'score': score_results([matched])}, screen)))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and args.phase == 'confirm':
        selection = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                   candidate=selection, improves_reference=quality_improved(reference, selection)))
        print([item['marks'] for item in selection['evidence']], flush=True)


if __name__ == '__main__':
    main()

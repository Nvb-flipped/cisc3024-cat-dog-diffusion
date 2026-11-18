"""Extend the frozen uniform v009 run by 4k updates and screen the exact sampler."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved

NAME = 'v009_extension36k'


def frozen_sources(source_hashes):
    archive = Path('private_build/v009_uniform_training_sources')
    stage = Path('private_build') / f'{NAME}_sources'
    extras = ('src/__init__.py', 'src/common.py', 'src/training/__init__.py',
              'src/curation/__init__.py', 'submission/__init__.py', 'submission/edm.py')
    staged_hashes = {}
    for name in (*source_hashes, *extras):
        local = Path(name)
        source = archive / name if (archive / name).is_file() else local
        expected = source_hashes.get(name, sha256(source))
        if sha256(source) != expected:
            raise ValueError(f'Original training source unavailable: {name}')
        target = stage / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if sha256(target) != expected:
            raise ValueError(f'Frozen training source changed: {name}')
        staged_hashes[str(target)] = expected
    return stage, staged_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate', 'confirm'))
    phase = parser.parse_args().phase
    verify_starter()
    manifest = Path('manifests/v009.csv')
    digest = sha256(manifest)
    original = Path('checkpoints/v009_coverage32k_seed17')
    parent = verified_checkpoint(original, 32000, digest)
    original_plan = Path('results/plans/v009_coverage32k.json')
    matched_path = Path('results/experiments/v009_coverage32k/screen.json')
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    control, reference = read(matched_path)['candidate'], read(reference_path)['candidate']
    if sha256(parent) != control['checkpoint_sha256']:
        raise ValueError('Continuation must start from the evaluated uniform v009 checkpoint')
    source_hashes = read(original / 'run.json')['source_code_sha256']
    stage, staged_hashes = frozen_sources(source_hashes)
    run = Path('checkpoints') / f'{NAME}_seed17'
    export = Path('private_build') / NAME
    out = Path('results/experiments') / NAME
    plan = Path('results/plans') / f'{NAME}.json'
    original_recipe = read(original_plan)
    command = original_recipe['command'].copy()
    for key, value in (('--updates', '36000'), ('--out', run)):
        command[command.index(key) + 1] = str(value)
    boot = ('import runpy,sys; from pathlib import Path; '
            'sys.path[:0]=[str(Path(sys.argv.pop(1)).resolve()), str(Path("starter_kit").resolve())]; '
            'runpy.run_module("src.training.train",run_name="__main__")')
    actual = [sys.executable, '-u', '-c', boot, str(stage), *command[4:], '--resume', str(parent)]
    sources = (Path(__file__), original_plan, original / 'run.json', matched_path, reference_path, manifest,
               manifest.with_suffix('.validation.csv'), manifest.with_suffix('.provenance.json'),
               parent, Path(command[command.index('--config') + 1]))
    spec = dict(inputs=staged_hashes | {str(p): sha256(p) for p in sources},
                command=actual, sampler=original_recipe['sampler'], seeds=[0, 101, 202], n_per_class=1000,
                rule='Use the 2.10% memorization margin for one 4k continuation with exact prior trainer bytes.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen continuation inputs changed')
    completion = run / 'completed.json'
    if phase == 'train' and not completion.exists():
        saved = sorted(run.glob('update_*.json'))
        if saved:
            receipt = read(saved[-1])
            actual[-1] = str(verified_checkpoint(run, receipt['update'], digest))
        subprocess.run(actual, cwd=ROOT, check=True)
    receipt = read(completion)
    first, = run.glob('segment_032000_*.json')
    if (receipt['last_update'], receipt['manifest_sha256'], read(first)['parent_checkpoint_sha256'],
            read(first)['source_code_sha256']) != (36000, digest, sha256(parent), source_hashes):
        raise ValueError('Continuation lineage or trainer bytes differ')
    checkpoint = verified_checkpoint(run, 36000, digest)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if phase == 'train':
        return
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json') != read(Path(control['exported']) / 'inference.json'):
        raise ValueError('Expected the exact 32k control sampler')
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
    if advances and phase == 'confirm':
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selected,
                   improves_reference=quality_improved(reference, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

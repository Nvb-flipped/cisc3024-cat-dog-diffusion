"""Extend the exact attention fit from 34k to 35k and screen its matched sampler."""

import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v009_attention16_extension35k'
    manifest = Path('manifests/v009.csv')
    original = Path('checkpoints/v009_attention16_extension34k')
    original_recipe = Path('results/plans/v009_attention16_32k.json')
    control_path = Path('results/experiments/v009_attention16_extension34k/screen.json')
    lead_path = Path('results/experiments/v007_freeu_cfg310/selection.json')
    control, lead = (read(path)['candidate'] for path in (control_path, lead_path))
    digest = sha256(manifest)
    parent = verified_checkpoint(original, 34000, digest)
    source_hashes = read(original / 'run.json')['source_code_sha256']
    if sha256(parent) != control['checkpoint_sha256'] or any(
            sha256(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('Continuation requires the verified checkpoint and exact original trainer')
    run, out, export = (Path(folder) / name for folder in ('checkpoints', 'results/experiments', 'private_build'))
    plan = Path('results/plans') / f'{name}.json'
    command = read(original_recipe)['command'].copy()
    for key, value in (('--updates', '35000'), ('--out', str(run))):
        command[command.index(key) + 1] = value
    command[0] = sys.executable
    command += ['--resume', str(parent)]
    runtime = dict(weights='ema', steps=50, guidance=3.25)
    sampler = runtime | dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5,
                             class_noise_max=.10, freeu=True, freeu_strength=2.0)
    inputs = (Path(__file__), parent, original_recipe, original / 'run.json', control_path, lead_path, manifest,
              manifest.with_suffix('.validation.csv'), manifest.with_suffix('.provenance.json'),
              Path(command[command.index('--config') + 1]), Path('src/release/export.py'))
    spec = dict(inputs=source_hashes | {str(p): sha256(p) for p in inputs}, command=command,
                sampler=sampler, seeds=[0, 101, 202], n_per_class=1000,
                rule='Use the 4.25% copy rate as headroom for one final 1k continuation. Advance only if '
                     'the official screen passes, reaches 13/15, and improves over both controls.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen continuation inputs changed')
    verify_starter()
    completion = run / 'completed.json'
    if not completion.exists():
        saved = sorted(run.glob('update_*.json'))
        if saved:
            command[-1] = str(verified_checkpoint(run, read(saved[-1])['update'], digest))
        subprocess.run(command, check=True)
    receipt = read(completion)
    first, = run.glob('segment_034000_*.json')
    lineage = read(first)
    if (receipt['last_update'], receipt['manifest_sha256'], lineage['parent_checkpoint_sha256'],
            lineage['source_code_sha256']) != (35000, digest, sha256(parent), source_hashes):
        raise ValueError('Continuation lineage or trainer bytes differ')
    checkpoint = verified_checkpoint(run, 35000, digest)
    print({'training_completed': receipt, 'checkpoint_sha256': sha256(checkpoint)}, flush=True)
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **sampler)
    if read(export / 'inference.json') != read(Path(control['exported']) / 'inference.json'):
        raise ValueError('Expected the exact 34k control inference settings')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    paired = [validated_result(item['evidence'][0]['path'], 0, item['exported']) for item in (control, lead)]
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    beats_control, beats_lead = [quality_improved({'score': score_results([row])}, screen) for row in paired]
    marks = screen['evidence'][0]['marks']
    advances = beats_control and beats_lead and marks['total'] >= 13
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances,
                                        improves_matched=beats_control))
    print({'seed': 0, 'marks': marks, 'advances': advances}, flush=True)
    if advances:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selected,
                   improves_reference=quality_improved(lead, selected)))
        print({'selection_marks': [row['marks'] for row in selected['evidence']]}, flush=True)


if __name__ == '__main__':
    main()

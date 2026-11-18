"""Test FreeU strength 2.2 on the frozen 34k attention weights."""

import sys
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v009_34k_freeu220'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    prior_path = Path('results/experiments/v009_attention16_extension34k/stability.json')
    prior = read(prior_path)['candidate']
    checkpoint = Path(prior['checkpoint'])
    manifest = Path('manifests/v009.csv')
    if sha256(checkpoint) != prior['checkpoint_sha256']:
        raise ValueError('Expected the exact FreeU checkpoint')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5,
                   class_noise_max=.10, freeu=True, freeu_strength=2.2)
    inputs = (Path(__file__), prior_path, checkpoint, manifest,
              *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                          'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in inputs}, sampler=sampler | options,
                screen_seed=0, selection_seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v009_34k_freeu220'],
                rule='Increase decoder feature shifts from 2.0 to 2.2. Advance only if the '
                     'full seed-0 screen scores at least 13/15 with every hard constraint satisfied.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen stronger FreeU inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    expected = read(Path(prior['exported']) / 'inference.json')
    expected['model']['freeu_strength'] = options['freeu_strength']
    if (read(export / 'inference.json') != expected
            or sha256(export / 'weights.pt') != prior['export_hashes']['weights.pt']):
        raise ValueError('FreeU strength must be the only inference configuration change')
    if not (out / 'interface.json').exists():
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'freeu_strength': options['freeu_strength']})
    screen = evaluate(checkpoint, manifest, export, out, seeds=[spec['screen_seed']], **sampler)
    marks = screen['evidence'][0]['marks']
    advances = screen['score'] is not None and marks['total'] >= 13
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': spec['screen_seed'], 'marks': marks, 'advances': advances}, flush=True)
    if advances:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['selection_seeds'], **sampler)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selected,
                   improves_reference=quality_improved(prior, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

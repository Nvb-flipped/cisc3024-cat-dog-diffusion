"""Test one lower CFG setting against the FreeU memorization failure."""

import sys
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v007_freeu_cfg310'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    prior_path = Path('results/experiments/v007_freeu_mild/selection.json')
    control_path = Path('results/experiments/v007_class_annealing010/selection.json')
    prior, control = (read(path)['candidate'] for path in (prior_path, control_path))
    checkpoint = Path(prior['checkpoint'])
    if sha256(checkpoint) != prior['checkpoint_sha256']:
        raise ValueError('Expected the exact FreeU checkpoint')
    sampler = dict(weights='ema', steps=50, guidance=3.10)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5, class_noise_max=.10, freeu=True)
    sources = (Path(__file__), prior_path, control_path, manifest, checkpoint,
               *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                           'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in sources}, sampler=sampler | options,
                screen_seed=202, selection_seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_freeu_lower_guidance'],
                rule='Test the previous FreeU failure seed first. Advance only if it passes the hard gate '
                     'and improves compared with the unmodified v007 control on that seed.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen FreeU lower-guidance inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    expected = read(Path(prior['exported']) / 'inference.json')
    expected['sampling']['guidance'] = sampler['guidance']
    if (read(export / 'inference.json') != expected
            or any(sha256(export / item) != prior['export_hashes'][item]
                   for item in ('model.py', 'generate.py', 'diffusion.py', 'weights.pt', 'manifest.csv'))):
        raise ValueError('CFG strength must be the only inference change')
    if not (out / 'interface.json').exists():
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'guidance': sampler['guidance']})
    seed = spec['screen_seed']
    anchor_entry = next(row for row in control['evidence'] if row['seed'] == seed)
    anchor = validated_result(anchor_entry['path'], seed, control['exported'])
    screen = evaluate(checkpoint, manifest, export, out, seeds=[seed], **sampler)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': seed, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['selection_seeds'], **sampler)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selected,
                   improves_reference=quality_improved(control, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

"""Test a finer DDIM grid on the verified FreeU CFG 3.10 export."""

from pathlib import Path
import sys

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v007_freeu_ddim100'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    control_path = Path('results/experiments/v007_freeu_cfg310/selection.json')
    control = read(control_path)['candidate']
    control_export = Path(control['exported'])
    checkpoint = Path(control['checkpoint'])
    if sha256(checkpoint) != control['checkpoint_sha256']:
        raise ValueError('Expected the frozen FreeU checkpoint')
    runtime = dict(weights='ema', steps=100, guidance=3.10)
    settings = runtime | dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5,
                              class_noise_max=.10, freeu=True)
    paths = (Path(__file__), control_path, checkpoint, manifest,
             *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=settings,
                screen_seed=202, selection_seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_freeu_ddim100'],
                rule='Change only the number of uniform DDIM steps. Screen the prior highest-copy seed. '
                     'Advance only if the full official screen earns at least 13/15 and passes every gate.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen DDIM100 inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **settings)
    expected = read(control_export / 'inference.json')
    expected['sampling']['steps'] = runtime['steps']
    if (read(export / 'inference.json') != expected or
            any(sha256(export / item) != control['export_hashes'][item]
                for item in ('weights.pt', 'manifest.csv', 'generate.py', 'diffusion.py'))):
        raise ValueError('The DDIM grid must be the only inference configuration change')
    seed = spec['screen_seed']
    control_entry = next(row for row in control['evidence'] if row['seed'] == seed)
    anchor = validated_result(control_entry['path'], seed, control_export)
    if not (out / 'interface.json').exists():
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'steps': runtime['steps']})
    screen = evaluate(checkpoint, manifest, export, out, seeds=[seed], **runtime)
    marks = screen['evidence'][0]['marks']
    advances = marks['total'] >= 13 and quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': seed, 'marks': marks, 'advances': advances}, flush=True)
    if advances:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['selection_seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selected,
                   improves_reference=quality_improved(control, selected)))
        print({'selection_marks': [row['marks'] for row in selected['evidence']]}, flush=True)


if __name__ == '__main__':
    main()

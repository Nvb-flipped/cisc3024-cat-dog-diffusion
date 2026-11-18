"""Check the first held-out loss plateau before the 32k copy-rate failure."""

from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v007_timestep_focus065_20k'
    parent = 'v007_timestep_focus065'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    run = Path('checkpoints') / f'{parent}_seed17'
    checkpoint = verified_checkpoint(run, 20000, sha256(manifest))
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    reference = read(reference_path)['candidate']
    reference_config = read(Path(reference['exported']) / 'inference.json')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    original_plan = Path('results/plans') / f'{parent}.json'
    settings = read(original_plan)['sampler']
    paths = (Path(__file__), manifest, checkpoint, reference_path, original_plan,
             run / 'validation_020000.json', run / 'validation_032000.json',
             Path('results/experiments') / parent / 'screen.json',
             *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=settings, seed=0,
                n_per_class=1000, command='python -u -m scripts.run_v007_timestep_early',
                rule='Test the first held-out EMA-loss plateau, 20k, with the same official sampler and seed. '
                     'This is one checkpoint-age comparison after the 32k hard-cap failure.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen early-checkpoint inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **settings)
    if read(export / 'inference.json') != reference_config:
        raise ValueError('The early checkpoint must use the same sampler as the retained reference')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screen = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', {'plan_sha256': sha256(plan), 'candidate': screen, 'advances': advances})
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)


if __name__ == '__main__':
    main()

"""Evaluate one intermediate checkpoint before deciding whether to keep this training route."""

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
    name = 'v007_timestep_focus065_26k'
    manifest = Path('manifests/v007.csv')
    run = Path('checkpoints/v007_timestep_focus065_seed17')
    checkpoint = verified_checkpoint(run, 26000, sha256(manifest))
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    trial_path = Path('results/plans/v007_timestep_focus065.json')
    parent_screen = Path('results/experiments/v007_timestep_focus065/screen.json')
    early_screen = Path('results/experiments/v007_timestep_focus065_20k/screen.json')
    reference = read(reference_path)['candidate']
    settings = read(trial_path)['sampler']
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    inputs = (Path(__file__), manifest, checkpoint, reference_path, trial_path, parent_screen, early_screen,
              *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                          'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in inputs}, sampler=settings, seed=0,
                n_per_class=1000, command='python -u -m scripts.run_v007_timestep_midpoint',
                rule='Test 26k, the midpoint between the weak 20k and copying 32k checkpoints. Keep the exact '
                     'sampler and allow confirmation only after a valid quality gain over the current 12/15 control.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen midpoint plan changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **settings)
    if read(export / 'inference.json') != read(Path(reference['exported']) / 'inference.json'):
        raise ValueError('Expected the exact control sampler')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    result = evaluate(checkpoint, manifest, export, out, weights='ema', steps=50, guidance=3.25)
    advances = quality_improved({'score': score_results([anchor])}, result)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=result, advances=advances))
    print({'advances': advances, 'marks': result['evidence'][0]['marks']}, flush=True)


if __name__ == '__main__':
    main()

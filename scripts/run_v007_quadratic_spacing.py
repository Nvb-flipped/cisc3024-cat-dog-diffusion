"""Screen equal-cost quadratic DDIM spacing on the retained v007 checkpoint."""

import argparse
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    name = 'v007_quadratic_ddim50'
    manifest = Path('manifests/v007.csv')
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    reference = read(reference_path)['candidate']
    checkpoint = Path(reference['checkpoint'])
    if sha256(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Expected the frozen v007 weights')
    export = Path('private_build') / name
    out = Path('results/experiments') / name
    plan = Path('results/plans') / f'{name}.json'
    runtime = dict(weights='ema', steps=50, guidance=3.25)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5, class_noise_max=.10,
                   timestep_spacing='quadratic')
    files = (Path(__file__), reference_path, checkpoint, manifest,
             *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(p): sha256(p) for p in files}, sampler=runtime | options,
                seeds=[0, 101, 202], n_per_class=1000, source='https://arxiv.org/abs/2404.14507',
                rule='One equal-cost spacing screen, preserving both discrete endpoints and the exact control.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen timestep-spacing inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    expected = read(Path(reference['exported']) / 'inference.json')
    expected['sampling']['timestep_spacing'] = 'quadratic'
    if read(export / 'inference.json') != expected:
        raise ValueError('Expected DDIM spacing to be the only configuration change')
    if not (out / 'interface.json').exists():
        checks = verify_interface(reference['exported'], export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'timestep_spacing': 'quadratic'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and args.confirm:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                   candidate=selected, improves_reference=quality_improved(reference, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

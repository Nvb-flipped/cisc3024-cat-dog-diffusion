"""Screen a mild condition-annealed variant of the leading v007 sampler."""

import argparse
import sys
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    name = 'v007_class_annealing010'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    reference_path = Path('results/experiments/v007_32k_temporal103/selection.json')
    reference = read(reference_path)['candidate']
    checkpoint = Path(reference['checkpoint'])
    if sha256(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('The sampler must use the exact retained checkpoint')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5, class_noise_max=.10)
    paths = (Path(__file__), manifest, checkpoint, reference_path,
             *map(Path, ('submission/diffusion.py', 'submission/model.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=sampler | options,
                seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_class_annealing'],
                source='https://arxiv.org/abs/2310.17347',
                rule='Test mild per-output Gaussian class-embedding annealing. Use a nonzero class-signal '
                     'floor and preserve each embedding mean and standard deviation. Keep the null condition '
                     'and all reference settings fixed. Advance only a valid official improvement.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen class-annealing inputs changed')
    verify_starter()
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    anchor_config = read(Path(reference['exported']) / 'inference.json')
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    exported_config = read(export / 'inference.json')
    if exported_config != (anchor_config | {
            'sampling': anchor_config['sampling'] | {'class_noise_max': options['class_noise_max']}}):
        raise ValueError('Class annealing must be the only change from the leading sampler configuration')
    if not (out / 'interface.json').exists():
        checks = verify_interface(reference['exported'], export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'options': options})
    screen = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', {'plan_sha256': sha256(plan), 'candidate': screen, 'advances': advances})
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if args.confirm and advances:
        result = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **sampler)
        write_json(out / 'selection.json', {'screen_sha256': sha256(out / 'screen.json'), 'candidate': result,
                                           'improves_reference': quality_improved(reference, result)})
        print({'confirmation_marks': [row['marks'] for row in result['evidence']]}, flush=True)


if __name__ == '__main__':
    main()

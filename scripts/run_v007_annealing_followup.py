"""Test one sampler change at a time against a retained v007 export."""

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
    parser.add_argument('setting', choices=('strength', 'rescale'))
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    manifest = Path('manifests/v007.csv')
    prior_path = Path('results/experiments/v007_class_annealing010/selection.json')
    fallback_path = Path('results/experiments/v007_32k_temporal103/selection.json')
    prior = read(prior_path)
    reference = prior['candidate'] if prior['improves_reference'] else read(fallback_path)['candidate']
    anchor_config = read(Path(reference['exported']) / 'inference.json')
    anchor_sampling = anchor_config['sampling']
    options = {key: anchor_sampling[key] for key in
               ('pag_scale', 'score_sharpness', 'score_sigma', 'class_noise_max') if key in anchor_sampling}
    if args.setting == 'strength':
        name = 'v007_class_annealing025'
        change = {'class_noise_max': .25}
        rule = 'Test one increase in class-annealing strength. Keep class-noise streams and all other inputs fixed.'
    else:
        name = f"{Path(reference['exported']).name}_rescale025"
        change = {'guidance_rescale': .25}
        rule = 'Transfer the previously tested v002 guidance rescale strength. Keep all other inputs fixed.'
    export = Path('private_build') / name
    out = Path('results/experiments') / name
    plan = Path('results/plans') / f'{name}.json'
    checkpoint = Path(reference['checkpoint'])
    if sha256(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Annealing follow-up must use the retained v007 checkpoint')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    options.update(change)
    paths = (Path(__file__), manifest, checkpoint, prior_path, fallback_path,
             *map(Path, ('submission/diffusion.py', 'submission/model.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    if args.setting == 'rescale':
        paths += (Path('results/experiments/rescale_followup/comparison.json'),)
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=sampler | options,
                seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_annealing_followup', args.setting],
                rule=rule)
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen annealing follow-up inputs changed')
    verify_starter()
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json') != (anchor_config | {'sampling': anchor_sampling | change}):
        raise ValueError('Only the declared sampler setting may change from the reference')
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

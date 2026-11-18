"""Test whether stronger PAG can use the perturbation checkpoint's copy-rate margin."""

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
    name = 'v007_perturbation_pag050'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    control_path = Path('results/experiments/v007_input_perturbation/comparison.json')
    reference_path = Path('results/experiments/v007_32k_temporal103/selection.json')
    control = read(control_path)['screened']
    reference = read(reference_path)['candidate']
    checkpoint = Path(control['checkpoint'])
    if sha256(checkpoint) != control['checkpoint_sha256']:
        raise ValueError('Perturbation checkpoint differs from the official control')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    paths = (Path(__file__), manifest, checkpoint, control_path, reference_path,
             *map(Path, ('submission/diffusion.py', 'submission/generate.py', 'submission/model.py',
                         'src/release/export.py', 'src/training/optimize.py', 'src/evaluation/run.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=sampler | {'pag_scale': .5},
                seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_perturbation_pag'],
                rule='Change only PAG from .15 to .50 on the existing perturbation checkpoint. '
                     'Confirm only a seed-0 improvement over the leading official v007 temporal export.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen comparison inputs changed')
    verify_starter()
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    validated_result(control['evidence'][0]['path'], 0, control['exported'])
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    if read(export / 'inference.json')['sampling']['pag_scale'] != .5:
        raise ValueError('Export does not match the planned PAG strength')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(control['exported'], export) | {'solver': 'ddim'})
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

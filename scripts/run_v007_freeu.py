"""Screen one inference-only FreeU decoder setting on the frozen v007 weights."""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    name = 'v007_freeu_mild'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    reference_path = Path('results/experiments/v007_class_annealing010/selection.json')
    reference = read(reference_path)['candidate']
    reference_export = Path(reference['exported'])
    checkpoint = Path(reference['checkpoint'])
    if sha256(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Expected the frozen v007 checkpoint')

    sampler = dict(weights='ema', steps=50, guidance=3.25)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5, class_noise_max=.10, freeu=True)
    inputs = (Path(__file__), reference_path, checkpoint, manifest,
              *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                          'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in inputs}, sampler=sampler | options,
                seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_freeu'],
                source='https://arxiv.org/abs/2309.11497',
                rule='One mild feature reweight on the first two decoder stages. Keep exact learned weights, '
                     'sampling settings and pool rows. Advance only a valid official improvement.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen FreeU inputs changed')
    verify_starter()
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference_export)
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    expected = read(reference_export / 'inference.json')
    expected['model']['freeu'] = True
    if (read(export / 'inference.json') != expected
            or sha256(export / 'weights.pt') != reference['export_hashes']['weights.pt']):
        raise ValueError('The decoder feature reweight must be the only inference change')
    if not (out / 'interface.json').exists():
        reference_checks = verify_interface(reference_export, reference_export)
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'freeu': True,
                                                    'reference_checks': reference_checks})
    screen = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances and args.confirm:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **sampler)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                   candidate=selected, improves_reference=quality_improved(reference, selected)))
        print([row['marks'] for row in selected['evidence']], flush=True)


if __name__ == '__main__':
    main()

"""Run three prespecified single-factor sampler comparisons on exact 34k weights."""

import argparse
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


ARMS = {'cfg310': {'guidance': 3.10}, 'pag010': {'pag_scale': .10},
        'freeu180': {'freeu_strength': 1.80}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=ARMS)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    name = f'v009_34k_{args.arm}'
    out, export = Path('results/experiments') / name, Path('private_build') / name
    plan = Path('results/plans/v009_34k_refine.json')
    parent_path = Path('results/experiments/v009_attention16_extension34k/stability.json')
    lead_path = Path('results/experiments/v007_freeu_cfg310/selection.json')
    parent, lead = (read(path)['candidate'] for path in (parent_path, lead_path))
    checkpoint, manifest = Path(parent['checkpoint']), Path('manifests/v009.csv')
    paths = (Path(__file__), parent_path, lead_path, checkpoint, manifest,
             *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                         'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, arms=ARMS,
                seeds=[0, 101, 202], n_per_class=1000,
                rule='Three single-factor seed-0 screens on the frozen 34k checkpoint. Confirm only a valid '
                     'candidate that improves on the paired 34k control and either reaches 13/15 or reduces '
                     'FID by at least 1 with recall at least 0.30 and accuracy at least 0.90. '
                     'Promotion uses the original minimum-tier/normalized-margin rule against both '
                     'three-seed controls, without an additional worst-recall veto.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen sampler comparison inputs changed')
    if sha256(checkpoint) != parent['checkpoint_sha256']:
        raise ValueError('Expected the exact 34k weights')
    verify_starter()
    options = dict(weights='ema', steps=50, guidance=3.25, pag_scale=.15, score_sharpness=1.03,
                   score_sigma=.5, class_noise_max=.10, freeu=True, freeu_strength=2.0) | ARMS[args.arm]
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **options)
    expected = read(Path(parent['exported']) / 'inference.json')
    for key, value in ARMS[args.arm].items():
        expected['model' if key == 'freeu_strength' else 'sampling'][key] = value
    if (read(export / 'inference.json') != expected or any(
            sha256(export / item) != parent['export_hashes'][item]
            for item in ('weights.pt', 'manifest.csv', 'model.py', 'generate.py', 'diffusion.py'))):
        raise ValueError('Expected a single configuration change and the exact immutable weights')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim', 'arm': args.arm})
    anchor = validated_result(parent['evidence'][0]['path'], 0, parent['exported'])
    runtime = {key: options[key] for key in ('weights', 'steps', 'guidance')}
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    row = screen['evidence'][0]
    metrics = row['metrics']
    advances = (quality_improved({'score': score_results([anchor])}, screen)
                and metrics['accuracy'] >= .90 and metrics['recall'] >= .30
                and (row['marks']['total'] >= 13 or metrics['fid'] <= anchor['metrics']['fid'] - 1))
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'arm': args.arm, 'seed': 0, 'metrics': metrics, 'marks': row['marks'], 'advances': advances}, flush=True)
    if args.confirm and advances:
        result = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        promotion = quality_improved(parent, result) and quality_improved(lead, result)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'),
                                               candidate=result, improves_reference=promotion))
        print({'arm': args.arm, 'marks': [item['marks'] for item in result['evidence']],
               'improves_reference': promotion}, flush=True)


if __name__ == '__main__':
    main()

"""Screen the average epsilon prediction of two full v007 fits."""

import sys
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.run_v007_autoguidance import build_export
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v007_score_average'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    reference_path = Path('results/experiments/v007_32k_temporal103/selection.json')
    peer_path = Path('results/experiments/v007_input_perturbation/comparison.json')
    reference = read(reference_path)['candidate']
    peer = read(peer_path)['screened']
    checkpoint = Path(reference['checkpoint'])
    partner = Path(peer['checkpoint'])
    if sha256(partner) != peer['checkpoint_sha256'] or sha256(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Both checkpoints must match the verified official fits')
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    validated_result(peer['evidence'][0]['path'], 0, peer['exported'])
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    scale = -.5
    paths = (Path(__file__), manifest, reference_path, peer_path, checkpoint, partner,
             *map(Path, ('scripts/run_v007_autoguidance.py', 'scripts/autoguided_generate.py',
                         'scripts/run_v002_dpmpp.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=sampler,
                average_weights=[.5, .5], seed=0, n_per_class=1000,
                source='https://arxiv.org/abs/2601.11444',
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_score_average'],
                rule='Average conditional and null predictions separately before applying identical CFG. '
                     'Disable PAG and temporal rescaling to test the plain score average.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen score-average inputs changed')
    verify_starter()
    if not export.exists():
        build_export(export, reference, checkpoint, partner, sampler, scale)
    if (read(export / 'inference.json')['autoguidance_scale'],
            read(export / 'provenance.json')['guide_checkpoint_sha256']) != (scale, sha256(partner)):
        raise ValueError('Export does not match the declared score-average pair')
    if not (out / 'interface.json').exists():
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'average_weights': [.5, .5]})
    result = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, result)
    write_json(out / 'screen.json', {'plan_sha256': sha256(plan), 'candidate': result, 'advances': advances})
    print({'seed': 0, 'marks': result['evidence'][0]['marks'], 'advances': advances}, flush=True)


if __name__ == '__main__':
    main()

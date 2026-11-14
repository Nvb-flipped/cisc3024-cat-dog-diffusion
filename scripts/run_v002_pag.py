"""Screen one perturbed-attention guidance setting on the retained v002 weights."""

import argparse
import json
from pathlib import Path

from scripts.run_v002_dpmpp import MANIFEST, read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


NAME = 'v002_pag'
PLAN = Path(f'results/plans/{NAME}.json')
OUT = Path(f'results/experiments/{NAME}')
EXPORT = Path(f'private_build/{NAME}')
REFERENCE = Path('results/experiments/rescale_followup/comparison.json')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
OVERRIDES = dict(guidance_rescale=.25, pag_scale=1.0)
SEEDS = [0, 101, 202]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    ref = read(REFERENCE)['candidate']
    checkpoint = Path(ref['checkpoint'])
    paths = (REFERENCE, MANIFEST, checkpoint, Path(__file__), Path('submission/diffusion.py'),
             Path('submission/model.py'), Path('src/release/export.py'), Path('scripts/run_v002_dpmpp.py'),
             Path('src/training/optimize.py'), Path('src/evaluation/run.py'))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=SAMPLER | OVERRIDES,
                seeds=SEEDS, n_per_class=1000)
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, spec | dict(created=timestamp(), source='https://arxiv.org/html/2403.17377',
                   rule='Use identity attention in the first 8x8 encoder attention of the auxiliary '
                        'conditional prediction. Add the conditional-minus-perturbed residual at scale 1.0 '
                        'after CFG rescale, only for timesteps 149–949. Preserve the existing parameters '
                        'and sample path outside those steps. Advance only by the existing quality gate.'))
        return
    plan = read(PLAN)
    if any(plan[key] != value for key, value in spec.items()):
        raise ValueError('Frozen perturbed-attention comparison changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve completed comparison')
    verify_starter()
    if not EXPORT.exists():
        export_checkpoint(checkpoint, MANIFEST, EXPORT, **plan['sampler'])
    sampling = read(EXPORT / 'inference.json')['sampling']
    if any(sampling.get(key) != value for key, value in OVERRIDES.items()):
        raise ValueError('Perturbed-attention export differs from plan')
    if not (OUT / 'interface.json').exists():
        checks = verify_interface(ref['exported'], EXPORT)
        write_json(OUT / 'interface.json', checks | OVERRIDES | {'solver': 'ddim', 'exact_anchor_sampler': True})
    anchor = validated_result(ref['evidence'][0]['path'], 0, ref['exported'])
    screened = evaluate(checkpoint, MANIFEST, EXPORT, OUT, **SAMPLER)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=screened['evidence'])), flush=True)
    selected = None
    if quality_improved({'score': score_results([anchor])}, screened):
        selected = evaluate(checkpoint, MANIFEST, EXPORT, OUT, seeds=SEEDS, **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(ref, selected)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=screened,
                                            candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

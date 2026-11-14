"""Screen bounded temporal score rescaling on the retained v002 checkpoint."""

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


NAME = 'v002_temporal_score'
PLAN = Path(f'results/plans/{NAME}.json')
OUT = Path(f'results/experiments/{NAME}')
EXPORT = Path(f'private_build/{NAME}')
REFERENCE = Path('results/experiments/rescale_followup/comparison.json')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
SETTINGS = dict(guidance_rescale=.25, score_sigma=.5)
ARMS = {'broader': .93, 'sharper': 1.07}
SEEDS = [0, 101, 202]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    ref = read(REFERENCE)['candidate']
    checkpoint = Path(ref['checkpoint'])
    paths = (REFERENCE, MANIFEST, checkpoint, Path(__file__), Path('submission/diffusion.py'),
             Path('submission/model.py'), Path('src/release/export.py'), Path('src/training/optimize.py'),
             Path('src/evaluation/run.py'), Path('scripts/run_v002_dpmpp.py'))
    inputs = {str(path): sha256(path) for path in paths}
    spec = dict(inputs=inputs, sampler=SAMPLER | SETTINGS, arms=ARMS, seeds=SEEDS, n_per_class=1000)
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, spec | dict(created=timestamp(), source='https://arxiv.org/html/2510.01184v2#S4',
                   rule='Test the published mild broadening factor and a similarly sized sharpening factor. '
                        'Fix sigma at 0.5 in normalized pixel space. Scale epsilon after CFG rescale with equation (5). '
                        'Keep the exact v002 checkpoint and evaluate both directions on full seed 0. '
                        'Advance at most one valid improvement.'))
        return
    plan = read(PLAN)
    if any(plan[key] != value for key, value in spec.items()):
        raise ValueError('Frozen temporal score comparison changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve completed evidence')
    verify_starter()
    anchor = validated_result(ref['evidence'][0]['path'], 0, ref['exported'])
    anchor_score = {'score': score_results([anchor])}
    screened = {}
    for name, sharpness in ARMS.items():
        export, folder = EXPORT / name, OUT / name
        settings = SETTINGS | {'score_sharpness': sharpness}
        if not export.exists():
            export_checkpoint(checkpoint, MANIFEST, export, **SAMPLER, **settings)
        sampling = read(export / 'inference.json')['sampling']
        if any(sampling.get(key) != value for key, value in settings.items()):
            raise ValueError(f'Temporal score configuration changed: {name}')
        if not (folder / 'interface.json').exists():
            checks = verify_interface(ref['exported'], export)
            write_json(folder / 'interface.json', checks | settings | {'solver': 'ddim', 'exact_anchor_sampler': True})
        row = evaluate(checkpoint, MANIFEST, export, folder, **SAMPLER)
        screened[name] = row
        print(json.dumps(dict(event='official_evaluation_completed', name=name, evidence=row['evidence'])), flush=True)
    best = max((name for name, trial in screened.items() if quality_improved(anchor_score, trial)),
               key=lambda name: screened[name]['score'], default=None)
    selected = None
    if best:
        selected = evaluate(checkpoint, MANIFEST, EXPORT / best, OUT / best, seeds=SEEDS, **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', name=best, evidence=selected['evidence'][1:])),
              flush=True)
    improved = selected is not None and quality_improved(ref, selected)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=screened,
                                            candidate=selected, promotes=best, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

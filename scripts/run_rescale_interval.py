"""Evaluate restricted CFG on the retained v002 rescale export."""

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


PLAN = Path('results/plans/rescale_interval.json')
REFERENCE = Path('results/experiments/rescale_followup/comparison.json')
OUT = Path('results/experiments/rescale_interval')
EXPORT = Path('private_build/rescale_interval')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
RESCALE = .25
ARMS = {'late_off': dict(guidance_min_timestep=149),
        'ends_off': dict(guidance_min_timestep=149, guidance_max_timestep=949)}
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
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, sampler=SAMPLER,
                              rescale=RESCALE, arms=ARMS, seeds=SEEDS, n_per_class=1000,
                              sources=['https://arxiv.org/abs/2404.07724', 'https://arxiv.org/abs/2305.08891'],
                              rule='Use the exact v002 rescale .25 anchor. Screen both arms on full seed 0; '
                                   'advance at most one valid improvement. Use conditional epsilon outside '
                                   'the specified bounds. The high-noise cutoff removes the first three DDIM '
                                   'steps, a smaller change than the earlier failed max-849 screen.'))
        return
    plan = read(PLAN)
    if (plan['inputs'], plan['sampler'], plan['rescale'], plan['arms'], plan['seeds'], plan['n_per_class']) != (
            inputs, SAMPLER, RESCALE, ARMS, SEEDS, 1000):
        raise ValueError('Frozen restricted-guidance plan changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison')
    verify_starter()
    anchor = validated_result(ref['evidence'][0]['path'], 0, ref['exported'])
    anchor_score = {'score': score_results([anchor])}
    screened = {}
    for name, bounds in ARMS.items():
        export, folder = EXPORT / name, OUT / name
        overrides = bounds | {'guidance_rescale': RESCALE}
        if not export.exists():
            export_checkpoint(checkpoint, MANIFEST, export, **SAMPLER, **overrides)
        actual = read(export / 'inference.json')['sampling']
        if any(actual.get(key) != value for key, value in overrides.items()):
            raise ValueError(f'Unexpected sampler bounds in {name}')
        receipt = folder / 'interface.json'
        if not receipt.exists():
            checks = verify_interface(ref['exported'], export)
            write_json(receipt, checks | overrides | {'solver': 'ddim', 'exact_anchor_sampler': True})
        trial = evaluate(checkpoint, MANIFEST, export, folder, **SAMPLER)
        screened[name] = trial
        print(json.dumps(dict(event='official_evaluation_completed', name=name, evidence=trial['evidence'])), flush=True)
    promoted = max((name for name, result in screened.items() if quality_improved(anchor_score, result)),
                   key=lambda name: screened[name]['score'], default=None)
    selected = None
    if promoted:
        selected = evaluate(checkpoint, MANIFEST, EXPORT / promoted, OUT / promoted, seeds=SEEDS, **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', name=promoted,
                              evidence=selected['evidence'][1:])), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=screened,
                                            promoted=promoted, candidate=selected,
                                            improves_reference=selected is not None and quality_improved(ref, selected)))


if __name__ == '__main__':
    main()

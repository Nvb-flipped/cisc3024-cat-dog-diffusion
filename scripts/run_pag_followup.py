"""Test a single reduced PAG scale on the retained v002 and low-copy v006 checkpoints."""

import argparse
import json
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


NAME = 'pag_followup'
PLAN = Path(f'results/plans/{NAME}.json')
OUT = Path(f'results/experiments/{NAME}')
EXPORT = Path(f'private_build/{NAME}')
REF = Path('results/experiments/rescale_followup/comparison.json')
LOW_COPY = Path('results/experiments/v006_dropout15/screening.json')
FULL_PAG = Path('results/experiments/v002_pag/comparison.json')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
SEEDS = [0, 101, 202]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REF)['candidate']
    low_copy = next(row for row in read(LOW_COPY)['screened'] if row['update'] == 30000)
    arms = {'v002_half': (reference, Path('manifests/v002.csv'), .25),
            'v006_dropout15_half': (low_copy, Path('manifests/v006.csv'), 0.0)}
    paths = [REF, LOW_COPY, FULL_PAG, Path(__file__), Path('submission/diffusion.py'),
             Path('submission/model.py'), Path('src/release/export.py'), Path('scripts/run_v002_dpmpp.py'),
             Path('src/training/optimize.py'), Path('src/evaluation/run.py')]
    definitions = {}
    for name, (row, manifest, rescale) in arms.items():
        paths.extend((manifest, Path(row['checkpoint'])))
        definitions[name] = dict(manifest=str(manifest), checkpoint=row['checkpoint'],
                                 guidance_rescale=rescale, pag_scale=.5)
    spec = dict(inputs={str(path): sha256(path) for path in paths}, arms=definitions,
                sampler=SAMPLER, seeds=SEEDS, n_per_class=1000)
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, spec | dict(created=timestamp(),
                   rule='Full scale lowered FID but exceeded the hard memorization cap. Test only one half scale '
                        'with the v002 anchor and with the v006 dropout .15 checkpoint that had 1.05% '
                        'memorization. Each arm uses its own exact no-PAG sampler and manifest as the '
                        'paired control. Advance only a valid improvement versus the leading v002 anchor.'))
        return
    plan = read(PLAN)
    if any(plan[key] != value for key, value in spec.items()):
        raise ValueError('Frozen PAG follow-up changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve completed evidence')
    verify_starter()
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    anchor_score = {'score': score_results([anchor])}
    trials = {}
    for name, (control, manifest, rescale) in arms.items():
        export, folder = EXPORT / name, OUT / name
        overrides = dict(guidance_rescale=rescale, pag_scale=.5)
        if not export.exists():
            export_checkpoint(control['checkpoint'], manifest, export, **SAMPLER, **overrides)
        sampling = read(export / 'inference.json')['sampling']
        if any(sampling.get(key) != value for key, value in overrides.items()):
            raise ValueError(f'PAG settings changed: {name}')
        if not (folder / 'interface.json').exists():
            checks = verify_interface(control['exported'], export)
            write_json(folder / 'interface.json', checks | overrides | {'solver': 'ddim'})
        trial = evaluate(control['checkpoint'], manifest, export, folder, **SAMPLER)
        trials[name] = trial
        print(json.dumps(dict(event='official_evaluation_completed', name=name, evidence=trial['evidence'])), flush=True)
    best = max((name for name, trial in trials.items() if quality_improved(anchor_score, trial)),
               key=lambda name: trials[name]['score'], default=None)
    selected = None
    if best:
        control, manifest, _ = arms[best]
        selected = evaluate(control['checkpoint'], manifest, EXPORT / best, OUT / best, seeds=SEEDS, **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', name=best,
                              evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=trials,
                                            promoted=best, candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

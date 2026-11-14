"""Test reduced CFG rescale on v002 and the larger v006 checkpoint with spare memorization margin."""

import argparse
import json
from pathlib import Path

from scripts.run_v002_dpmpp import REFERENCE, read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


PLAN = Path('results/plans/rescale_followup.json')
V006 = Path('results/experiments/v006_coverage/screening.json')
OUT = Path('results/experiments/rescale_followup')
EXPORTS = Path('private_build/rescale_followup')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REFERENCE)
    v006 = next(row for row in read(V006)['screened'] if row['update'] == 30000)
    arms = [('v002_rescale025', reference, Path('manifests/v002.csv'), .25),
            ('v006_rescale050', v006, Path('manifests/v006.csv'), .50)]
    inputs = {str(p): sha256(p) for p in (REFERENCE, V006, Path(__file__), Path('submission/diffusion.py'),
              Path('src/release/export.py'), Path('scripts/run_v002_dpmpp.py'), Path('src/evaluation/run.py'),
              *(manifest for _, _, manifest, _ in arms), *(Path(row['checkpoint']) for _, row, _, _ in arms))}
    definitions = [{'name': name, 'manifest': str(manifest), 'checkpoint': row['checkpoint'],
                    'rescale': rescale} for name, row, manifest, rescale in arms]
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, arms=definitions, sampler=SAMPLER,
                              seeds=[0, 101, 202], n_per_class=1000,
                              rule='Compare each arm with its exact no-rescale export on seed 0. '
                                   'Advance at most one valid candidate beating v002 by the existing rule.'))
        return
    plan = read(PLAN)
    if (plan['inputs'], plan['arms'], plan['sampler']) != (inputs, definitions, SAMPLER):
        raise ValueError('Frozen rescale follow-up inputs changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve the existing comparison')
    verify_starter()
    trials = []
    for name, row, manifest, rescale in arms:
        folder, export = OUT / name, EXPORTS / name
        if not export.exists():
            export_checkpoint(row['checkpoint'], manifest, export, **SAMPLER, guidance_rescale=rescale)
        if read(export / 'inference.json')['sampling'].get('guidance_rescale') != rescale:
            raise ValueError(f'Rescale differs from plan: {name}')
        receipt = folder / 'interface.json'
        if not receipt.exists():
            checks = verify_interface(row['exported'], export)
            write_json(receipt, checks | dict(solver='ddim', guidance_rescale=rescale))
        control = validated_result(row['evidence'][0]['path'], 0, row['exported'])
        trial = evaluate(row['checkpoint'], manifest, export, folder, **SAMPLER)
        trials.append(trial | dict(name=name, control_metrics=control['metrics']))
        print(json.dumps(dict(event='official_evaluation_completed', name=name, evidence=trial['evidence'])), flush=True)
    baseline = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    best = max((trial for trial in trials if quality_improved({'score': score_results([baseline])}, trial)),
               key=lambda row: row['score'], default=None)
    selected = None
    if best:
        name = best['name']
        manifest = next(path for key, _, path, _ in arms if name == key)
        selected = evaluate(best['checkpoint'], manifest, best['exported'], OUT / name, seeds=plan['seeds'], **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', name=name, evidence=selected['evidence'][1:])), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=trials, candidate=selected,
                                             improves_reference=selected is not None and quality_improved(reference, selected)))


if __name__ == '__main__':
    main()

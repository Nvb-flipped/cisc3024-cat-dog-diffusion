"""Evaluate one average of same-run EMA snapshots with the leading sampler."""

import argparse
import json
from pathlib import Path

import torch

from scripts.run_v002_dpmpp import MANIFEST, read
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


PLAN = Path('results/plans/ema_trajectory_probe.json')
OUT = Path('results/experiments/ema_trajectory_probe')
EXPORT = Path('private_build/ema_trajectory_probe')
REFINED = Path('results/experiments/rescale_followup/comparison.json')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
RESCALE = .25


def average_snapshots(paths, destination):
    snapshots = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    parent = snapshots[-1]
    if any((state['manifest_sha256'], state['segment_id'], state['seed'], state['config']) != (
            sha256(MANIFEST), parent['segment_id'], parent['seed'], parent['config']) for state in snapshots):
        raise ValueError('EMA averaging requires the same exact dataset and training segment')
    ema = {name: torch.stack([state['ema'][name] for state in snapshots]).mean(dim=0) for name in parent['ema']}
    derived = {key: parent[key] for key in ('config', 'update', 'seed', 'manifest_sha256', 'stats',
                                         'source_code_sha256', 'data_review_provenance_sha256', 'segment_id')}
    derived['ema'] = ema
    derived['ema_snapshot_sources'] = {str(path): sha256(path) for path in paths}
    torch.save(derived, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REFINED)['candidate']
    parent = Path(reference['checkpoint']).parent
    checkpoints = [parent / f'update_{step:06d}.pt' for step in (28000, 30000)]
    sources = (REFINED, MANIFEST, *checkpoints, Path(__file__), Path('src/release/export.py'),
               Path('submission/diffusion.py'), Path('submission/model.py'))
    inputs = {str(path): sha256(path) for path in sources}
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, source_updates=[28000, 30000],
                              weights=[.5, .5], sampler=SAMPLER | {'guidance_rescale': RESCALE}, seeds=[0, 101, 202],
                              rule='A single same-trajectory weight average. Compare against the leading 30k export '
                                   'with the exact same sampler. Advance only by the existing quality rule.'))
        return
    plan = read(PLAN)
    if plan['inputs'] != inputs or plan['sampler'] != SAMPLER | {'guidance_rescale': RESCALE}:
        raise ValueError('Frozen EMA experiment inputs changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed EMA comparison')
    verify_starter()
    derived = EXPORT.with_suffix('.pt')
    if not derived.exists():
        average_snapshots(checkpoints, derived)
    stored = torch.load(derived, map_location='cpu', weights_only=True)
    if stored['ema_snapshot_sources'] != {str(path): inputs[str(path)] for path in checkpoints}:
        raise ValueError('Derived weights differ from the declared checkpoint sources')
    if not EXPORT.exists():
        export_checkpoint(derived, MANIFEST, EXPORT, **plan['sampler'])
    trial = evaluate(derived, MANIFEST, EXPORT, OUT, **SAMPLER)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=trial['evidence'])), flush=True)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    selected = None
    if quality_improved({'score': score_results([anchor])}, trial):
        selected = evaluate(derived, MANIFEST, EXPORT, OUT, seeds=plan['seeds'], **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=trial, candidate=selected,
                                             improves_reference=selected is not None and quality_improved(reference, selected)))


if __name__ == '__main__':
    main()

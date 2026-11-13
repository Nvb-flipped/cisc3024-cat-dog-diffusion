"""Run one bounded v006 continuation with its frozen optimizer and sampler."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_coverage_comparison import REFERENCE, SAMPLER, read, verified_checkpoint
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.training.optimize import evaluate, quality_improved


MANIFEST = Path('manifests/v006.csv')
PARENT_PLAN = Path('results/plans/v006_coverage.json')
PLAN = Path('results/plans/v006_continuation.json')
RESULTS = Path('results/experiments/v006_coverage')
OUT = RESULTS / 'continuation_032000'
EXPORT = Path('private_build/v006_coverage/update_032000')
TARGET = 32000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    parent = read(PARENT_PLAN)
    manifest_hash = sha256(MANIFEST)
    run = Path(parent['command'][parent['command'].index('--out') + 1])
    checkpoint = verified_checkpoint(run, 30000, manifest_hash)
    paths = (PARENT_PLAN, RESULTS / 'screening.json', RESULTS / 'comparison.json',
             REFERENCE, MANIFEST, checkpoint, Path(__file__))
    inputs = {str(path): sha256(path) for path in paths}
    for name, expected in parent['inputs'].items():
        if sha256(name) != expected:
            raise ValueError(f'Parent comparison input changed: {name}')
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        command = [*parent['command'], '--resume', str(checkpoint), '--updates', str(TARGET)]
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, command=command, target_update=TARGET,
                              sampler=SAMPLER, selection_seeds=[0, 101, 202],
                              rationale='The 28k to 30k comparison improved FID and recall while memorization '
                                        'remained at 2.55%. Test one additional 2,000-update interval.',
                              rule='One continuation. Advance only if seed 0 improves the existing v002 '
                                   'score or margin, then require the same three-seed improvement.'))
        return
    plan = read(PLAN)
    if plan['inputs'] != inputs or plan['sampler'] != SAMPLER or plan['target_update'] != TARGET:
        raise ValueError('Continuation inputs changed')
    verify_starter()
    if not (run / f'update_{TARGET:06d}.json').exists():
        subprocess.run(plan['command'], cwd=ROOT, check=True)
    target = verified_checkpoint(run, TARGET, manifest_hash)
    completion = read(run / 'completed.json')
    segment = read(run / f"segment_{completion['segment_id']}.json")
    if (completion['last_update'], segment['parent_checkpoint_sha256'], completion['manifest_sha256']) != (
            TARGET, sha256(checkpoint), manifest_hash):
        raise ValueError('Continuation receipt does not match the declared parent')
    print(json.dumps(dict(event='training_completed', **completion)), flush=True)
    reference = read(REFERENCE)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    trial = evaluate(target, MANIFEST, EXPORT, OUT, **SAMPLER)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=trial['evidence'])), flush=True)
    selected = None
    if quality_improved({'score': score_results([anchor])}, trial):
        selected = evaluate(target, MANIFEST, EXPORT, OUT, seeds=plan['selection_seeds'], **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=trial, candidate=selected,
                                              improves_reference=selected is not None and quality_improved(reference, selected)))


if __name__ == '__main__':
    main()

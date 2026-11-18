"""Check the 34k attention candidate on the two remaining development seeds."""

import sys
from pathlib import Path

from scripts.run_v002_dpmpp import read
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.training.optimize import evaluate, quality_improved


def main():
    out = Path('results/experiments/v009_attention16_extension34k')
    plan = Path('results/plans/v009_attention16_stability.json')
    manifest = Path('manifests/v009.csv')
    screen_path = out / 'screen.json'
    later_path = Path('results/experiments/v009_attention16_extension35k/screen.json')
    lead_path = Path('results/experiments/v007_freeu_cfg310/selection.json')
    screen, lead = (read(path)['candidate'] for path in (screen_path, lead_path))
    export = Path(screen['exported'])
    checkpoint = Path(screen['checkpoint'])
    inputs = (Path(__file__), screen_path, later_path, lead_path, manifest, checkpoint,
              Path('src/training/optimize.py'), Path('src/evaluation/run.py'))
    spec = dict(inputs={str(p): sha256(p) for p in inputs}, seeds=[0, 101, 202], n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.check_v009_attention16_stability'],
                rule='Check stability after improved FID and recall on seed 0. Require the established '
                     'three-seed gain, lower worst FID, at least the incumbent worst recall, and '
                     'a peak copy rate at or below the incumbent.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen stability inputs changed')
    if sha256(checkpoint) != screen['checkpoint_sha256'] or any(
            sha256(export / item) != expected for item, expected in screen['export_hashes'].items()):
        raise ValueError('Expected the exact seed-0 checkpoint and inference export')
    verify_starter()
    runtime = {key: screen[key] for key in ('weights', 'steps', 'guidance')}
    result = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
    metrics = [row['metrics'] for row in result['evidence']]
    control = [row['metrics'] for row in lead['evidence']]
    promotion = (
        quality_improved(lead, result)
        and max(row['fid'] for row in metrics) < max(row['fid'] for row in control)
        and min(row['recall'] for row in metrics) >= min(row['recall'] for row in control)
        and max(row['memorization_rate'] for row in metrics) <= max(row['memorization_rate'] for row in control)
    )
    write_json(out / 'stability.json', dict(plan_sha256=sha256(plan), screen_sha256=sha256(screen_path),
                                           candidate=result, improves_reference=promotion))
    print({'marks': [row['marks'] for row in result['evidence']], 'improves_reference': promotion}, flush=True)


if __name__ == '__main__':
    main()

"""Screen the two endpoint settings retained from the v001 development comparison."""

import json
from pathlib import Path

from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved

REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
PRIOR = Path('results/experiments/sampler_endpoint_screen_v001/screening.json')
PLAN = Path('results/plans/v002_sampler_comparison.json')
OUTPUT = Path('results/experiments/v002_sampler_comparison')
MANIFEST = Path('manifests/v002.csv')
SETTINGS = {
    'reference': {},
    'start_990': {'start_timestep': 990},
    'guidance_after_849': {'guidance_max_timestep': 849},
}
SAMPLER = {'weights': 'ema', 'steps': 50, 'guidance': 2.5}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    reference = read(REFERENCE)
    write_json(PLAN, {'created': timestamp(), 'reference_sha256': sha256(REFERENCE),
                      'prior_screen_sha256': sha256(PRIOR), 'manifest_sha256': sha256(MANIFEST),
                      'checkpoint_sha256': sha256(reference['checkpoint']),
                      'inference_source_sha256': sha256('submission/diffusion.py'),
                      'worker_sha256': sha256(__file__), 'settings': SETTINGS, 'sampler': SAMPLER,
                      'screen_n_per_class': 250, 'screen_seed': 0,
                      'selection_n_per_class': 1000, 'selection_seeds': [0, 101, 202],
                      'rule': 'Advance at most one valid setting that improves on the matched screen '
                              'by official tier floor or at least 0.05 worst normalized margin. '
                              'Advance it only if it also improves the three-seed reference by that rule.'})


def main():
    verify_starter()
    plan, reference = read(PLAN), read(REFERENCE)
    if (plan['reference_sha256'], plan['prior_screen_sha256'], plan['manifest_sha256'],
        plan['checkpoint_sha256'], plan['inference_source_sha256'], plan['worker_sha256']) != (
        sha256(REFERENCE), sha256(PRIOR), sha256(MANIFEST), sha256(reference['checkpoint']),
        sha256('submission/diffusion.py'), sha256(__file__)
    ):
        raise ValueError('Sampler comparison input changed')
    screened = {}
    for name, overrides in plan['settings'].items():
        exported = Path(reference['exported']) if not overrides else Path('private_build/v002_sampler_comparison') / name
        if not exported.exists():
            export_checkpoint(reference['checkpoint'], MANIFEST, exported, **plan['sampler'], **overrides)
        settings = read(exported / 'inference.json')['sampling']
        if any(settings[key] != value for key, value in overrides.items()):
            raise ValueError(f'Existing export has different timestep settings: {name}')
        screened[name] = evaluate(reference['checkpoint'], MANIFEST, exported, OUTPUT / 'screen' / name,
                                  seeds=(plan['screen_seed'],), n_per_class=plan['screen_n_per_class'],
                                  **plan['sampler'])
    promoted = max((name for name in screened if name != 'reference'
                    and quality_improved(screened['reference'], screened[name])),
                   key=lambda name: screened[name]['score'], default=None)
    write_json(OUTPUT / 'screening.json', {'plan_sha256': sha256(PLAN), 'screened': screened,
                                         'promoted': promoted})
    selected = None
    if promoted:
        selected = evaluate(reference['checkpoint'], MANIFEST, screened[promoted]['exported'],
                            OUTPUT / 'selection' / promoted, seeds=plan['selection_seeds'],
                            n_per_class=plan['selection_n_per_class'], **plan['sampler'])
    write_json(OUTPUT / 'comparison.json', {'plan_sha256': sha256(PLAN), 'promoted': promoted,
                                           'candidate': selected,
                                           'improves_reference': selected is not None and
                                           quality_improved(reference, selected)})


if __name__ == '__main__':
    main()

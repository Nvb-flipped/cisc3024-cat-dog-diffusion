"""Test a bounded high-noise DDIM change and isolate the self-conditioning failure."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from scripts.run_v002_dpmpp import MANIFEST, REFERENCE, read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


PLAN = Path('results/plans/high_noise_probe.json')
OUT = Path('results/experiments/high_noise_probe')
EXPORT = Path('private_build/high_noise_probe')
REFINED = Path('results/experiments/rescale_followup/comparison.json')
SELF = Path('results/experiments/v006_self_condition/screening.json')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
THRESHOLD = dict(threshold_percentile=.995, threshold_min_timestep=950)


def feedback_probe(exported):
    from submission.diffusion import sample

    api = importlib.import_module('.'.join((*Path(exported).parts, 'generate')))
    model = api.load_model(torch.device('cuda'))
    labels = torch.arange(16, device='cuda') % 2
    options = model.sampling_config
    results = {}
    for feedback in (True, False):
        images = sample(model, labels, 17, **(options | {'self_condition_feedback': feedback}))
        if not torch.isfinite(images).all():
            raise ValueError('Nonfinite self-conditioning diagnostic')
        results[str(feedback).lower()] = dict(mean=images.mean().item(), std=images.std().item(),
                                             mean_pairwise_l2=torch.pdist(images.flatten(1)).mean().item())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REFERENCE)
    refined = read(REFINED)['candidate']
    trained_self = next(row for row in read(SELF)['screened'] if row['update'] == 30000)
    sources = (PLAN, REFERENCE, REFINED, SELF, MANIFEST, Path(reference['checkpoint']), Path(__file__),
               Path('submission/diffusion.py'), Path('src/release/export.py'), Path('scripts/run_v002_dpmpp.py'))
    inputs = {str(path): sha256(path) for path in sources if path != PLAN}
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, sampler=SAMPLER | THRESHOLD,
                              seeds=[0, 101, 202], source='https://arxiv.org/abs/2205.11487',
                              rule='Screen dynamic thresholding only for t>=950. Advance only if it improves '
                                   'the v002 rescale 0.25 seed-zero anchor by the declared rule.'))
        return
    plan = read(PLAN)
    if plan['inputs'] != inputs or plan['sampler'] != SAMPLER | THRESHOLD:
        raise ValueError('Frozen high-noise plan changed')
    if (OUT / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison')
    verify_starter()
    diagnostic = OUT / 'self_condition_feedback.json'
    if not diagnostic.exists():
        write_json(diagnostic, dict(scope='16-sample pixel diagnostic without official features or grading',
                                   seed=17, checkpoint_sha256=trained_self['checkpoint_sha256'],
                                   results=feedback_probe(trained_self['exported'])))
    if not EXPORT.exists():
        export_checkpoint(reference['checkpoint'], MANIFEST, EXPORT, **plan['sampler'])
    config = read(EXPORT / 'inference.json')['sampling']
    if any(config.get(key) != value for key, value in THRESHOLD.items()):
        raise ValueError('Exported threshold does not match the frozen plan')
    if not (OUT / 'interface.json').exists():
        write_json(OUT / 'interface.json', verify_interface(reference['exported'], EXPORT) | THRESHOLD | {'solver': 'ddim'})
    trial = evaluate(reference['checkpoint'], MANIFEST, EXPORT, OUT, **SAMPLER)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=trial['evidence'])), flush=True)
    anchor = validated_result(refined['evidence'][0]['path'], 0, refined['exported'])
    selected = None
    if quality_improved({'score': score_results([anchor])}, trial):
        selected = evaluate(reference['checkpoint'], MANIFEST, EXPORT, OUT, seeds=plan['seeds'], **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=trial, candidate=selected,
                                             improves_reference=selected is not None and quality_improved(refined, selected)))


if __name__ == '__main__':
    main()

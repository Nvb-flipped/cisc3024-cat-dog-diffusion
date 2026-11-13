"""Screen a single DPM-Solver++ setting against the immutable v002 DDIM export."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
MANIFEST = Path('manifests/v002.csv')
PLAN = Path('results/plans/v002_dpmpp.json')
EXPORT = Path('private_build/v002_dpmpp')
OUT = Path('results/experiments/v002_dpmpp')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
SOLVER = 'dpmpp_2m'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_interface(reference, current):
    from submission.diffusion import sample

    reference_api = importlib.import_module('.'.join((*Path(reference).parts, 'generate')))
    candidate_api = importlib.import_module('.'.join((*Path(current).parts, 'generate')))
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    baseline = reference_api.load_model(device)
    model = candidate_api.load_model(device)
    labels = torch.arange(35, device=device) % 2
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    if not torch.equal(reference_api.generate(baseline, labels, 17),
                       sample(model, labels, 17, **baseline.sampling_config)):
        raise ValueError('The default DDIM path changed from the immutable v002 export')
    output = candidate_api.generate(model, labels, 17)
    same_seed = candidate_api.generate(model, labels, 17)
    other_seed = candidate_api.generate(model, labels, 29)
    prefix = candidate_api.generate(model, labels[:1], 17)
    regrouped = sample(model, labels, 17, **(model.sampling_config | {'microbatch': 7}))
    altered = torch.where(torch.arange(len(labels), device=device) % 2 == 1, 1 - labels, labels)
    altered_output = candidate_api.generate(model, altered, 17)
    if (output.shape != (35, 3, 32, 32) or not torch.isfinite(output).all()
            or not (output.ge(0) & output.le(1)).all() or not torch.equal(output, same_seed)
            or torch.equal(output, other_seed) or not torch.equal(output[:1], prefix)
            or not torch.equal(output, regrouped) or not torch.equal(output[::2], altered_output[::2])
            or not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
        raise ValueError('Trained DPM-Solver++ CUDA interface check failed')
    return dict(solver=SOLVER, steps=SAMPLER['steps'], device='cuda', count=35, microbatches=[32, 7],
                exact_ddim_reference=True, deterministic_and_label_isolated=True, caller_rng_preserved=True,
                candidate_provenance_sha256=sha256(current / 'provenance.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REFERENCE)
    paths = (REFERENCE, MANIFEST, Path(reference['checkpoint']), Path('submission/diffusion.py'),
             Path('src/release/export.py'), Path(__file__))
    inputs = {str(path): sha256(path) for path in paths}
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, dict(created=timestamp(), inputs=inputs, sampler=SAMPLER | {'solver': SOLVER},
                              grid='Identical discrete uniform timestep indices as DDIM50. Fixed [-1,1] x0 clipping, '
                                   'first-order bootstrap and final clean projection.',
                              seeds=[0, 101, 202], n_per_class=1000,
                              source='https://arxiv.org/abs/2211.01095',
                              rule='Advance by the existing hard gate, score tier or 0.05 margin gain.'))
        return
    plan = read(PLAN)
    if inputs != plan['inputs'] or plan['sampler'] != SAMPLER | {'solver': SOLVER}:
        raise ValueError('Declared solver inputs changed')
    verify_starter()
    if not EXPORT.exists():
        export_checkpoint(reference['checkpoint'], MANIFEST, EXPORT, **plan['sampler'])
    if read(EXPORT / 'inference.json')['sampling']['solver'] != SOLVER:
        raise ValueError('Export solver does not match the plan')
    if not (OUT / 'interface.json').exists():
        write_json(OUT / 'interface.json', verify_interface(reference['exported'], EXPORT))
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screened = evaluate(reference['checkpoint'], MANIFEST, EXPORT, OUT, **SAMPLER)
    print(json.dumps({'event': 'official_evaluation_completed', 'evidence': screened['evidence']}), flush=True)
    selected = None
    if quality_improved({'score': score_results([anchor])}, screened):
        selected = evaluate(reference['checkpoint'], MANIFEST, EXPORT, OUT, seeds=plan['seeds'], **SAMPLER)
        print(json.dumps({'event': 'official_selection_completed', 'evidence': selected['evidence'][1:]}), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=screened, candidate=selected,
                                              improves_reference=selected is not None and quality_improved(reference, selected)))


if __name__ == '__main__':
    main()

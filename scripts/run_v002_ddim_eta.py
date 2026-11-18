"""Compare two seeded DDIM reverse-path variances against the immutable v002 export."""

import importlib
import json
import sys
from pathlib import Path

import torch

from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved

REFERENCE = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
PLAN = Path('results/plans/v002_ddim_eta.json')
MANIFEST = Path('manifests/v002.csv')
OUT = Path('results/experiments/v002_ddim_eta')
SAMPLER = {'weights': 'ema', 'steps': 50, 'guidance': 2.5}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_sampler(eta, reference_export):
    from submission.diffusion import sample

    folder = Path(reference_export)
    reference = importlib.import_module('.'.join((*folder.parts, 'generate')))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = reference.load_model(torch.device('cuda'))
    labels = torch.arange(35, device='cuda') % 2
    options = model.sampling_config
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    old = reference.generate(model, labels, 17)
    if not torch.equal(old, sample(model, labels, 17, **options)):
        raise ValueError('The zero-eta path changed from the verified reference')
    options = options | {'eta': eta}
    original = sample(model, labels, 17, **options)
    regrouped = sample(model, labels, 17, **(options | {'microbatch': 7}))
    single = sample(model, labels[:1], 17, **options)
    changed = torch.cat((labels[:1], 1 - labels[1:]))
    other_labels = sample(model, changed, 17, **options)
    other_seed = sample(model, labels, 29, **options)
    if (not torch.equal(original, regrouped) or not torch.equal(original[:1], single)
        or not torch.equal(original[:1], other_labels[:1]) or torch.equal(original, other_seed)
        or not torch.equal(cpu, torch.get_rng_state()) or not torch.equal(cuda, torch.cuda.get_rng_state())
        or original.shape != (35, 3, 32, 32) or not torch.isfinite(original).all()
        or not (original.ge(0) & original.le(1)).all()):
        raise ValueError('Seeded reverse noise violated the CUDA interface')
    return {'eta': eta, 'device': 'cuda', 'count': 35, 'microbatches': [32, 7],
            'exact_prefix_isolation_and_rng': True, 'exact_zero_eta_reference': True,
            'reference_export_provenance_sha256': sha256(Path(reference_export) / 'provenance.json'),
            'sampler_sha256': sha256('submission/diffusion.py')}


def main():
    reference = read(REFERENCE)
    sources = (REFERENCE, MANIFEST, Path(reference['checkpoint']), Path('submission/diffusion.py'),
               Path('src/release/export.py'), Path(__file__))
    inputs = {str(path): sha256(path) for path in sources}
    if sys.argv[1:] == ['--prepare']:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, {'created': timestamp(), 'inputs': inputs, 'sampler': SAMPLER,
                          'eta_values': [0.35, 0.70], 'seeds': [0, 101, 202], 'n_per_class': 1000,
                          'reference': 'https://arxiv.org/html/2010.02502',
                          'rule': 'Advance at most one valid seed-0 improvement by tier floor or 0.05 margin.'})
        return
    plan = read(PLAN)
    if inputs != plan['inputs']:
        raise ValueError('Seeded DDIM experiment inputs changed')
    seed0 = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    anchor = {'score': score_results([seed0])}
    trials = []
    for eta in plan['eta_values']:
        name = f'eta_{int(eta * 100):02}'
        out = OUT / name
        exported = Path('private_build/v002_ddim_eta') / name
        probe = out / 'interface.json'
        if not probe.exists():
            write_json(probe, verify_sampler(eta, reference['exported']))
        if not exported.exists():
            export_checkpoint(reference['checkpoint'], MANIFEST, exported, **plan['sampler'], eta=eta)
        if read(exported / 'inference.json')['sampling']['eta'] != eta:
            raise ValueError(f'Export eta differs from the plan: {exported}')
        trial = evaluate(reference['checkpoint'], MANIFEST, exported, out, n_per_class=plan['n_per_class'],
                         **plan['sampler'])
        trials.append(trial | {'eta': eta, 'name': name})
    promoted = max((trial for trial in trials if quality_improved(anchor, trial)),
                   key=lambda trial: trial['score'], default=None)
    selected = None
    if promoted:
        selected = evaluate(reference['checkpoint'], MANIFEST, promoted['exported'], OUT / promoted['name'],
                            seeds=plan['seeds'], n_per_class=plan['n_per_class'], **plan['sampler'])
    write_json(OUT / 'comparison.json', {'plan_sha256': sha256(PLAN), 'trials': trials, 'promoted': promoted,
                                        'candidate': selected, 'improves_reference': selected is not None and
                                        quality_improved(reference, selected)})


if __name__ == '__main__':
    main()

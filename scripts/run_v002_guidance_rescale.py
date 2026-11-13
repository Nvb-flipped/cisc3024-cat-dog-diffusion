"""Full matched DDIM evaluation of one prespecified CFG rescale strength."""

import argparse
import json
from pathlib import Path

from scripts.run_v002_dpmpp import MANIFEST, REFERENCE, read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


PLAN = Path('results/plans/v002_guidance_rescale.json')
EXPORT = Path('private_build/v002_guidance_rescale')
OUT = Path('results/experiments/v002_guidance_rescale')
SAMPLER = dict(weights='ema', steps=50, guidance=2.5)
RESCALE = .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    reference = read(REFERENCE)
    inputs = [REFERENCE, MANIFEST, Path(reference['checkpoint']), Path(__file__),
              Path('submission/diffusion.py'), Path('src/release/export.py'),
              Path('scripts/run_v002_dpmpp.py'), Path('src/evaluation/run.py')]
    hashes = {str(path): sha256(path) for path in inputs}
    if args.prepare:
        if PLAN.exists():
            raise FileExistsError(PLAN)
        write_json(PLAN, dict(created=timestamp(), inputs=hashes, sampler=SAMPLER | {'guidance_rescale': RESCALE},
                              seeds=[0, 101, 202], n_per_class=1000,
                              method='Per-image epsilon standard deviation matching, mixed 50% with ordinary CFG.',
                              source='https://arxiv.org/html/2305.08891#S3.SS4',
                              rule='Advance only on the existing valid seed-0 tier or margin improvement.'))
        return
    plan = read(PLAN)
    if plan['inputs'] != hashes or plan['sampler'] != SAMPLER | {'guidance_rescale': RESCALE}:
        raise ValueError('Frozen guidance rescale inputs changed')
    verify_starter()
    if not EXPORT.exists():
        export_checkpoint(reference['checkpoint'], MANIFEST, EXPORT, **plan['sampler'])
    if read(EXPORT / 'inference.json')['sampling']['guidance_rescale'] != RESCALE:
        raise ValueError('Export differs from the declared rescale strength')
    interface = OUT / 'interface.json'
    if not interface.exists():
        record = verify_interface(reference['exported'], EXPORT)
        record.update(solver='ddim', guidance_rescale=RESCALE)
        write_json(interface, record)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    trial = evaluate(reference['checkpoint'], MANIFEST, EXPORT, OUT, **SAMPLER)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=trial['evidence'])), flush=True)
    selected = None
    if quality_improved({'score': score_results([anchor])}, trial):
        selected = evaluate(reference['checkpoint'], MANIFEST, EXPORT, OUT, seeds=plan['seeds'], **SAMPLER)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    write_json(OUT / 'comparison.json', dict(plan_sha256=sha256(PLAN), screened=trial, candidate=selected,
                                             improves_reference=selected is not None and quality_improved(reference, selected)))


if __name__ == '__main__':
    main()

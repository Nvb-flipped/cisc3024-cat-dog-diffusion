"""Evaluate one CFG increase on the v007 half-strength PAG checkpoint."""

import json
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    control_file = Path('results/experiments/v007_pag_half/comparison.json')
    reference_file = Path('results/experiments/rescale_followup/comparison.json')
    plan = Path('results/plans/v007_pag_cfg3.json')
    out = Path('results/experiments/v007_pag_cfg3')
    export = Path('private_build/v007_pag_cfg3')
    manifest = Path('manifests/v007.csv')
    control = read(control_file)['screened']
    reference = read(reference_file)['candidate']
    checkpoint = Path(control['checkpoint'])
    sampler = dict(weights='ema', steps=50, guidance=3.0)
    options = dict(pag_scale=.5)
    seeds = [0, 101, 202]
    inputs = (control_file, reference_file, manifest, checkpoint, Path(__file__),
              Path('submission/model.py'), Path('submission/diffusion.py'), Path('src/release/export.py'),
              Path('src/training/optimize.py'), Path('src/evaluation/run.py'))
    spec = dict(inputs={str(path): sha256(path) for path in inputs}, sampler=sampler | options,
                selection_seeds=seeds, n_per_class=1000)
    if not plan.exists():
        write_json(plan, spec | dict(created=timestamp(), rule='Change only CFG from 2.5 to 3.0 on v007 PAG 0.5. '
                   'Test whether accuracy reaches 90% while all hard checks pass. Advance only a valid '
                   'seed-0 improvement versus the retained v002 anchor.'))
    elif any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen CFG comparison changed')
    if (out / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison')
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **sampler, **options)
    if read(export / 'inference.json')['sampling'].get('pag_scale') != .5:
        raise ValueError('PAG scale differs from the paired control')
    if not (out / 'interface.json').exists():
        checks = verify_interface(control['exported'], export)
        write_json(out / 'interface.json', checks | options | {'solver': 'ddim'})
    screened = evaluate(checkpoint, manifest, export, out, **sampler)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=screened['evidence'])), flush=True)
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    selected = None
    if quality_improved({'score': score_results([anchor])}, screened):
        selected = evaluate(checkpoint, manifest, export, out, seeds=seeds, **sampler)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(out / 'comparison.json', dict(plan_sha256=sha256(plan), screened=screened, candidate=selected,
                                            improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

"""Test one small CFG increase on the v007 half-strength PAG export."""

import json
from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    control_path = Path('results/experiments/v007_pag_cfg3/comparison.json')
    reference_path = Path('results/experiments/rescale_followup/comparison.json')
    out = Path('results/experiments/v007_pag_cfg325')
    export = Path('private_build/v007_pag_cfg325')
    plan_path = Path('results/plans/v007_pag_cfg325.json')
    manifest = Path('manifests/v007.csv')
    control = read(control_path)['screened']
    reference = read(reference_path)['candidate']
    checkpoint = Path(control['checkpoint'])
    seeds = [0, 101, 202]
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    pag = dict(pag_scale=.5)
    inputs = (control_path, reference_path, manifest, checkpoint, Path(__file__),
              Path('submission/diffusion.py'), Path('submission/model.py'), Path('src/release/export.py'),
              Path('src/training/optimize.py'), Path('src/evaluation/run.py'), Path('scripts/run_v002_dpmpp.py'))
    spec = dict(inputs={str(path): sha256(path) for path in inputs}, sampler=sampler | pag,
                selection_seeds=seeds, n_per_class=1000)
    if not plan_path.exists():
        write_json(plan_path, spec | dict(created=timestamp(), rule='Only change CFG from 3.0 to 3.25. '
                   'This measures whether the previous FID and accuracy gains continue within the copy limit. '
                   'Run the full seed 0 first. Advance only a valid improvement over v002 rescale 0.25.'))
    elif any(read(plan_path)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen CFG comparison changed')
    if (out / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison')
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **sampler, **pag)
    if read(export / 'inference.json')['sampling'].get('pag_scale') != pag['pag_scale']:
        raise ValueError('Unmatched PAG scale')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(control['exported'], export) | pag | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screened = evaluate(checkpoint, manifest, export, out, **sampler)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=screened['evidence'])), flush=True)
    candidate = None
    if quality_improved({'score': score_results([anchor])}, screened):
        candidate = evaluate(checkpoint, manifest, export, out, seeds=seeds, **sampler)
        print(json.dumps(dict(event='official_selection_completed', evidence=candidate['evidence'][1:])), flush=True)
    improved = candidate is not None and quality_improved(reference, candidate)
    write_json(out / 'comparison.json', dict(plan_sha256=sha256(plan_path), screened=screened,
                                            candidate=candidate, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

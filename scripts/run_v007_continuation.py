"""Resume exact v007 for one interval and compare the fixed PAG sampler."""

import json
import subprocess
from pathlib import Path

from scripts.run_coverage_comparison import read, verified_checkpoint
from scripts.run_v002_dpmpp import verify_interface
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    parent_path = Path('results/plans/v007_coverage.json')
    control_path = Path('results/experiments/v007_pag_cfg325/comparison.json')
    reference_path = Path('results/experiments/rescale_followup/comparison.json')
    name = 'v007_continuation_032000'
    plan_path = Path(f'results/plans/{name}.json')
    out = Path(f'results/experiments/{name}')
    export = Path(f'private_build/{name}')
    manifest = Path('manifests/v007.csv')
    control = read(control_path)['screened']
    reference = read(reference_path)['candidate']
    parent = read(parent_path)
    for path, digest in parent['inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'Training lineage input changed: {path}')
    command = parent['command'].copy()
    run = Path(command[command.index('--out') + 1])
    digest = sha256(manifest)
    checkpoint = verified_checkpoint(run, 30000, digest)
    if sha256(checkpoint) != control['checkpoint_sha256']:
        raise ValueError('Resume point differs from the matched sampler control')
    command[command.index('--updates') + 1] = '32000'
    command.extend(['--resume', str(checkpoint)])
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    pag = dict(pag_scale=.5)
    seeds = [0, 101, 202]
    inputs = (parent_path, control_path, reference_path, manifest, checkpoint, Path(__file__),
              Path('scripts/run_v002_dpmpp.py'))
    spec = dict(inputs={str(path): sha256(path) for path in inputs}, command=command,
                sampler=sampler | pag, selection_seeds=seeds, n_per_class=1000, target_update=32000)
    if not plan_path.exists():
        write_json(plan_path, spec | dict(created=timestamp(), rule='One continuation from the exact v007 30k state. '
                   'Keep the 30k DDIM50/CFG3.25/PAG0.5 sampler fixed. Advance only a valid improvement over v002.'))
    else:
        plan = read(plan_path)
        if any(plan[key] != value for key, value in spec.items()):
            raise ValueError('Frozen continuation changed')
    if (out / 'comparison.json').exists():
        raise FileExistsError('Preserve the completed comparison')
    if not (run / 'update_032000.json').exists():
        subprocess.run(command, cwd=ROOT, check=True)
    target = verified_checkpoint(run, 32000, digest)
    completed = read(run / 'completed.json')
    segment = read(run / f"segment_{completed['segment_id']}.json")
    if (completed['last_update'], segment['parent_checkpoint_sha256'], completed['manifest_sha256']) != (
            32000, sha256(checkpoint), digest):
        raise ValueError('Continuation receipt differs from its frozen parent')
    print(json.dumps(dict(event='training_completed', **completed)), flush=True)
    if not export.exists():
        export_checkpoint(target, manifest, export, **sampler, **pag)
    if read(export / 'inference.json')['sampling'].get('pag_scale') != pag['pag_scale']:
        raise ValueError('PAG scale differs from the comparison plan')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | pag | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screened = evaluate(target, manifest, export, out, **sampler)
    print(json.dumps(dict(event='official_evaluation_completed', update=32000, evidence=screened['evidence'])), flush=True)
    selected = None
    if quality_improved({'score': score_results([anchor])}, screened):
        selected = evaluate(target, manifest, export, out, seeds=seeds, **sampler)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(out / 'comparison.json', dict(plan_sha256=sha256(plan_path), screened=screened,
                                            candidate=selected, improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

"""Train one exact-pool noise-input-perturbation challenger against v007."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import read, verified_checkpoint
from scripts.run_v002_dpmpp import verify_interface
from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v007_input_perturbation'
    manifest = Path('manifests/v007.csv')
    holdout = manifest.with_suffix('.validation.csv')
    config_path = Path(f'configs/runs/{name}.json')
    run = Path(f'checkpoints/{name}_seed17')
    out = Path(f'results/experiments/{name}')
    plan_path = Path(f'results/plans/{name}.json')
    export = Path(f'private_build/{name}')
    diagnostic_path = Path(f'results/diagnostics/{name}/result.json')
    reference_path = Path('results/experiments/v007_32k_pag015/comparison.json')
    reference = read(reference_path)['candidate']
    parent_run = Path(reference['checkpoint']).parent / 'run.json'
    config = read(parent_run)['config']
    config['training']['noise_input_perturbation'] = .1
    diagnostic = read(diagnostic_path)
    if (diagnostic['noise_input_perturbation'], diagnostic['updates'], diagnostic['passed'],
            diagnostic['diagnostic_only']) != (.1, 2000, True, True):
        raise ValueError('A passing fixed diagnostic with the exact perturbation is required')
    if any(sha256(path) != digest for path, digest in diagnostic['perturbation_sources'].items()):
        raise ValueError('Training code changed since the perturbation diagnostic')
    if not config_path.exists():
        write_json(config_path, config)
    if read(config_path) != config:
        raise ValueError('Only the added input noise may differ from v007')
    sampler = dict(weights='ema', steps=50, guidance=3.25)
    pag = dict(pag_scale=.15)
    seeds = [0, 101, 202]
    paths = (Path(__file__), reference_path, parent_run, manifest, manifest.with_suffix('.provenance.json'),
             holdout, config_path, diagnostic_path, Path('results/curation/v007_cache_parity.json'),
             *map(Path, ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                         'src/training/lineage.py', 'src/training/validation.py', 'submission/model.py',
                         'submission/diffusion.py', 'submission/generate.py', 'src/release/export.py',
                         'scripts/run_v002_dpmpp.py', 'src/training/optimize.py', 'src/evaluation/run.py')))
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config_path),
               '--manifest', str(manifest), '--validation-manifest', str(holdout), '--seed', '17',
               '--updates', '32000', '--out', str(run)]
    spec = dict(inputs={str(path): sha256(path) for path in paths}, command=command,
                sampler=sampler | pag, selection_seeds=seeds, n_per_class=1000)
    if not plan_path.exists():
        write_json(plan_path, spec | dict(created=timestamp(), source='https://proceedings.mlr.press/v202/ning23a.html',
                   rule='At training only, add 0.1 times independent Gaussian noise to the forward noise input. '
                        'Keep the exact x0 crop, original epsilon target, sampler and full 32k update budget. '
                        'Advance one matched full seed-0 improvement only.'))
    elif any(read(plan_path)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen input perturbation comparison changed')
    if (out / 'comparison.json').exists():
        raise FileExistsError('Preserve completed evaluation evidence')
    verify_starter()
    digest = sha256(manifest)
    complete = run / 'completed.json'
    if not complete.exists():
        if (run / 'run.json').exists():
            last = sorted(run.glob('update_*.json'))[-1]
            command.extend(['--resume', str(verified_checkpoint(run, read(last)['update'], digest))])
        subprocess.run(command, cwd=ROOT, check=True)
    finished = read(complete)
    initial_path, = run.glob('segment_000000_*.json')
    initial = read(initial_path)
    if (finished['last_update'], finished['manifest_sha256'], initial['parent_checkpoint_sha256'],
            initial['seed'], initial['config_sha256'], initial['validation']['sha256']) != (
            32000, digest, None, 17, sha256(config_path), sha256(holdout)):
        raise ValueError('Training receipts differ from the approved exact comparison')
    print(json.dumps(dict(event='training_completed', **finished)), flush=True)
    checkpoint = verified_checkpoint(run, 32000, digest)
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **sampler, **pag)
    if read(export / 'inference.json')['sampling'].get('pag_scale') != .15:
        raise ValueError('Sampler must match the v007 anchor')
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | pag | {'solver': 'ddim'})
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    screened = evaluate(checkpoint, manifest, export, out, **sampler)
    print(json.dumps(dict(event='official_evaluation_completed', evidence=screened['evidence'])), flush=True)
    selected = None
    if quality_improved({'score': score_results([anchor])}, screened):
        selected = evaluate(checkpoint, manifest, export, out, seeds=seeds, **sampler)
        print(json.dumps(dict(event='official_selection_completed', evidence=selected['evidence'][1:])), flush=True)
    improved = selected is not None and quality_improved(reference, selected)
    write_json(out / 'comparison.json', dict(plan_sha256=sha256(plan_path), screened=screened, candidate=selected,
                                            improves_reference=improved))
    print(json.dumps(dict(event='comparison_completed', improves_reference=improved)), flush=True)


if __name__ == '__main__':
    main()

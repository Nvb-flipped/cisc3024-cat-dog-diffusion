"""Export declared sampler candidates and invoke the untouched official evaluator."""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.release.export import export_checkpoint
from src.release.check import check_files
from src.evaluation.run import run_official


def score_results(results, *, n_per_class=1000):
    """Project plan's prespecified lexicographic selection rule, across all selection seeds."""
    if not results:
        raise ValueError('No evaluation evidence')
    for result in results:
        if not all(isinstance(result['metrics'].get(name), (float, int)) and
                   math.isfinite(result['metrics'][name]) for name in
                   ['accuracy', 'fid', 'kid', 'kid_std', 'precision', 'recall', 'memorization_rate']):
            return None
        if result['violations'] or not result['marks']['valid']:
            return None
        if not result['training_data'].get('stats', {}).get('valid_rows') or result['training_data'].get('errors'):
            return None
        metrics, resources = result['metrics'], result['resources']
        if any(not 0 <= metrics[key] <= 1 for key in ['accuracy', 'precision', 'recall']) or \
           not 0 <= metrics['memorization_rate'] <= .05 or metrics['fid'] < 0 or metrics['kid_std'] < 0:
            return None
        if any(not isinstance(resources.get(key), (float, int)) or not math.isfinite(resources[key])
               for key in ['seconds_per_1000', 'num_parameters', 'submission_mb']):
            return None
        if resources['seconds_per_1000'] <= 0 or not 0 <= resources['num_parameters'] <= 20_000_000 or \
           not 0 <= resources['submission_mb'] <= 100:
            return None
        if result['metrics']['num_samples'] != 2 * n_per_class:
            raise ValueError(f'Comparison requires exactly {n_per_class} samples per class')
    tier_floor = min(r['marks']['total'] for r in results)
    margin = min(min((r['metrics']['accuracy'] - .90) / .05,
                     (50 - r['metrics']['fid']) / 10,
                     (r['metrics']['recall'] - .40) / .10) for r in results)
    memorization = max(r['metrics']['memorization_rate'] for r in results)
    speed = max(r['resources']['seconds_per_1000'] for r in results)
    return tier_floor, margin, -memorization, -speed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/evaluation.yaml')
    p.add_argument('--phase', choices=['screening', 'selection', 'confirmation'], default='screening')
    p.add_argument('--plan', action='store_true', help='Write commands without running them')
    p.add_argument('--winner', type=Path, help='Immutable winner.json required for confirmation')
    args = p.parse_args()
    config = read_config(args.config)
    if not config['candidates']:
        raise ValueError('No trained candidates declared; add actual checkpoint/manifest paths after training')
    phase = args.phase
    if phase == 'selection' and len(config['candidates']) > 3:
        raise ValueError('At most three checkpoint/sampler combinations may be promoted to selection')
    if phase == 'confirmation':
        if len(config['candidates']) != 1 or args.winner is None:
            raise ValueError('Confirmation requires exactly one frozen winner and --winner winner.json')
        winner = json.loads(args.winner.read_text())
        if config['candidates'][0] != winner['winner']['candidate']:
            raise ValueError('Confirmation candidate differs from frozen winner')
        if config['confirmation_seeds'] != winner['confirmation_seeds']:
            raise ValueError('Confirmation seeds changed after freeze')
        folder = Path(winner['winner']['exported'])
        for name, expected in winner['export_hashes'].items():
            if sha256(folder / name) != expected:
                raise ValueError(f'Frozen winner changed: {name}')
    seeds = config[f'{phase}_seeds']
    count = config['screening_n_per_class'] if phase == 'screening' else config['selection_n_per_class']
    out = Path(config['output']) / phase
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    for candidate in config['candidates']:
        # Selection/confirmation candidates must pin their exact previously-screened sampler.
        if phase == 'screening':
            variants = itertools.product(config['weights'], config['steps'], config['guidance'])
        else:
            variants = [(candidate['weights'], candidate['steps'], candidate['guidance'])]
        for weights, steps, guidance in variants:
            name = f'{candidate["name"]}_{weights}_s{steps}_g{guidance:g}'
            exported = ROOT / 'private_build/evaluation' / name
            if not args.plan:
                if not (exported / 'weights.pt').exists():
                    export_checkpoint(candidate['checkpoint'], candidate['manifest'], exported,
                        weights=weights, steps=steps, guidance=guidance, microbatch=config['microbatch'],
                        execution_batch=config['execution_batch'])
                else:
                    provenance = json.loads((exported / 'provenance.json').read_text())
                    inference = json.loads((exported / 'inference.json').read_text())
                    if provenance['checkpoint_sha256'] != sha256(candidate['checkpoint']) or \
                       provenance['manifest_sha256'] != sha256(candidate['manifest']) or \
                       provenance['weights_variant'] != weights or \
                       inference['sampling']['steps'] != steps or inference['sampling']['guidance'] != guidance or \
                       inference['sampling']['microbatch'] != config['microbatch'] or \
                       inference['sampling'].get('execution_batch', 32) != config['execution_batch']:
                        raise ValueError(f'Candidate name reused for different state: {name}')
                check_files(exported)
            for seed in seeds:
                result = out / f'{name}_seed{seed}.json'
                command = [sys.executable, str(ROOT / 'starter_kit/evaluate.py'), '--submission', str(exported),
                    '--grading', '--n-per-class', str(count), '--seed', str(seed), '--batch-size',
                    str(config['batch_size']), '--device', config['device'], '--data-root',
                    str((ROOT / config['data_root']).resolve()), '--out', str(result.resolve())]
                commands.append({'candidate': name, 'seed': seed, 'command': command, 'result': str(result)})
    write_json(out / 'commands.json', {'created': timestamp(), 'phase': phase, 'commands': commands,
               'config_sha256': sha256(args.config), 'protocol': 'unchanged official evaluator --submission --grading'})
    if args.plan:
        return print(json.dumps(commands, indent=2))
    for row in commands:
        result = Path(row['result'])
        if result.exists():
            raise FileExistsError(f'Original evaluation output already exists: {result}; use a fresh output version')
        print(f'Running {row["candidate"]}, seed {row["seed"]}', flush=True)
        run_official(row['command'][row['command'].index('--submission') + 1], result,
            seed=row['seed'], n_per_class=count, batch_size=config['batch_size'],
            data_root=ROOT / config['data_root'], device=config['device'])


if __name__ == '__main__':
    main()

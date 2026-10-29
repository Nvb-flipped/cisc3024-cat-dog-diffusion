"""Export declared sampler candidates and invoke the untouched official evaluator."""
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.release.export import export_checkpoint


def score_results(results):
    """Project plan's prespecified lexicographic selection rule, across all selection seeds."""
    if not results:
        raise ValueError('No evaluation evidence')
    for result in results:
        if result['violations'] or not result['marks']['valid']:
            return None
        if result['metrics']['memorization_rate'] is None or not result['training_data'].get('stats'):
            return None
        if result['metrics']['num_samples'] != 2000:
            raise ValueError('Full selection ranking requires exactly 1000 per class')
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
    args = p.parse_args()
    config = read_config(args.config)
    if not config['candidates']:
        raise ValueError('No trained candidates declared; add actual checkpoint/manifest paths after training')
    phase = args.phase
    if phase == 'selection' and len(config['candidates']) > 3:
        raise ValueError('At most three checkpoint/sampler combinations may be promoted to selection')
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
                        weights=weights, steps=steps, guidance=guidance, microbatch=config['microbatch'])
                else:
                    provenance = json.loads((exported / 'provenance.json').read_text())
                    inference = json.loads((exported / 'inference.json').read_text())
                    if provenance['checkpoint_sha256'] != sha256(candidate['checkpoint']) or \
                       provenance['manifest_sha256'] != sha256(candidate['manifest']) or \
                       provenance['weights_variant'] != weights or \
                       inference['sampling']['steps'] != steps or inference['sampling']['guidance'] != guidance:
                        raise ValueError(f'Candidate name reused for different state: {name}')
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
    child_env = dict(os.environ) | {'TORCH_HOME': str(ROOT / 'models/evaluator')}
    for row in commands:
        result = Path(row['result'])
        if result.exists():
            raise FileExistsError(f'Original evaluation output already exists: {result}; use a fresh output version')
        print(f'Running {row["candidate"]}, seed {row["seed"]}', flush=True)
        with result.with_suffix('.stdout.log').open('w', encoding='utf-8') as log:
            subprocess.run(row['command'], cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = json.loads(result.read_text())
        if actual['metrics']['memorization_rate'] is None:
            raise ValueError('Evaluation omitted memorization; not valid evidence')
        write_json(result.with_suffix('.provenance.json'), {'result_sha256': sha256(result),
                   'candidate': row['candidate'], 'seed': row['seed'], 'command': row['command'],
                   'violations': actual['violations']})


if __name__ == '__main__':
    main()

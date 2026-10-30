"""Materialize/run the plan's equal-budget LR comparison, reusing the 2e-4 baseline."""
import argparse
import copy
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
import torch
import yaml
from src.common import ROOT, read_config, sha256, timestamp, write_json

LEARNING_RATES = [2e-4, 1e-4, 3e-4]


def configurations(base):
    recipes = []
    for rate in LEARNING_RATES:
        config = copy.deepcopy(base)
        config['training']['updates'] = 50000
        config['training']['learning_rate'] = rate
        recipes.append({'learning_rate': rate, 'training_seed': 17, 'updates': 50000,
                        'is_reused_baseline': rate == 2e-4, 'config': config})
    return recipes


def materialize(manifest, base_path='configs/ddpm.yaml', *, output='results/plans'):
    manifest, base_path = Path(manifest).resolve(), Path(base_path).resolve()
    base = read_config(base_path)
    recipes = configurations(base)
    if not manifest.exists():
        return {'status': 'pending_personally_reviewed_manifest', 'manifest': str(manifest),
                'recipes': [{key: value for key, value in recipe.items() if key != 'config'} for recipe in recipes],
                'commands': [], 'scope': 'Declared experiment recipes only; no training or quality result.'}
    digest = sha256(manifest)
    provenance = json.loads(manifest.with_suffix('.provenance.json').read_text())
    if provenance.get('origin') != 'student_reviewed_manifest' or provenance['manifests'][0]['sha256'] != digest:
        raise ValueError('LR comparison requires the exact frozen personally reviewed baseline manifest')
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        counts = Counter(int(row['label']) for row in csv.DictReader(stream))
    if min(counts[0], counts[1]) < 500:
        raise ValueError('LR comparison needs at least 500 reviewed training heads per class')
    config_folder = ROOT / 'configs/runs' / digest[:10]
    config_folder.mkdir(parents=True, exist_ok=True)
    commands = []
    for recipe in recipes:
        name = f'lr_{recipe["learning_rate"]:g}_seed17_u050000_{digest[:10]}'
        config_path = config_folder / f'{name}.yaml'
        source = yaml.safe_dump(recipe['config'], sort_keys=False)
        if config_path.exists() and config_path.read_text(encoding='utf-8') != source:
            raise ValueError(f'Frozen experiment configuration differs: {config_path}')
        if not config_path.exists():
            config_path.write_text(source, encoding='utf-8', newline='\n')
        run = ROOT / 'checkpoints' / (f'ddpm_seed17_{digest[:10]}' if recipe['is_reused_baseline'] else name)
        command = [sys.executable, '-m', 'src.training.train', '--config', str(config_path),
                   '--manifest', str(manifest), '--seed', '17', '--out', str(run)]
        commands.append({'name': name, 'learning_rate': recipe['learning_rate'],
            'is_reused_baseline': recipe['is_reused_baseline'], 'config': str(config_path),
            'config_sha256': sha256(config_path), 'command': command, 'output': str(run),
            'final_checkpoint': str(run / 'update_050000.pt'),
            'images_seen': recipe['updates'] * base['training']['effective_batch']})
    plan = {'manifest': str(manifest), 'manifest_sha256': digest,
        'training_by_class': {str(label): counts[label] for label in [0, 1]},
        'base_config_sha256': sha256(base_path), 'training_seed': 17, 'updates': 50000,
        'commands': commands, 'comparison': 'Only learning rate changes; same initialization seed, data and optimizer-update budget',
        'selection': 'Use actual aggregate official metrics; do not infer a winning learning rate from training loss alone.'}
    path = Path(output) / f'lr_comparison_{digest[:10]}.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if {key: value for key, value in previous.items() if key != 'created'} != plan:
            raise ValueError('Prespecified comparison plan changed; preserve it and choose a new output version')
    else:
        write_json(path, plan | {'created': timestamp()})
    return plan | {'status': 'ready_for_fresh_training', 'plan_path': str(path)}


def run_plan(plan):
    if plan['status'] != 'ready_for_fresh_training':
        raise ValueError('Training remains gated on the personally reviewed baseline manifest')
    if sha256(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Manifest changed after the comparison plan was frozen')
    pending = []
    for recipe in plan['commands']:
        if sha256(recipe['config']) != recipe['config_sha256']:
            raise ValueError('Experiment configuration changed after planning')
        checkpoint, out = Path(recipe['final_checkpoint']), Path(recipe['output'])
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            receipt = json.loads(checkpoint.with_suffix('.json').read_text())
            if state['update'] != 50000 or state['seed'] != 17 or state['manifest_sha256'] != plan['manifest_sha256'] or \
               state['config'] != read_config(recipe['config']) or receipt['sha256'] != sha256(checkpoint):
                raise ValueError(f'Existing completed experiment differs from the declared recipe: {out}')
            print(f'Reusing verified completed run: {out}', flush=True)
            continue
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Interrupted/partial run exists at {out}; resume an explicit checkpoint with src.training.train before continuing')
        pending.append(recipe)
    for recipe in pending:
        subprocess.run(recipe['command'], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--base-config', default='configs/ddpm.yaml', type=Path)
    parser.add_argument('--plan-output', default='results/plans', type=Path)
    parser.add_argument('--run', action='store_true', help='Run three from-scratch 50k-update jobs serially after data gates pass')
    args = parser.parse_args()
    plan = materialize(args.manifest, args.base_config, output=args.plan_output)
    print(json.dumps(plan, indent=2))
    if args.run:
        run_plan(plan)


if __name__ == '__main__':
    main()

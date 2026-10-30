"""Freeze one winner from all prespecified selection seeds, preserving original evidence."""
import argparse
import json
from pathlib import Path
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.sweep import score_results
from src.release.check import check_files


def variant_name(candidate):
    return f'{candidate["name"]}_{candidate["weights"]}_s{candidate["steps"]}_g{candidate["guidance"]:g}'


def validated_result(path, seed, exported, n_per_class=1000, batch_size=500):
    path = Path(path)
    result = json.loads(path.read_text())
    provenance = json.loads(path.with_suffix('.provenance.json').read_text())
    if provenance['result_sha256'] != sha256(path):
        raise ValueError('Original evaluator JSON changed')
    if provenance.get('submission_file_sha256') != check_files(exported)['file_sha256']:
        raise ValueError('Evaluation is stale for this export')
    protocol = result['protocol']
    if protocol['seed'] != seed or protocol['n_per_class'] != n_per_class or \
       protocol['batch_size'] != batch_size or not protocol['grading']:
        raise ValueError('Evaluation protocol does not match declared full selection protocol')
    if Path(result['submission']).resolve() != Path(exported).resolve():
        raise ValueError('Evaluation belongs to a different exported candidate')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/evaluation.yaml')
    p.add_argument('--out', type=Path)
    p.add_argument('--reopen', action='store_true', help='Record that prior confirmation results have been seen')
    args = p.parse_args()
    config = read_config(args.config)
    if not 1 <= len(config['candidates']) <= 3:
        raise ValueError('Selection requires one to three fully evaluated sampler candidates')
    root = Path(config['output'])
    previous_confirmation = sorted((root / 'confirmation').glob('*_seed*.json'))
    previous_confirmation = [p for p in previous_confirmation if '.provenance.' not in p.name]
    if previous_confirmation and not args.reopen:
        raise ValueError('Confirmation evidence already exists; explicitly reopen and disclose reuse')
    ranked = []
    for candidate in config['candidates']:
        name = variant_name(candidate)
        exported = ROOT / 'private_build/evaluation' / name
        check_files(exported)
        paths = [root / 'selection' / f'{name}_seed{seed}.json' for seed in config['selection_seeds']]
        results = [validated_result(path, seed, exported, config['selection_n_per_class'], config['batch_size'])
                   for path, seed in zip(paths, config['selection_seeds'])]
        score = score_results(results)
        ranked.append({'name': name, 'candidate': candidate, 'score': score,
                       'evidence': {str(path): sha256(path) for path in paths}, 'exported': str(exported)})
    valid = [r for r in ranked if r['score'] is not None]
    if not valid:
        raise ValueError('No candidate passes all hard conditions and complete evaluation requirements')
    valid.sort(key=lambda row: tuple(row['score']), reverse=True)
    winner = valid[0]
    exported = Path(winner['exported'])
    result = {'frozen_at': timestamp(), 'selection_config_sha256': sha256(args.config),
        'selection_seeds': config['selection_seeds'], 'confirmation_seeds': config['confirmation_seeds'],
        'winner': winner, 'all_candidates': ranked,
        'export_hashes': {name: sha256(exported / name) for name in ['weights.pt', 'manifest.csv', 'inference.json', 'provenance.json']},
        'previous_confirmation': {str(path): sha256(path) for path in previous_confirmation},
        'confirmation_untouched_at_freeze': not previous_confirmation,
        'rule': 'highest minimum official tier total; then minimum normalized three-metric margin; '
                'then lower worst memorization; then faster worst sampling'}
    out = args.out or root / 'selection/winner.json'
    if out.exists():
        raise FileExistsError('Winner record is immutable; use a new output version when reopening selection')
    write_json(out, result)
    print(winner['name'], winner['score'])


if __name__ == '__main__':
    main()

"""Summarize every experiment folder under results/experiments into results/experiments_index.csv.

    python -m scripts.build_experiments_index

For each folder: number of official evaluation results, and the best one (most marks, then lowest FID; an evaluation that
violates a hard limit counts as 0 marks) with its metrics and where it came from. Reads only the recorded result JSONs.
"""
import csv
import json
from pathlib import Path

ROOT = Path('results/experiments')
FIELDS = ['experiment', 'evaluations', 'best_marks', 'best_variant', 'seed', 'accuracy', 'fid', 'recall', 'precision',
          'memorization', 'hard_limit_violations']


def rows():
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        results = []
        for path in folder.rglob('seed_*.json'):
            if 'provenance' in path.name:
                continue
            try:
                data = json.loads(path.read_text())
                m = data['metrics']
                marks = data['marks']['total'] if data['marks'].get('valid') else 0
            except (ValueError, KeyError, TypeError):
                continue
            results.append((marks, -m['fid'], path, m, data['seed'] if 'seed' in data else data.get('protocol', {}).get('seed')))
        if not results:
            continue
        marks, _, path, m, seed = max(results, key=lambda r: (r[0], r[1]))
        variant = path.parent.relative_to(folder).as_posix() or '.'
        yield {'experiment': folder.name, 'evaluations': len(results), 'best_marks': marks, 'best_variant': variant, 'seed': seed,
               'accuracy': round(m['accuracy'], 4), 'fid': round(m['fid'], 2), 'recall': round(m['recall'], 4),
               'precision': round(m['precision'], 4), 'memorization': m.get('memorization_rate'),
               'hard_limit_violations': sum(1 for r in results if r[0] == 0)}


def main():
    out = Path('results/experiments_index.csv')
    data = list(rows())
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(data)
    print(len(data), 'experiment folders ->', out)


if __name__ == '__main__':
    main()

"""Break the student's completed official audit down by class, source and error reason.

    python -m src.evaluation.audit_breakdown --out results/audit/breakdown.json

Reads only audit/audit.csv (the student's personal verdicts), manifests/v010.csv (the student's
personally reviewed rows) and the recorded candidate tables; it never changes a verdict. The
"recrop" flag marks AI rows whose square was re-made from a face-only detector box.
"""
import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

from src.common import write_json
from src.curation.calibration import wilson
from src.curation.rank_classifier import candidate_rows

R = Path('results/fullpool_20261119')


def key(row):
    return (row['pool_id'].strip(), int(row['x0']), int(row['y0']), int(row['size']),
            float(row['angle'] or 0), int(row['label']))


def stat(rows):
    n = len(rows)
    good = sum(r['verdict'].strip().lower() == 'good' for r in rows)
    lo, hi = wilson(good, n) if n else (None, None)
    return {'n': n, 'good': good, 'precision': good / n if n else None, 'wilson_95': [lo, hi]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('audit/audit.csv'))
    p.add_argument('--personal', type=Path, default=Path('manifests/v010.csv'))
    p.add_argument('--out', type=Path, default=Path('results/audit/breakdown.json'))
    args = p.parse_args()
    rows = list(csv.DictReader(args.audit.open(newline='', encoding='utf-8-sig')))
    personal = {key(r) for r in csv.DictReader(args.personal.open(newline='', encoding='utf-8-sig'))}
    candidates = candidate_rows()
    for path in sorted(glob.glob(str(R / 'recrop_batch*.csv'))):
        for r in csv.DictReader(open(path, newline='', encoding='utf-8')):
            candidates[r['candidate_id']] = r
    recrop = {(c['pool_id'], int(c['x0']), int(c['y0']), int(c['size'])) for c in candidates.values()
              if c.get('parent_candidate_id')}
    for r in rows:
        r['source'] = 'personal' if key(r) in personal else 'ai_found'
        r['recrop'] = (r['pool_id'].strip(), int(r['x0']), int(r['y0']), int(r['size'])) in recrop
    bad = [r for r in rows if r['verdict'].strip().lower() == 'bad']
    ai = [r for r in rows if r['source'] == 'ai_found']
    result = {
        'audit_csv': str(args.audit), 'overall': stat(rows),
        'by_class': {n: stat([r for r in rows if int(r['label']) == v]) for v, n in ((0, 'cat'), (1, 'dog'))},
        'by_source': {s: stat([r for r in rows if r['source'] == s]) for s in ('personal', 'ai_found')},
        'by_source_and_class': {f'{s}_{n}': stat([r for r in rows if r['source'] == s and int(r['label']) == v])
                                for s in ('personal', 'ai_found') for v, n in ((0, 'cat'), (1, 'dog'))},
        'bad_reasons': dict(Counter(r['reason'].strip().lower() for r in bad).most_common()),
        'bad_by_source': dict(Counter(r['source'] for r in bad)),
        'ai_recrop_rows': stat([r for r in ai if r['recrop']]),
        'ai_original_square_rows': stat([r for r in ai if not r['recrop']]),
        'ears_cut_in_ai_recrop_rows': sum('ears' in r['reason'].lower() for r in ai if r['recrop']),
        'ears_cut_in_ai_original_rows': sum('ears' in r['reason'].lower() for r in ai if not r['recrop']),
        'scope': 'Descriptive statistics of the student\'s own verdicts; not a re-judgment.'}
    write_json(args.out, result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()

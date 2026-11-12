"""Recompute geometry statistics from the exact frozen student review evidence."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import quantiles

from src.common import sha256, timestamp, write_json
from src.curation.freeze import latest_decisions


def geometry_key(row):
    return (row['pool_id'], *(int(row[key]) for key in ('x0', 'y0', 'size', 'label')), float(row['angle']))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def distribution(values):
    q1, median, q3 = quantiles(values, n=4, method='inclusive')
    return {'minimum': min(values), 'q1': q1, 'median': median, 'q3': q3, 'maximum': max(values)}


def report(manifest, decisions, groups, output):
    provenance = manifest.with_suffix('.provenance.json')
    holdout = manifest.with_suffix('.validation.csv')
    frozen = json.loads(provenance.read_text(encoding='utf-8'))
    files = {str(path): sha256(path) for path in (manifest, holdout, provenance, decisions, groups)}
    expected = [row['sha256'] for row in frozen['manifests']]
    expected.extend((frozen['decision_log_sha256'], frozen['duplicate_review_sha256']))
    if [files[str(path)] for path in (manifest, holdout, decisions, groups)] != expected:
        raise ValueError('The supplied review ledgers and manifests must match the frozen bundle')
    latest = latest_decisions(decisions)
    kept = {geometry_key(latest[row['candidate_id']]): latest[row['candidate_id']]
            for row in read_csv(groups) if row['keep'] == 'yes'}
    rows = []
    for split, path in [('training', manifest), ('holdout', holdout)]:
        for index, row in enumerate(read_csv(path), start=1):
            review = kept[geometry_key(row)]
            if review['verdict'] != 'good' or review['origin'] != 'student_browser_review':
                raise ValueError(f'Kept crop lacks its recorded personal acceptance: {row["pool_id"]}')
            rows.append({'split': split, 'row': index, 'pool_id': row['pool_id'], 'label': int(row['label']),
                         'candidate_id': review['candidate_id'], 'preview_sha256': review['preview_sha256'],
                         'head_width': review['head_width'], 'side': review['size'],
                         'head_width_basis': review['head_width_basis'], 'recorded_occupancy': review['occupancy'],
                         'recalculated_occupancy': review['head_width'] / review['size']})
    if len(rows) != len(kept) or len(rows) != frozen['accepted_count']:
        raise ValueError('Frozen rows do not reconcile to the kept reviews')

    training = [row for row in rows if row['split'] == 'training']
    stats = {}
    for label, name in enumerate(('cat', 'dog')):
        cohort = [row for row in training if row['label'] == label]
        stats[name] = {'n': len(cohort), 'head_width_px': distribution([r['head_width'] for r in cohort]),
                       'head_width_basis': dict(Counter(r['head_width_basis'] for r in cohort)),
                       'occupancy': distribution([r['recalculated_occupancy'] for r in cohort])}
    result = {'created': timestamp(), 'source_hashes': files, 'kept': len(rows), 'training': stats,
              'legacy_occupancy_difference_above_0_005': sum(
                  abs(r['recorded_occupancy'] - r['recalculated_occupancy']) > .00500001 for r in rows),
              'strict_range_rechecks': [r for r in rows if not .6 <= r['recalculated_occupancy'] <= .8],
              'scope': 'Recomputed from recorded widths. Human remeasurement and the official audit are separate.'}
    table = output.with_suffix('.csv')
    if output.exists() or table.exists():
        raise FileExistsError('Geometry receipt exists; retain the frozen comparison')
    output.parent.mkdir(parents=True, exist_ok=True)
    with table.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output, result | {'table_sha256': sha256(table)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('manifests/v001.csv'))
    parser.add_argument('--decisions', type=Path, default=Path('annotations/decisions.jsonl'))
    parser.add_argument('--groups', type=Path, default=Path('annotations/duplicate_groups.csv'))
    parser.add_argument('--out', type=Path, default=Path('results/curation/v001_review_geometry.json'))
    args = parser.parse_args()
    result = report(args.manifest, args.decisions, args.groups, args.out)
    print(json.dumps({'kept': result['kept'], 'strict_range_rechecks': result['strict_range_rechecks']}))


if __name__ == '__main__':
    main()

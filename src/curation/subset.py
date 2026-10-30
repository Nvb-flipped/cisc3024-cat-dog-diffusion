"""Create the prespecified balanced data-quantity ablation from a frozen reviewed parent."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from src.common import sha256
from src.curation.freeze import Pool, COLUMNS, check_manifest
from src.curation.manifest_bundle import publish_bundle


def balanced_subset(parent, out, pool, *, per_class=None, seed=3024):
    parent = Path(parent)
    provenance_path = parent.with_suffix('.provenance.json')
    provenance = json.loads(provenance_path.read_text())
    if provenance['manifests'][0]['sha256'] != sha256(parent):
        raise ValueError('Reviewed parent manifest changed after freeze')
    if provenance.get('diagnostic_only'):
        raise ValueError('Diagnostic rows cannot form a training ablation')
    if provenance.get('origin') not in ['student_reviewed_manifest', 'subset_of_student_reviewed_manifest']:
        raise ValueError('A frozen personally reviewed parent lineage is required')
    checked = check_manifest(parent, pool)
    if checked['errors']:
        raise ValueError(f'Invalid parent manifest: {checked["errors"]}')
    rows = checked['rows']
    candidate_ids = provenance['candidate_ids']
    if len(candidate_ids) != len(rows) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Parent review lineage is incomplete')
    counts = Counter(row['label'] for row in rows)
    target = per_class if per_class is not None else (1000 if min(counts[0], counts[1]) >= 1000 else 500)
    if not isinstance(target, int) or isinstance(target, bool) or target < 500 or min(counts[0], counts[1]) < target:
        raise ValueError('Balanced ablation needs at least the requested 500+ reviewed rows per class')
    if len(rows) <= 2*target:
        raise ValueError('No meaningful quantity contrast remains; omit this ablation instead of duplicating rows')
    def rank(row):
        key = ','.join(str(row[column]) for column in COLUMNS)
        return hashlib.sha256(f'pool-only-quantity:{seed}:{key}'.encode()).hexdigest()
    selected = []
    for label in [0, 1]:
        selected.extend(sorted([row for row in rows if row['label'] == label], key=rank)[:target])
    selected.sort(key=lambda row: (row['pool_id'], row['label'], row['x0'], row['y0']))
    by_geometry = {tuple(row[column] for column in COLUMNS): identifier
                   for row, identifier in zip(rows, candidate_ids)}
    held = parent.with_suffix('.validation.csv')
    if provenance['manifests'][1]['sha256'] != sha256(held):
        raise ValueError('Parent holdout changed after freeze')
    held_check = check_manifest(held, pool)
    if held_check['errors']:
        raise ValueError('Parent holdout is invalid')
    if {row['pool_id'] for row in rows} & {row['pool_id'] for row in held_check['rows']}:
        raise ValueError('Parent training and holdout share sources')
    metadata = {'origin': 'subset_of_student_reviewed_manifest',
        'pool_content_sha256': provenance['pool_content_sha256'],
        'parent_manifest_sha256': sha256(parent), 'parent_provenance_sha256': sha256(provenance_path),
        'parent_holdout_sha256': sha256(held), 'train_by_class': {0: target, 1: target},
        'parent_train_by_class': dict(counts), 'candidate_ids': [by_geometry[tuple(row[column] for column in COLUMNS)] for row in selected],
        'seed': seed, 'selection': 'Balanced hash-ranked subset of exact reviewed parent training rows; no dev features or new crops',
        'fresh_initialization_required': True, 'training_budget': 'Match 50,000 optimizer updates to the full-data comparison; report 6,400,000 images seen at effective batch 128',
        'caveat': 'Report the full-data class ratio and size contrast; a balanced subset can change class proportions if the parent is unbalanced.'}
    return publish_bundle(out, selected, held_check['rows'], metadata, columns=COLUMNS,
                          check=lambda path: check_manifest(path, pool), validation_source=held)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pool', default='data/student')
    parser.add_argument('--per-class', type=int)
    parser.add_argument('--seed', default=3024, type=int)
    args = parser.parse_args()
    print(balanced_subset(args.parent, args.out, Pool(args.pool), per_class=args.per_class, seed=args.seed))


if __name__ == '__main__':
    main()

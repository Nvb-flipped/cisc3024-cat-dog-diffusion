"""Freeze reviewed pool crops and a duplicate-family-grouped holdout manifest."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from src.common import ROOT, read_config, sha256
from src.curation.review import CRITERIA, rendered_crop
from src.curation.provenance import crop_review_identity
from src.curation.manifest_bundle import publish_bundle
from src.curation.pool_fingerprint import fingerprint

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool
from catdog_eval.manifest import COLUMNS, check_manifest


def latest_decisions(path):
    decisions = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            decisions[item['candidate_id']] = item
    return decisions


def split_by_group(rows, group_map, fraction, seed):
    """Hash groups once; class-specific sorted order yields a fixed approximately 5% holdout."""
    train, validation = [], []
    for label in (0, 1):
        subset = [r for r in rows if r['label'] == label]
        groups = {}
        for row in subset:
            group = group_map[row['candidate_id']]
            groups.setdefault(group, []).append(row)
        order = sorted(groups, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
        count = 0
        target = max(1, round(len(subset) * fraction)) if fraction and subset else 0
        for group in order:
            if count < target:
                validation.extend(groups[group])
                count += len(groups[group])
            else:
                train.extend(groups[group])
    return train, validation


def freeze(config, out, decisions_path=None, groups_path='annotations/duplicate_groups.csv',
           *, preserve_split_from=None):
    pool = Pool(config['pool'])
    source = Path(decisions_path or config['decisions'])
    decisions = latest_decisions(source)
    groups_path = Path(groups_path)
    if not groups_path.exists():
        raise ValueError('Student duplicate review required: missing annotations/duplicate_groups.csv')
    with groups_path.open(newline='', encoding='utf-8') as f:
        group_rows = list(csv.DictReader(f))
    groups = {r['candidate_id']: r for r in group_rows}
    if len(groups) != len(group_rows):
        raise ValueError('Duplicate candidate IDs in duplicate review')
    accepted = []
    group_labels = {}
    for item in decisions.values():
        if item['verdict'] != 'good' or item['candidate_id'].startswith('reject:'):
            continue
        if item.get('origin') != 'student_browser_review' or not item.get('reviewer') or not item.get('timestamp'):
            raise ValueError('Missing authentic review provenance')
        if not all(item.get('criteria', {}).get(k) for k in CRITERIA):
            raise ValueError('A good crop lacks specification checks')
        if not math.isfinite(item['head_width']) or item['head_width'] < 32 or not 0.60 <= item['occupancy'] <= 0.80:
            raise ValueError('Invalid confirmed source head size/occupancy')
        if item['angle'] and not item.get('angle_reason'):
            raise ValueError('Nonzero rotation lacks explanation')
        geometry = {k: item[k] for k in ['x0', 'y0', 'size', 'angle']}
        if hashlib.sha256(rendered_crop(pool, item['pool_id'], geometry)).hexdigest() != item['preview_sha256']:
            raise ValueError('Source/geometry changed after review')
        group = groups.get(item['candidate_id'])
        if not group or not group.get('reviewer') or not group.get('timestamp') or not group.get('group_id'):
            raise ValueError(f'Duplicate review pending: {item["candidate_id"]}')
        if group.get('origin') not in ['student_duplicate_review', 'student_manual_duplicate_review'] or \
           group.get('crop_review_sha256') != crop_review_identity(item):
            raise ValueError(f'Duplicate review is stale for this crop/label: {item["candidate_id"]}')
        if group.get('keep') not in ('yes', 'no'):
            raise ValueError('Duplicate review keep must be yes/no')
        previous_label = group_labels.setdefault(group['group_id'], item['label'])
        if previous_label != item['label']:
            raise ValueError('Duplicate family has contradictory labels')
        if group['keep'] == 'yes':
            if not 0.60 <= item['head_width'] / item['size'] <= 0.80:
                raise ValueError(f"Recheck source width or crop occupancy for {item['candidate_id']}")
            accepted.append(item)
    if not accepted:
        raise ValueError('No student-approved crops to freeze')
    if len({row['preview_sha256'] for row in accepted}) != len(accepted):
        raise ValueError('Exactly identical crop pixels are repeated; keep one representative')
    kept_groups = [groups[r['candidate_id']]['group_id'] for r in accepted]
    if len(set(kept_groups)) != len(kept_groups):
        raise ValueError('More than one kept crop per duplicate head family')
    # Multiple non-overlapping heads in one source still belong to the same split family.
    by_source = {}
    for item in accepted:
        source_group = by_source.setdefault(item['pool_id'], groups[item['candidate_id']]['group_id'])
        if source_group != groups[item['candidate_id']]['group_id']:
            raise ValueError('Multiple kept crops from one source require one source family; keep one for this baseline')
    pool_hash, _ = fingerprint(config['pool'])
    split_metadata = {}
    validation_source = None
    if preserve_split_from is None:
        train, validation = split_by_group(accepted,
            {r['candidate_id']: groups[r['candidate_id']]['group_id'] for r in accepted},
            config['validation_fraction'], config['seed'])
    else:
        parent = Path(preserve_split_from)
        parent_record = parent.with_suffix('.provenance.json')
        validation_source = parent.with_suffix('.validation.csv')
        lineage = json.loads(parent_record.read_text(encoding='utf-8'))
        digests = [sha256(path) for path in (parent, validation_source)]
        if lineage.get('origin') != 'student_reviewed_manifest' or lineage.get('diagnostic_only') or \
                lineage.get('pool_content_sha256') != pool_hash or \
                digests != [row['sha256'] for row in lineage['manifests']]:
            raise ValueError('The parent must be a verified review freeze on this exact pool')
        by_source = {row['pool_id']: row for row in accepted}
        parent_splits = []
        for path in (parent, validation_source):
            checked = check_manifest(path, pool)
            if checked['errors']:
                raise ValueError(f'Invalid parent split: {path}')
            parent_splits.append(checked['rows'])
        for row in parent_splits[0] + parent_splits[1]:
            current = by_source.get(row['pool_id'])
            if current is None or any(row[key] != current[key] for key in COLUMNS):
                raise ValueError(f'Parent row changed or is no longer kept: {row["pool_id"]}')
        holdout_sources = {row['pool_id'] for row in parent_splits[1]}
        if holdout_sources & {row['pool_id'] for row in parent_splits[0]}:
            raise ValueError('The parent splits share sources')
        train = [row for row in accepted if row['pool_id'] not in holdout_sources]
        validation = [by_source[row['pool_id']] for row in parent_splits[1]]
        split_metadata = {'parent_manifest_sha256': digests[0], 'parent_holdout_sha256': digests[1],
                          'parent_provenance_sha256': sha256(parent_record),
                          'split_policy': 'Retain every parent training row and the byte-identical parent holdout'}
    out = Path(out)
    train.sort(key=lambda r: (r['pool_id'], r['label'], r['x0'], r['y0']))
    validation.sort(key=lambda r: (r['pool_id'], r['label']))
    return publish_bundle(out, train, validation, {
            'origin': 'student_reviewed_manifest',
            'pool_content_sha256': pool_hash,
            'decision_log_sha256': sha256(source), 'duplicate_review_sha256': sha256(groups_path),
            'accepted_count': len(accepted), 'train_by_class': dict(Counter(r['label'] for r in train)),
            'validation_by_class': dict(Counter(r['label'] for r in validation)),
            'candidate_ids': [r['candidate_id'] for r in train], 'seed': config['seed'],
            'holdout_fraction': config['validation_fraction'],
            'policy': 'student reviewed; one head per confirmed duplicate family; pool-only grouped holdout',
            **split_metadata},
            columns=COLUMNS, check=lambda path: check_manifest(path, pool),
            validation_source=validation_source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/curation.yaml')
    p.add_argument('--out', default='manifests/final.csv')
    p.add_argument('--decisions')
    p.add_argument('--duplicate-groups', default='annotations/duplicate_groups.csv')
    p.add_argument('--preserve-split-from', type=Path)
    args = p.parse_args()
    print(freeze(read_config(args.config), args.out, args.decisions, args.duplicate_groups,
                 preserve_split_from=args.preserve_split_from))


if __name__ == '__main__':
    main()

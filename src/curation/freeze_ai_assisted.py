"""Freeze an AI-assisted full-pool manifest (Stage 7) on top of the personal v010 manifest.

Inputs: the personal parent manifest (kept exactly, with its byte-identical holdout), merged
candidate tables, and append-only VLM ledgers (Qwen3.5-4B primary, optional 9B escalation).

Resolution (plan section 5): 4B PASS without a 9B record, or 4B and 9B PASS with the same
species, is accepted. 4B UNCERTAIN/ERROR is accepted only on a 9B PASS. Any PASS/REJECT
conflict, or a species disagreement between models, stays unresolved and is excluded. VLM
decisions are model outputs, never personal verdicts; personal verdicts always win.

Deterministic checks: official bounds, side >= 40, detector head width >= 32 source pixels,
head extent / side within 0.60-0.80. Duplicate families (src.curation.deduplicate on exact
official crops, plus repost sources with matching source dHash and overlapping normalised boxes)
keep one row: the personal row if present, else the widest source head. Families or sources
touching the holdout are excluded from training. Every candidate receives a final status.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.common import ROOT, sha256, timestamp
from src.curation.deduplicate import UnionFind, families
from src.curation.geometry import inside
from src.curation.manifest_bundle import publish_bundle
from src.curation.pool_fingerprint import fingerprint

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, rebuild  # noqa: E402
from catdog_eval.manifest import COLUMNS, check_manifest  # noqa: E402

SPECIES = {'cat': 0, 'dog': 1}
POLICY_BASIS = ('Paco reported (19 Nov 2026) a TA clarification that AI-assisted cleaning qualifies as student '
                'work without manually reviewing every pool image; the official 200-crop audit stays personal.')


def read_ledgers(paths):
    latest = {}
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                latest[record['candidate_id']] = record  # later lines (re-screens) supersede
    return latest


def resolve(primary, escalation):
    first = primary['resolved'] if primary else 'MISSING'
    second = escalation['resolved'] if escalation else None
    species = [r['verdict']['species'] for r in (primary, escalation) if r and r.get('verdict')]
    if first == 'PASS' and second is None:
        return 'ACCEPT', species[0]
    if second == 'PASS' and first in ('PASS', 'UNCERTAIN', 'ERROR'):
        return ('ACCEPT', species[-1]) if len(set(species)) == 1 or first != 'PASS' else ('UNRESOLVED_SPECIES', None)
    if first == 'PASS' and second in ('REJECT', 'UNCERTAIN', 'ERROR'):
        return 'UNRESOLVED_CONFLICT', None
    if first == 'REJECT' and second == 'PASS':
        return 'UNRESOLVED_CONFLICT', None
    if first in ('UNCERTAIN', 'ERROR') and second is None:
        return 'PENDING_ESCALATION', None
    if first == 'MISSING':
        return 'NOT_SCREENED', None
    return 'REJECT', None


def geometry_ok(row, width, height):
    x0, y0, size, angle = int(row['x0']), int(row['y0']), int(row['size']), float(row['angle'] or 0)
    box = json.loads(row['box'])
    extent = max(box[2] - box[0], box[3] - box[1])
    return (inside(width, height, x0, y0, size, angle) and size >= 40 and box[2] - box[0] >= 32
            and 0.60 <= extent / size <= 0.80 + 1e-9)


def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def freeze(out, parent, candidate_tables, primary_ledgers, escalation_ledgers=(), pool_path='data/student',
           integrity='results/curation/integrity.csv', status_out=None):
    pool = Pool(pool_path)
    holdout_path = Path(parent).with_suffix('.validation.csv')
    parent_rows = check_manifest(parent, pool)['rows']
    holdout_rows = check_manifest(holdout_path, pool)['rows']
    holdout_sources = {r['pool_id'] for r in holdout_rows}
    dims = {r['pool_id']: (int(r['width']), int(r['height']), int(r['dhash'], 16) if r['dhash'] else None)
            for r in csv.DictReader(open(integrity, newline='', encoding='utf-8'))}
    primary, escalation = read_ledgers(primary_ledgers), read_ledgers(escalation_ledgers)
    status, accepted = {}, []
    for table in candidate_tables:
        for row in csv.DictReader(open(table, newline='', encoding='utf-8')):
            cid = row['candidate_id']
            if row.get('personal_verdict'):
                status[cid] = 'PERSONAL_DECISION'
                continue
            decision, species = resolve(primary.get(cid), escalation.get(cid))
            width, height, _ = dims[row['pool_id']]
            if decision == 'ACCEPT' and not geometry_ok(row, width, height):
                decision = 'REJECT_GEOMETRY'
            if decision == 'ACCEPT' and row['pool_id'] in holdout_sources:
                decision = 'EXCLUDED_HOLDOUT_SOURCE'
            status[cid] = decision
            if decision == 'ACCEPT':
                accepted.append(row | {'label': SPECIES[species]})
    # Several accepted heads in one source must be clearly different heads.
    by_source = defaultdict(list)
    for row in sorted(accepted, key=lambda r: (-float(r['box_width']), r['candidate_id'])):
        boxes = [json.loads(other['box']) for other in by_source[row['pool_id']]]
        if any(box_iou(json.loads(row['box']), box) > 0.1 for box in boxes):
            status[row['candidate_id']] = 'DUPLICATE_SAME_SOURCE'
        else:
            by_source[row['pool_id']].append(row)
    accepted = [row for rows in by_source.values() for row in rows]
    # Exact official 32x32 pixels for duplicate families.
    combined = ([dict(r, _kind='parent') for r in parent_rows] + [dict(r, _kind='holdout') for r in holdout_rows]
                + [dict(r, _kind='ai') for r in accepted])
    pixels = rebuild(combined, pool)[0].mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    roots = families(pixels)
    uf = UnionFind(len(combined))
    for i, root in enumerate(roots):
        uf.union(i, root)
    # Reposts of one photo across pool ids: near-identical source dHash and overlapping normalised boxes.
    groups = defaultdict(list)
    for i, row in enumerate(combined):
        width, height, source_hash = dims[row['pool_id']]
        if source_hash is not None:
            groups[source_hash >> 48].append(i)  # bucket by the top 16 bits, then compare exactly
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                ri, rj = combined[i], combined[j]
                if ri['pool_id'] == rj['pool_id']:
                    continue
                wi, hi, si = dims[ri['pool_id']]
                wj, hj, sj = dims[rj['pool_id']]
                if bin(si ^ sj).count('1') > 3:
                    continue
                ni = [int(ri['x0']) / wi, int(ri['y0']) / hi, (int(ri['x0']) + int(ri['size'])) / wi, (int(ri['y0']) + int(ri['size'])) / hi]
                nj = [int(rj['x0']) / wj, int(rj['y0']) / hj, (int(rj['x0']) + int(rj['size'])) / wj, (int(rj['y0']) + int(rj['size'])) / hj]
                if box_iou(ni, nj) > 0.5:
                    uf.union(i, j)
    family = defaultdict(list)
    for i in range(len(combined)):
        family[uf.find(i)].append(i)
    keep = []
    for members in family.values():
        kinds = {combined[i]['_kind'] for i in members}
        ai = [i for i in members if combined[i]['_kind'] == 'ai']
        if 'holdout' in kinds:
            for i in ai:
                status[combined[i]['candidate_id']] = 'EXCLUDED_HOLDOUT_FAMILY'
            continue
        if 'parent' in kinds:
            for i in ai:
                status[combined[i]['candidate_id']] = 'DUPLICATE_OF_PERSONAL'
            continue
        if len({int(combined[i]['label']) for i in ai}) > 1:
            for i in ai:
                status[combined[i]['candidate_id']] = 'UNRESOLVED_FAMILY_LABEL_CONFLICT'
            continue
        if ai:
            best = min(ai, key=lambda i: (-float(combined[i]['box_width']), combined[i]['candidate_id']))
            keep.append(combined[best])
            for i in ai:
                if i != best:
                    status[combined[i]['candidate_id']] = 'DUPLICATE_FAMILY'
            status[combined[best]['candidate_id']] = 'TRAIN'
    train = [{k: r[k] for k in COLUMNS} for r in parent_rows] + \
            [{'pool_id': r['pool_id'], 'x0': int(r['x0']), 'y0': int(r['y0']), 'size': int(r['size']),
              'angle': float(r['angle'] or 0), 'label': int(r['label'])} for r in keep]
    train.sort(key=lambda r: (r['pool_id'], int(r['label']), int(r['x0']), int(r['y0'])))
    pool_hash, _ = fingerprint(pool_path)
    status_out = Path(status_out or Path('results/fullpool_20261119') / f'{Path(out).stem}.candidate_status.csv')
    if Path(out).exists() or status_out.exists():
        raise FileExistsError('Manifest and status files are immutable; choose a new version')
    with status_out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'final_status'])
        writer.writerows(sorted(status.items()))
    parent_provenance = Path(parent).with_suffix('.provenance.json')
    metadata = {
        'origin': 'ai_assisted_reviewed_manifest', 'created': timestamp(), 'pool_content_sha256': pool_hash,
        'vlm_decisions_are_personal_verdicts': False, 'curation_policy_basis': POLICY_BASIS,
        'parent_manifest': str(parent), 'parent_manifest_sha256': sha256(parent),
        'parent_holdout_sha256': sha256(holdout_path), 'parent_provenance_sha256': sha256(parent_provenance),
        'parent_rows': len(parent_rows), 'ai_rows': len(keep),
        'train_by_class': dict(Counter(str(r['label']) for r in train)),
        'candidate_tables': {str(p): sha256(p) for p in candidate_tables},
        'primary_ledgers': {str(p): sha256(p) for p in primary_ledgers},
        'escalation_ledgers': {str(p): sha256(p) for p in escalation_ledgers},
        'candidate_status': str(status_out), 'candidate_status_sha256': sha256(status_out),
        'status_counts': dict(Counter(status.values())),
        'split_policy': 'Parent training rows kept exactly; parent holdout byte-identical; holdout sources/families excluded'}
    return publish_bundle(out, train, holdout_rows, metadata, columns=COLUMNS,
                          check=lambda path: check_manifest(path, pool), validation_source=holdout_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('manifests/v010.csv'))
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--primary', type=Path, nargs='+', required=True)
    parser.add_argument('--escalation', type=Path, nargs='*', default=())
    args = parser.parse_args()
    print(freeze(args.out, args.parent, args.candidates, args.primary, args.escalation))


if __name__ == '__main__':
    main()

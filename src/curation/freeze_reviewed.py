"""Freeze a manifest from agent-reviewed full-pool crops (acceptance authority = visual review).

Every AI-found row must have a latest agent visual verdict of "good" under the exact handout
specification (decision log entry 50). Personal parent rows (v010) are kept exactly. Species
labels come from Qwen3.5-9B v5, else v8 (cat/dog answers), else a local CLIP cat-vs-dog
zero-shot on the exact crop (curation only). Deterministic checks: official bounds, side >= 40,
detector head width >= 32 source pixels, box extent/side within 0.60-0.80 for ordinary rows and
0.45-0.65 for re-crops (whose full head with ears was visually verified at ~60-80%). Holdout
sources/families are excluded; same-photo square overlaps, pixel near-duplicates and reposts are
collapsed to one row (personal first, then larger source head).
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.common import ROOT, sha256, timestamp
from src.curation.deduplicate import UnionFind, families
from src.curation.freeze_ai_assisted import box_iou
from src.curation.geometry import inside
from src.curation.manifest_bundle import publish_bundle
from src.curation.pool_fingerprint import fingerprint
from src.curation.rank_classifier import candidate_rows

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, rebuild  # noqa: E402
from catdog_eval.manifest import COLUMNS, check_manifest  # noqa: E402

R = Path('results/fullpool_20261119')


def latest_reviews(paths):
    verdicts = {}
    for path in paths:  # later files override earlier ones (re-reviews come last)
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            verdicts[r['candidate_id']] = r
    return verdicts


def species_labels(ids):
    labels = {}
    for ledger in ('vlm/queue_9b_v5.jsonl', 'vlm/rescue_9b_v5.jsonl', 'vlm/recovery_9b_v8.jsonl'):
        for line in open(R / ledger, encoding='utf-8'):
            r = json.loads(line)
            v = r.get('verdict') or {}
            if r['candidate_id'] in ids and r['candidate_id'] not in labels and v.get('species') in ('cat', 'dog'):
                labels[r['candidate_id']] = (v['species'], ledger)
    return labels


def clip_species(rows):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    from catdog_eval.data import crop
    folder = Path('models/clip-photo-screen')
    processor = CLIPProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(folder, local_files_only=True).eval()
    pool = Pool('data/student')
    with torch.inference_mode():
        text = F.normalize(model.get_text_features(**processor(text=['a photo of a cat', 'a photo of a dog'], padding=True, return_tensors='pt')), dim=-1)
        out = {}
        for r in rows:
            image = crop(pool.image(r['pool_id']), int(r['x0']), int(r['y0']), int(r['size']), float(r['angle'] or 0), res=224)
            f = F.normalize(model.get_image_features(**processor(images=[image], return_tensors='pt')), dim=-1)
            out[r['candidate_id']] = ('dog' if float((f @ text.T)[0, 1]) > float((f @ text.T)[0, 0]) else 'cat', 'clip')
    return out


def geometry_ok(row, width, height):
    x0, y0, size, angle = int(row['x0']), int(row['y0']), int(row['size']), float(row['angle'] or 0)
    box = json.loads(row['box'])
    extent = max(box[2] - box[0], box[3] - box[1])
    low, high = (0.45, 0.65) if row.get('parent_candidate_id') else (0.60, 0.80)
    return (inside(width, height, x0, y0, size, angle) and size >= 40 and box[2] - box[0] >= 32
            and low - 1e-9 <= extent / size <= high + 1e-9)


def freeze(out, parent, reviews, recrop_tables, pool_path='data/student', integrity='results/curation/integrity.csv'):
    pool = Pool(pool_path)
    holdout_path = Path(parent).with_suffix('.validation.csv')
    parent_rows = check_manifest(parent, pool)['rows']
    holdout_rows = check_manifest(holdout_path, pool)['rows']
    holdout_sources = {r['pool_id'] for r in holdout_rows}
    dims = {r['pool_id']: (int(r['width']), int(r['height']), int(r['dhash'], 16) if r['dhash'] else None)
            for r in csv.DictReader(open(integrity, newline='', encoding='utf-8'))}
    rows = dict(candidate_rows())
    for table in recrop_tables:
        for r in csv.DictReader(open(table, newline='', encoding='utf-8')):
            rows[r['candidate_id']] = r
    verdicts = latest_reviews(reviews)
    good = [rows[c] for c, v in verdicts.items() if v['verdict'] == 'good' and c in rows]
    status = {c: ('REVIEW_REJECT' if v['verdict'] == 'bad' else 'REVIEW_GOOD') for c, v in verdicts.items()}
    base_ids = {r.get('parent_candidate_id') or r['candidate_id'] for r in good}
    labels = species_labels(base_ids)
    missing = [r for r in good if (r.get('parent_candidate_id') or r['candidate_id']) not in labels]
    labels.update(clip_species([r for r in missing]))
    accepted = []
    for r in good:
        width, height, _ = dims[r['pool_id']]
        if not geometry_ok(r, width, height):
            status[r['candidate_id']] = 'REJECT_GEOMETRY'
        elif r['pool_id'] in holdout_sources:
            status[r['candidate_id']] = 'EXCLUDED_HOLDOUT_SOURCE'
        else:
            label = labels.get(r.get('parent_candidate_id') or r['candidate_id']) or labels.get(r['candidate_id'])
            accepted.append(r | {'label': 0 if label[0] == 'cat' else 1, 'label_source': label[1],
                                 'x0': int(r['x0']), 'y0': int(r['y0']), 'size': int(r['size']),
                                 'angle': float(r['angle'] or 0)})

    def square(r):
        return [int(r['x0']), int(r['y0']), int(r['x0']) + int(r['size']), int(r['y0']) + int(r['size'])]
    kept_squares = defaultdict(list)
    for r in parent_rows + holdout_rows:
        kept_squares[r['pool_id']].append(('personal', square(r)))
    kept = []
    for r in sorted(accepted, key=lambda r: (-float(r['box_width']), r['candidate_id'])):
        overlaps = [k for k, sq in kept_squares[r['pool_id']] if box_iou(square(r), sq) > 0.3]
        if overlaps:
            status[r['candidate_id']] = 'DUPLICATE_OF_PERSONAL' if 'personal' in overlaps else 'DUPLICATE_SAME_SOURCE'
        else:
            kept.append(r)
            kept_squares[r['pool_id']].append(('ai', square(r)))
    combined = ([dict(r, _kind='parent') for r in parent_rows] + [dict(r, _kind='holdout') for r in holdout_rows]
                + [dict(r, _kind='ai') for r in kept])
    pixels = rebuild(combined, pool)[0].mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    roots = families(pixels)
    uf = UnionFind(len(combined))
    for i, root in enumerate(roots):
        uf.union(i, root)
    groups = defaultdict(list)
    for i, r in enumerate(combined):
        source_hash = dims[r['pool_id']][2]
        if source_hash is not None:
            groups[source_hash >> 48].append(i)
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
        if 'holdout' in kinds or 'parent' in kinds:
            for i in ai:
                status[combined[i]['candidate_id']] = 'EXCLUDED_HOLDOUT_FAMILY' if 'holdout' in kinds else 'DUPLICATE_OF_PERSONAL'
            continue
        if ai:
            best = min(ai, key=lambda i: (-float(combined[i]['box_width']), combined[i]['candidate_id']))
            keep.append(combined[best])
            for i in ai:
                status[combined[i]['candidate_id']] = 'TRAIN' if i == best else 'DUPLICATE_FAMILY'
    train = [{k: r[k] for k in COLUMNS} for r in parent_rows] + \
            [{'pool_id': r['pool_id'], 'x0': r['x0'], 'y0': r['y0'], 'size': r['size'], 'angle': r['angle'],
              'label': r['label']} for r in keep]
    train.sort(key=lambda r: (r['pool_id'], int(r['label']), int(r['x0']), int(r['y0'])))
    status_out = R / f'{Path(out).stem}.candidate_status.csv'
    if Path(out).exists() or status_out.exists():
        raise FileExistsError('Manifest and status files are immutable; choose a new version')
    with status_out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'final_status'])
        writer.writerows(sorted(status.items()))
    labels_out = R / f'{Path(out).stem}.labels.csv'
    with labels_out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'label', 'label_source'])
        writer.writerows((r['candidate_id'], r['label'], r['label_source']) for r in keep)
    pool_hash, _ = fingerprint(pool_path)
    parent_provenance = Path(parent).with_suffix('.provenance.json')
    metadata = {
        'origin': 'ai_assisted_reviewed_manifest', 'created': timestamp(), 'pool_content_sha256': pool_hash,
        'vlm_decisions_are_personal_verdicts': False, 'agent_reviews_are_personal_verdicts': False,
        'acceptance_authority': 'agent visual review under the exact handout specification (decision log 50)',
        'curation_policy_basis': 'Paco reported a TA clarification that AI-assisted cleaning qualifies; the official 200-crop audit stays personal.',
        'parent_manifest': str(parent), 'parent_manifest_sha256': sha256(parent),
        'parent_holdout_sha256': sha256(holdout_path), 'parent_provenance_sha256': sha256(parent_provenance),
        'parent_rows': len(parent_rows), 'ai_rows': len(keep),
        'train_by_class': dict(Counter(str(r['label']) for r in train)),
        'agent_reviews': {str(p): sha256(p) for p in reviews}, 'recrop_tables': {str(p): sha256(p) for p in recrop_tables},
        'primary_ledgers': {str(R / l): sha256(R / l) for l in ('vlm/queue_9b_v5.jsonl', 'vlm/rescue_9b_v5.jsonl', 'vlm/recovery_9b_v8.jsonl')},
        'label_sources': dict(Counter(r['label_source'] for r in keep)),
        'candidate_status': str(status_out), 'candidate_status_sha256': sha256(status_out),
        'labels': str(labels_out), 'labels_sha256': sha256(labels_out),
        'status_counts': dict(Counter(status.values())),
        'split_policy': 'Parent training rows kept exactly; parent holdout byte-identical; holdout sources/families excluded'}
    return publish_bundle(out, train, holdout_rows, metadata, columns=COLUMNS,
                          check=lambda path: check_manifest(path, pool), validation_source=holdout_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parent', type=Path, default=Path('manifests/v010.csv'))
    p.add_argument('--reviews', type=Path, nargs='+', required=True, help='Agent review ledgers, earliest first')
    p.add_argument('--recrops', type=Path, nargs='*', default=())
    args = p.parse_args()
    print(freeze(args.out, args.parent, args.reviews, args.recrops))


if __name__ == '__main__':
    main()

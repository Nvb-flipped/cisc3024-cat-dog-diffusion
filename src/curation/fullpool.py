"""Full-pool source registry and merged head-candidate ledger (Stages 0-1 of the full-pool plan).

Inputs are existing evidence only: the integrity scan, the completed Grounding DINO full-pool
proposals, the pool-only CLIP screen scores and the personal review ledger. Nothing here is a
semantic verdict. Outputs are deterministic and versioned under results/fullpool_<run>/.

Geometry rescue: a proposal rejected only because its 70%-occupancy square leaves the source may
be re-squared within the specification: occupancy 0.70-0.80 (side = ceil(extent/occ)), the head
box centre at most 8% of the side from the square centre ("roughly centred"), angle 0, the
official rotated-square boundary rule, head width >= 32 source pixels, and no padding.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.common import sha256, timestamp, write_json
from src.curation.freeze import latest_decisions
from src.curation.geometry import inside

RESCUE_OCCUPANCIES = (0.70, 0.72, 0.74, 0.76, 0.78, 0.80)
MAX_CENTRE_OFFSET = 0.08
MERGE_IOU = 0.5


def rescue_geometry(box, width, height):
    left, top, right, bottom = box
    if right - left < 32 or bottom <= top:
        return None
    extent = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    for occupancy in RESCUE_OCCUPANCIES:
        size = max(40, math.ceil(extent / occupancy))
        if size > min(width, height):
            continue
        x0 = min(max(round(cx - size / 2), 0), width - size)
        y0 = min(max(round(cy - size / 2), 0), height - size)
        dx, dy = (x0 + size / 2 - cx) / size, (y0 + size / 2 - cy) / size
        if abs(dx) <= MAX_CENTRE_OFFSET and abs(dy) <= MAX_CENTRE_OFFSET and inside(width, height, x0, y0, size):
            return {'x0': x0, 'y0': y0, 'size': size, 'angle': 0.0, 'occupancy_target': occupancy,
                    'centre_offset_x': round(dx, 4), 'centre_offset_y': round(dy, 4)}
    return None


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build(out, candidates='results/curation/full/candidates.csv', integrity='results/curation/integrity.csv',
          clip_scores='results/curation/photo_screen_full/scores.csv', decisions='annotations/decisions.jsonl',
          detections='results/curation/full/summary.json'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = {r['pool_id']: r for r in csv.DictReader(open(integrity, newline='', encoding='utf-8'))}
    clip = {r['candidate_id']: r for r in csv.DictReader(open(clip_scores, newline='', encoding='utf-8'))}
    personal = latest_decisions(decisions)
    by_source = defaultdict(list)
    rejection = Counter()
    for row in csv.DictReader(open(candidates, newline='', encoding='utf-8')):
        rejection[row['geometry_rejection'] or 'valid'] += 1
        box = json.loads(row['box'])
        width, height = int(sources[row['pool_id']]['width']), int(sources[row['pool_id']]['height'])
        if row['geometry_rejection'] == '':
            geometry = {'x0': int(row['x0']), 'y0': int(row['y0']), 'size': int(row['size']),
                        'angle': float(row['angle'] or 0), 'occupancy_target': 0.70,
                        'centre_offset_x': 0.0, 'centre_offset_y': 0.0}
            origin = 'dino_valid'
        elif row['geometry_rejection'] == 'square_outside_source':
            geometry = rescue_geometry(box, width, height)
            origin = 'dino_geometry_rescue'
            if geometry is None:
                continue
        else:
            continue
        by_source[row['pool_id']].append({'member': row['candidate_id'], 'box': box, 'origin': origin,
                                          'score': float(row['score']), 'label': int(row['label']),
                                          'phrase': row['detector_phrase'], **geometry})
    merged = []
    for pool_id, items in by_source.items():
        # Original valid geometry first, then detector score: deterministic representative choice.
        items.sort(key=lambda r: (r['origin'] != 'dino_valid', -r['score'], r['member']))
        groups = []
        for item in items:
            for group in groups:
                if iou(group[0]['box'], item['box']) >= MERGE_IOU:
                    group.append(item)
                    break
            else:
                groups.append([item])
        for group in groups:
            head = group[0]
            labels = Counter(m['label'] for m in group)
            hint = clip.get(head['member'], {})
            reviewed = [personal[m['member']] for m in group if m['member'] in personal]
            merged.append({'candidate_id': head['member'], 'pool_id': pool_id, 'origin': head['origin'],
                           'x0': head['x0'], 'y0': head['y0'], 'size': head['size'], 'angle': head['angle'],
                           'occupancy_target': head['occupancy_target'],
                           'centre_offset_x': head['centre_offset_x'], 'centre_offset_y': head['centre_offset_y'],
                           'box': json.dumps([round(v, 2) for v in head['box']]),
                           'box_width': round(head['box'][2] - head['box'][0], 2),
                           'detector_score': round(head['score'], 4), 'detector_phrase': head['phrase'],
                           'detector_label_votes': json.dumps({str(k): v for k, v in sorted(labels.items())}),
                           'clip_species_hint': hint.get('crop_species_hint', ''),
                           'clip_photo_margin': hint.get('crop_photo_margin', ''),
                           'members': json.dumps(sorted(m['member'] for m in group)),
                           'personal_verdict': reviewed[0]['verdict'] if reviewed else '',
                           'personal_candidate_id': reviewed[0]['candidate_id'] if reviewed else ''})
    merged.sort(key=lambda r: (r['pool_id'], r['candidate_id']))
    fields = list(merged[0])
    with (out / 'candidates_merged.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    per_source = defaultdict(Counter)
    for row in merged:
        per_source[row['pool_id']][row['origin']] += 1
    reviewed_sources = Counter(item['pool_id'] for item in personal.values() if 'pool_id' in item)
    registry = []
    for pool_id, src in sorted(sources.items()):
        counts = per_source.get(pool_id, Counter())
        status = ('SOURCE_UNREADABLE' if src['status'] != 'ok' else
                  'CANDIDATES_PENDING_SCREEN' if counts else 'NO_VALID_DETECTION_PENDING_RESCUE')
        registry.append({'pool_id': pool_id, 'shard': src['shard'], 'source_sha256': src['source_sha256'],
                         'width': src['width'], 'height': src['height'], 'integrity_status': src['status'],
                         'dino_valid_heads': counts['dino_valid'], 'geometry_rescue_heads': counts['dino_geometry_rescue'],
                         'personal_decisions': reviewed_sources.get(pool_id, 0), 'terminal_status': status})
    with (out / 'source_registry.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(registry[0]))
        writer.writeheader()
        writer.writerows(registry)
    summary = {'created': timestamp(), 'stage': 'source registry and merged DINO head candidates; no verdicts',
               'inputs': {p: sha256(p) for p in (candidates, integrity, clip_scores, decisions, detections)},
               'sources': len(registry), 'source_status': dict(Counter(r['terminal_status'] for r in registry)),
               'proposal_geometry': dict(rejection), 'merged_candidates': len(merged),
               'merged_by_origin': dict(Counter(r['origin'] for r in merged)),
               'with_personal_verdict': dict(Counter(r['personal_verdict'] for r in merged if r['personal_verdict'])),
               'rescue_rule': {'occupancies': RESCUE_OCCUPANCIES, 'max_centre_offset': MAX_CENTRE_OFFSET,
                               'angle': 0, 'min_head_width': 32, 'merge_iou': MERGE_IOU},
               'outputs': {name: sha256(out / name) for name in ('candidates_merged.csv', 'source_registry.csv')}}
    write_json(out / 'stage01_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/fullpool_20261119')
    print(json.dumps(build(parser.parse_args().out), indent=1))


if __name__ == '__main__':
    main()

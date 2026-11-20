"""Re-square candidates whose detector box covered only the face (agent marked "too tight").

The new square makes the detector box span 0.50 of the side (else 0.55, then 0.60), centred on the
box within an 8% offset, angle 0, inside the official bounds, no padding. Because such boxes omit
the ears, the full head with ears then spans roughly 60-80% of the side; every re-crop is still
visually reviewed before use. New candidate ids carry the suffix "-o50"/"-o55"/"-o60".
"""
import argparse
import csv
import json
import math
from pathlib import Path

from src.curation.geometry import inside
from src.curation.rank_classifier import candidate_rows

OCCUPANCIES = (0.50, 0.55, 0.60)


def resquare(box, width, height):
    left, top, right, bottom = box
    extent = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    for occupancy in OCCUPANCIES:
        size = max(40, math.ceil(extent / occupancy))
        if size > min(width, height):
            continue
        x0 = min(max(round(cx - size / 2), 0), width - size)
        y0 = min(max(round(cy - size / 2), 0), height - size)
        if abs(x0 + size / 2 - cx) <= 0.08 * size and abs(y0 + size / 2 - cy) <= 0.08 * size and \
                inside(width, height, x0, y0, size):
            return {'x0': x0, 'y0': y0, 'size': size, 'angle': 0.0, 'occupancy_target': occupancy}
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    dims = {r['pool_id']: (int(r['width']), int(r['height']))
            for r in csv.DictReader(open('results/curation/integrity.csv', newline='', encoding='utf-8'))}
    rows = candidate_rows()
    tight = set()
    for path in args.reviews:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r['verdict'] == 'bad' and 'too tight' in r['reason']:
                tight.add(r['candidate_id'])
            elif r['candidate_id'] in tight:
                tight.discard(r['candidate_id'])
    out, skipped = [], 0
    for cid in sorted(tight):
        row = rows[cid]
        geometry = resquare(json.loads(row['box']), *dims[row['pool_id']])
        if geometry is None:
            skipped += 1
            continue
        out.append(row | geometry | {'candidate_id': f"{cid}-o{round(geometry['occupancy_target'] * 100)}",
                                     'parent_candidate_id': cid, 'origin': row['origin'] + '+recrop'})
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print({'tight': len(tight), 'recropped': len(out), 'did_not_fit': skipped})


if __name__ == '__main__':
    main()

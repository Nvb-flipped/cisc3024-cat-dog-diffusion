"""Measure VLM screening against existing personal crop verdicts (pool images only).

  queue   write the personal-verdict queue (exact reviewed geometry), split source-disjointly
          into a calibration half (prompt development) and a held-out half (final estimate)
  score   compare a VLM ledger with the personal verdicts: PASS precision, recall of personal
          good crops, species agreement, with Wilson 95% intervals

Personal verdicts are read, never modified. Dev images are never used.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from src.common import write_json
from src.curation.freeze import latest_decisions

OUT = Path('results/fullpool_20261119/calibration')


def wilson(successes, total, z=1.959964):
    if total == 0:
        return None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(p, 4), round(centre - half, 4), round(centre + half, 4)]


def build_queue(decisions='annotations/decisions.jsonl'):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in latest_decisions(decisions).values():
        if item.get('verdict') not in ('good', 'bad') or item['candidate_id'].startswith('reject:'):
            continue
        if item.get('size') in (None, '') or item.get('x0') in (None, ''):
            continue
        split = 'holdout' if int(hashlib.sha256(f"vlm-cal:{item['pool_id']}".encode()).hexdigest(), 16) % 2 else 'calibration'
        rows.append({'candidate_id': item['candidate_id'], 'pool_id': item['pool_id'], 'x0': item['x0'],
                     'y0': item['y0'], 'size': item['size'], 'angle': item.get('angle') or 0,
                     'box_width': item.get('head_width') or '', 'personal_verdict': item['verdict'],
                     'personal_label': item['label'], 'split': split})
    rows.sort(key=lambda r: (r['split'], r['pool_id'], r['candidate_id']))
    for split in ('calibration', 'holdout'):
        subset = [r for r in rows if r['split'] == split]
        with (OUT / f'personal_{split}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
        print(split, len(subset), dict(Counter(r['personal_verdict'] for r in subset)))


def score(queue, ledger, out=None):
    with open(queue, newline='', encoding='utf-8') as stream:
        truth = {r['candidate_id']: r for r in csv.DictReader(stream)}
    records = {}
    with open(ledger, encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['candidate_id'] in truth:
                records[record['candidate_id']] = record
    table = Counter()
    species_ok = species_total = 0
    gated = Counter()
    for cid, record in records.items():
        good = truth[cid]['personal_verdict'] == 'good'
        table[(good, record['resolved'])] += 1
        # The real pipeline also applies the deterministic geometry gate (width >= 32, 60-80%).
        width, side = truth[cid]['box_width'], float(truth[cid]['size'])
        geometry = width != '' and float(width) >= 32 and 0.60 <= float(width) / side <= 0.80
        gated[(good, record['resolved'] if geometry else 'GEOMETRY_GATE')] += 1
        if good and record['verdict'] and record['verdict']['species'] in ('cat', 'dog'):
            species_total += 1
            species_ok += int(record['verdict']['species'] == ('cat', 'dog')[int(truth[cid]['personal_label'])])
    passed = table[(True, 'PASS')] + table[(False, 'PASS')]
    goods = sum(v for (g, _), v in table.items() if g)
    bads = sum(v for (g, _), v in table.items() if not g)
    result = {'queue': str(queue), 'ledger': str(ledger), 'scored': len(records), 'of': len(truth),
              'confusion': {f'{"good" if g else "bad"}->{r}': v for (g, r), v in sorted(table.items())},
              'pass_precision': wilson(table[(True, 'PASS')], passed),
              'good_recall_as_pass': wilson(table[(True, 'PASS')], goods),
              'bad_rejected_or_uncertain': wilson(bads - table[(False, 'PASS')], bads),
              'species_agreement_on_good': wilson(species_ok, species_total),
              'with_geometry_gate': {
                  'confusion': {f'{"good" if g else "bad"}->{r}': v for (g, r), v in sorted(gated.items())},
                  'pass_precision': wilson(gated[(True, 'PASS')], gated[(True, 'PASS')] + gated[(False, 'PASS')]),
                  'good_recall_as_pass': wilson(gated[(True, 'PASS')], goods)}}
    if out:
        write_json(out, result)
    print(json.dumps(result, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('queue', 'score'))
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    build_queue() if args.command == 'queue' else score(args.queue, args.ledger, args.out)


if __name__ == '__main__':
    main()

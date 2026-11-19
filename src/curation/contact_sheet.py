"""Numbered contact sheets of exact candidate squares for agent visual QA (no verdicts written).

  python -m src.curation.contact_sheet --ledger L --resolved PASS --n 48 --seed 1 --out sheet.png
Samples ledger records uniformly at random (seeded) and renders each square with the official
crop function at 128 px, numbered, plus a JSON index binding numbers to candidate ids.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from src.common import ROOT

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, crop  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--resolved', nargs='+', default=['PASS'])
    parser.add_argument('--n', type=int, default=48)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.ledger, encoding='utf-8')]
    records = [r for r in records if r['resolved'] in args.resolved]
    sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
    pool, tile = Pool('data/student'), 128
    rows = (len(sample) + args.cols - 1) // args.cols
    sheet = Image.new('RGB', (args.cols * tile, rows * (tile + 14)), 'white')
    draw = ImageDraw.Draw(sheet)
    index = []
    for i, record in enumerate(sample):
        g = record['geometry']
        image = crop(pool.image(record['pool_id']), int(g['x0']), int(g['y0']), int(g['size']),
                     float(g['angle'] or 0), res=tile)
        x, y = (i % args.cols) * tile, (i // args.cols) * (tile + 14)
        sheet.paste(image, (x, y + 14))
        species = (record.get('verdict') or {}).get('species', '?')
        draw.text((x + 2, y + 1), f'{i} {species[:3]}', fill='black')
        index.append({'n': i, 'candidate_id': record['candidate_id'], 'pool_id': record['pool_id'],
                      'geometry': g, 'species': species, 'resolved': record['resolved']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    args.out.with_suffix('.json').write_text(json.dumps({'ledger': str(args.ledger), 'seed': args.seed,
                                                         'resolved': args.resolved, 'items': index}, indent=1))
    print(len(records), 'eligible;', len(sample), 'sampled ->', args.out)


if __name__ == '__main__':
    main()

"""Save a deterministic diagnostic source/crop sheet; this does not label any images."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from src.common import ROOT
from src.curation.geometry import crop
sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='results/curation/pilot', type=Path)
    p.add_argument('--count', default=16, type=int)
    p.add_argument('--seed', default=3024, type=int)
    p.add_argument('--min-score', default=0.0, type=float)
    p.add_argument('--out', type=Path)
    p.add_argument('--scores', type=Path, help='Inspect highest review-priority scores, not a random sample')
    p.add_argument('--unique-source', action='store_true')
    args = p.parse_args()
    records = [json.loads(line) for line in (args.run / 'detections.jsonl').read_text().splitlines()]
    choices = [c for r in records for c in r['candidates'] if c['geometry'] and c['score'] >= args.min_score]
    if args.scores:
        with args.scores.open(newline='', encoding='utf-8') as f:
            scores = {r['candidate_id']: r for r in csv.DictReader(f)}
        choices = [c | {'photo_hint': int(scores[c['candidate_id']]['crop_species_hint'])}
                   for c in choices if c['candidate_id'] in scores]
        choices.sort(key=lambda c: -float(scores[c['candidate_id']]['review_priority_score']))
    else:
        random.Random(args.seed).shuffle(choices)
    if args.unique_source:
        by_source = {}
        for candidate in choices:
            by_source.setdefault(candidate['pool_id'], candidate)
        choices = list(by_source.values())
    choices = choices[:args.count]
    pool = Pool('data/student')
    sheet = Image.new('RGB', (4 * 320, ((len(choices) + 3) // 4) * 240), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:\\Windows\\Fonts\\arial.ttf', 14)
    for index, c in enumerate(choices):
        x, y = index % 4 * 320, index // 4 * 240
        source = pool.image(c['pool_id'])
        tile = source.copy()
        ImageDraw.Draw(tile).rectangle((c['geometry']['x0'], c['geometry']['y0'],
            c['geometry']['x0'] + c['geometry']['size'], c['geometry']['y0'] + c['geometry']['size']), outline='red', width=2)
        tile.thumbnail((205, 195))
        sheet.paste(tile, (x, y + 25))
        head = crop(source, **c['geometry'])
        sheet.paste(head.resize((96, 96), Image.Resampling.NEAREST), (x + 215, y + 50))
        label = 'cat' if c['label'] == 0 else 'dog'
        hint = f' / hint {"cat" if c["photo_hint"] == 0 else "dog"}' if 'photo_hint' in c else ''
        draw.text((x + 5, y + 2), f'{c["pool_id"]} {label}{hint}', fill='black', font=font)
        draw.text((x + 215, y + 150), '32px -> 96px', fill='black', font=font)
    out = args.out or args.run / 'images/diagnostic_sources.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    out.with_suffix('.json').write_text(json.dumps(choices, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()

"""Save a deterministic diagnostic source/crop sheet; this does not label any images."""
import argparse
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
    args = p.parse_args()
    records = [json.loads(line) for line in (args.run / 'detections.jsonl').read_text().splitlines()]
    choices = [c for r in records for c in r['candidates'] if c['geometry'] and c['score'] >= args.min_score]
    choices = random.Random(args.seed).sample(choices, min(args.count, len(choices)))
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
        draw.text((x + 5, y + 2), f'{c["pool_id"]} {"cat" if c["label"] == 0 else "dog"} {c["score"]:.2f}', fill='black', font=font)
        draw.text((x + 215, y + 150), '32px -> 96px', fill='black', font=font)
    out = args.out or args.run / 'images/diagnostic_sources.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    out.with_suffix('.json').write_text(json.dumps(choices, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()

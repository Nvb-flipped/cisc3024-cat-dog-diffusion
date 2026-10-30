"""Create three synthetic colors for a local UI check; never student/course judgments."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.curation.review import CRITERIA, rendered_crop
from catdog_eval.data import Pool


def main():
    root = ROOT / 'private_build/duplicate_ui_fixture_v001'
    root.mkdir(parents=True, exist_ok=False)
    images = root / 'pool/images'
    images.mkdir(parents=True)
    colors = {'fixture_a': (120, 80, 40), 'fixture_b': (120, 80, 40), 'fixture_c': (40, 80, 120)}
    for identifier, color in colors.items():
        Image.new('RGB', (120, 120), color).save(images / f'{identifier}.jpg')
    with (root / 'pool/index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['pool_id', 'width', 'height'])
        writer.writeheader()
        writer.writerows(dict(pool_id=key, width=120, height=120) for key in colors)
    pool = Pool(root / 'pool')
    geometry = dict(x0=20, y0=20, size=60, angle=0)
    with (root / 'crop_decisions.jsonl').open('w') as stream:
        for identifier in colors:
            event = dict(candidate_id=identifier, pool_id=identifier, label=int(identifier == 'fixture_c'),
                **geometry, verdict='good', origin='software_fixture_crop_review',
                reviewer='AUTOMATED SYNTHETIC SOFTWARE FIXTURE', timestamp='fixture setup, not student review',
                criteria=dict.fromkeys(CRITERIA, True), diagnostic_only=True,
                preview_sha256=hashlib.sha256(rendered_crop(pool, identifier, geometry)).hexdigest())
            stream.write(json.dumps(event) + '\n')
    (root / 'integrity.csv').write_text('pool_id,status,dhash\n' + ''.join(f'{key},ok,0000000000000000\n' for key in colors))
    print(root)


if __name__ == '__main__':
    main()

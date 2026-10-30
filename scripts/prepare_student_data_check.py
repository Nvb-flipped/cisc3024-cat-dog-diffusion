"""Prepare 30 unreviewed local pairs for the student's independent data check."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'starter_kit'))
from src.common import sha256, write_json
from src.curation.geometry import crop as project_crop, inside
from src.curation.review import parse_geometry
from catdog_eval.data import Pool, crop as official_crop
from catdog_eval.manifest import COLUMNS


def selection(pool):
    pilot = ROOT / 'results/curation/pilot'
    selected = set(json.loads((pilot / 'calibration_selection.json').read_text())['proposed_candidate_ids'])
    with (pilot / 'candidates.csv').open(newline='') as stream:
        candidates = [row for row in csv.DictReader(stream) if row['candidate_id'] in selected]
    with (ROOT / 'results/curation/photo_screen/scores.csv').open(newline='') as stream:
        hints = {row['candidate_id']: row for row in csv.DictReader(stream)}
    def stable(row):
        return hashlib.sha256(('independent-data-check-3024:' + row['candidate_id']).encode()).hexdigest()
    def edge_distance(row):
        width, height = pool.size(row['pool_id'])
        g = parse_geometry(row)
        return min(g['x0'], g['y0'], width-g['x0']-g['size'], height-g['y0']-g['size'])
    chosen, sources = [], set()
    def take(rows, count, reason):
        for row in rows:
            if row['pool_id'] not in sources:
                chosen.append(dict(row) | {'selection_reason': reason})
                sources.add(row['pool_id'])
                count -= 1
                if count == 0:
                    break
    disagreements = [row for row in candidates if row['candidate_id'] in hints and
                     int(hints[row['candidate_id']]['crop_species_hint']) != int(row['label'])]
    take(sorted(disagreements, key=stable), 6, 'Detector/CLIP species suggestions disagree; neither is ground truth')
    take(sorted(candidates, key=lambda row: (edge_distance(row), stable(row))), 6, 'Near a source boundary')
    take(sorted(candidates, key=lambda row: (float(row['head_width']), stable(row))), 6, 'Small proposed source-head width; measure independently')
    take(sorted(candidates, key=stable), 29-len(chosen), 'Fixed diagnostic selection from calibration proposals')
    for row in sorted(candidates, key=stable):
        geometry = parse_geometry(row) | {'angle': 10.0}
        if row['pool_id'] not in sources and inside(*pool.size(row['pool_id']), **geometry):
            chosen.append(dict(row) | geometry | {'candidate_id': row['candidate_id'] + ':rotation-probe',
                'selection_reason': '10-degree geometry probe only; not an approved rotation/crop'})
            break
    if len(chosen) != 30:
        raise ValueError('Could not prepare exactly 30 pairs including a valid rotation probe')
    return chosen


def main():
    out = ROOT / 'private_build/student_data_check_v001'
    if out.exists():
        raise FileExistsError('Check pack exists; preserve any personal measurements and use a new version')
    pool = Pool(ROOT / 'data/student')
    rows = selection(pool)
    images = out / 'images'
    images.mkdir(parents=True)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 14)
    title_font = ImageFont.truetype(str(font_path), 20)
    sheet = Image.new('RGB', (1260, 2460), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 10), '30 unreviewed source/crop pairs — independent study, NOT the final audit', font=title_font, fill='black')
    records = []
    for index, row in enumerate(rows, 1):
        geometry = parse_geometry(row)
        source = pool.image(row['pool_id'])
        official, project = official_crop(source, **geometry), project_crop(source, **geometry)
        difference = int(np.abs(np.asarray(official).astype(np.int16) - np.asarray(project).astype(np.int16)).max())
        stem = f'{index:02d}_{row["pool_id"]}'
        source.save(images / f'{stem}_source.png')
        official.save(images / f'{stem}_official32.png')
        project.save(images / f'{stem}_project32.png')
        thumbnail = source.copy()
        thumbnail.thumbnail((180, 160), Image.Resampling.LANCZOS)
        x, y = (index-1) % 3 * 420, 55 + (index-1) // 3 * 240
        draw.text((x+10,y), f'{index:02d} {row["pool_id"]} · proposed {"cat" if int(row["label"])==0 else "dog"}', font=font, fill='black')
        draw.text((x+10,y+20), 'source (thumbnail)    official 32px     project 32px', font=font, fill='#475569')
        sheet.paste(thumbnail, (x+10, y+45))
        sheet.paste(official.resize((96,96), Image.Resampling.NEAREST), (x+202,y+45))
        sheet.paste(project.resize((96,96), Image.Resampling.NEAREST), (x+310,y+45))
        draw.text((x+10,y+211), f'x={geometry["x0"]}, y={geometry["y0"]}, side={geometry["size"]}, angle={geometry["angle"]:g}; diff={difference}', font=font, fill='#475569')
        records.append({'case': index, 'candidate_id': row['candidate_id'], 'pool_id': row['pool_id'],
            'label': int(row['label']), **geometry, 'selection_reason': row['selection_reason'],
            'official_project_max_absolute_pixel_difference': difference,
            'source_size': list(source.size), 'files': {name: str((images / f'{stem}_{suffix}.png').relative_to(out))
                for name, suffix in [('source','source'),('official','official32'),('project','project32')]}})
    sheet.save(out / 'contact_sheet.png')
    with (out / 'diagnostic_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    fields = ['case', 'pool_id', 'label', 'x0', 'y0', 'size', 'angle', 'selection_reason',
              'student_verdict', 'student_species', 'student_head_width_px', 'student_occupancy', 'student_notes']
    with (out / 'personal_measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    write_json(out / 'cases.json', {'diagnostic_only': True, 'human_check': 'pending; no student labels or measurements supplied',
        'sampling': 'Purposefully difficult selection from the fixed calibration set, not a precision estimate', 'cases': records})
    write_json(ROOT / 'results/student_checks/data_pack.json', {'diagnostic_only': True, 'pairs': len(records),
        'rotated_geometry_probes': sum(bool(row['angle']) for row in records),
        'pixel_parity_max_absolute_difference': max(row['official_project_max_absolute_pixel_difference'] for row in records),
        'manifest_sha256': sha256(out / 'diagnostic_manifest.csv'), 'cases_sha256': sha256(out / 'cases.json'),
        'generator_sha256': sha256(Path(__file__)), 'human_check': 'pending', 'local_pack': str(out),
        'scope': 'Agent-prepared engineering study pack; no authentic student check, final audit or accepted training data.'})
    print(out)


if __name__ == '__main__':
    main()

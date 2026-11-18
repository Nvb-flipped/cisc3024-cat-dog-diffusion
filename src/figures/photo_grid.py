"""Build fixed-position generated and nearest-neighbor grids from local receipts."""

import argparse
import base64
import csv
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.figures.build import SVG, XLINK


def label(identifier, text, x, y, *, bold=False, color='#202020', size=8, anchor='start'):
    return {'id': identifier, 'kind': 'text', 'text': text, 'x': x, 'y': y, 'bold': bold,
            'color': color, 'size': size, 'anchor': anchor}


def tile(identifier, image, x, y, side):
    if image.size != (32, 32) or image.mode != 'RGB':
        raise ValueError(f'Expected a native RGB 32×32 tile for {identifier}')
    return {'id': identifier, 'kind': 'image', 'image': image, 'x': x, 'y': y, 'side': side}


def save_assets(out, spec, groups, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.image import BboxImage
    from matplotlib.transforms import Bbox, TransformedBbox

    width, height = spec['width'], spec['height']
    out.mkdir(parents=True, exist_ok=False)
    originals = out / 'tiles'
    originals.mkdir()
    svg = ET.Element(f'{{{SVG}}}svg', {'width': f'{width}pt', 'height': f'{height}pt',
        'viewBox': f'0 0 {width} {height}', 'font-family': 'Arial', 'font-size': '8'})
    ET.SubElement(svg, f'{{{SVG}}}title').text = spec['title']
    ET.SubElement(svg, f'{{{SVG}}}desc').text = spec['caption']
    source_tiles = {}
    with plt.rc_context({'font.family': 'Arial', 'pdf.fonttype': 42}):
        figure = plt.figure(figsize=(width / 72, height / 72), facecolor='white')
        for group, items in groups:
            node = ET.SubElement(svg, f'{{{SVG}}}g', {'id': group})
            for item in items:
                x, y = item['x'], item['y']
                attrs = {'id': item['id'], 'x': str(x), 'y': str(y)}
                if item['kind'] == 'text':
                    weight = 'bold' if item['bold'] else 'normal'
                    attrs.update({'fill': item['color'], 'font-size': str(item['size']),
                                  'font-weight': weight, 'text-anchor': item['anchor']})
                    ET.SubElement(node, f'{{{SVG}}}text', attrs).text = item['text']
                    figure.text(x / width, 1 - y / height, item['text'], fontsize=item['size'],
                                ha='center' if item['anchor'] == 'middle' else 'left',
                                va='baseline', color=item['color'], weight=weight)
                    continue
                side = item['side']
                original = originals / f"{item['id']}.png"
                item['image'].save(original)
                source_tiles[item['id']] = sha256(original)
                display = item['image'].resize((128, 128), Image.Resampling.NEAREST)
                buffer = io.BytesIO()
                display.save(buffer, format='PNG')
                attrs.update({'width': str(side), 'height': str(side),
                              f'{{{XLINK}}}href': 'data:image/png;base64,' +
                              base64.b64encode(buffer.getvalue()).decode('ascii')})
                ET.SubElement(node, f'{{{SVG}}}image', attrs)
                box = Bbox.from_bounds(x / width, 1 - (y + side) / height, side / width, side / height)
                bitmap = BboxImage(TransformedBbox(box, figure.transFigure), interpolation='nearest')
                bitmap.set_data(display)
                figure.add_artist(bitmap)
        figure.savefig(out / 'figure.png', dpi=300)
        figure.savefig(out / 'figure.pdf')
        plt.close(figure)
    ET.indent(svg)
    ET.ElementTree(svg).write(out / 'figure.svg', encoding='utf-8', xml_declaration=True)
    with (out / 'source.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / 'source_summary.json', {'inputs': spec['sources'], 'source_tiles_sha256': source_tiles,
                                             'native_pixels': [32, 32], 'display_pixels': [128, 128]})
    write_json(out / 'figure.spec.json', spec | {'picture_count': len(source_tiles)})
    (out / 'caption.txt').write_text(spec['caption'] + '\n', encoding='utf-8')


def generated(result, out):
    path = result.with_suffix('.png')
    grid = Image.open(path).convert('RGB')
    if grid.size != (546, 274):
        raise ValueError('Unexpected official grid dimensions')
    groups, rows = [], []
    for label_id, name in enumerate(('Cat', 'Dog')):
        top = 18 + 126 * label_id
        objects = [label(f'{name.lower()}_label', name, 19, top, bold=True, size=9)]
        for index in range(16):
            source_x, source_y = 2 + index * 34, 2 + label_id * 4 * 34
            image = grid.crop((source_x, source_y, source_x + 32, source_y + 32))
            row, column = divmod(index, 8)
            identifier = f'{name.lower()}_{index:02d}'
            objects.append(tile(identifier, image, 19 + column * 59, top + 7 + row * 52, 50))
            rows.append({'tile': identifier, 'class': name.lower(), 'label': label_id,
                         'class_index_zero_based': index, 'official_grid_x': source_x, 'official_grid_y': source_y})
        groups.append((f'n:{name.lower()}', objects))
    groups.append(('n:protocol', [label('protocol', 'Official generation seed 0 · first 16 per class',
                                         252, 273, anchor='middle', size=7)]))
    caption = ('The first 16 generated crops in each class from the unchanged official seed-0 grid. '
               'The original 32×32 RGB tiles are enlarged using nearest-neighbor interpolation.')
    save_assets(out, {'id': 'F6', 'kind': 'photo_grid', 'title': 'Generated cat and dog heads',
        'width': 504, 'height': 282, 'caption': caption, 'sources': {str(p): sha256(p) for p in (result, path)}}, groups, rows)


def nearest(result, replay, out):
    meta = json.loads((replay / 'replay.json').read_text(encoding='utf-8'))
    if meta['seed'] != 0 or meta['official_result_sha256'] != sha256(result):
        raise ValueError('Nearest-neighbor replay does not match the seed-0 official result')
    with (replay / 'neighbors.csv').open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    groups, chosen = [], []
    for label_id, name in enumerate(('Cat', 'Dog')):
        top = 23 + 91 * label_id
        groups.append((f'n:{name.lower()}', [label(f'{name.lower()}_label', name, 17, top - 10, bold=True, size=9)]))
        for status_index, status in enumerate(('True', 'False')):
            matching = [row for row in records if row['label'] == str(label_id) and row['flagged'] == status][:2]
            if not matching:
                key = f'{name.lower()}_missing_{status.lower()}'
                status_label = 'flagged' if status == 'True' else 'unflagged'
                groups.append((f'n:{key}', [label(key, f'No {status_label} examples',
                                                    17 + 240 * status_index, top, size=7)]))
            for position, record in enumerate(matching):
                generated_index = int(record['generated_index'])
                if generated_index not in meta['examples']:
                    raise ValueError(f'The predeclared replay did not retain index {generated_index}')
                identifier = f'sample_{generated_index:04d}'
                x = 17 + (2 * status_index + position) * 120
                status_text = 'Flagged' if status == 'True' else 'Unflagged'
                status_color = '#9b3429' if status == 'True' else '#343434'
                objects = [label(f'{identifier}_status', f'{status_text} · #{generated_index}', x, top,
                                 bold=True, color=status_color)]
                for column, (kind, heading) in enumerate((('generated', 'Generated'), ('nearest', 'Nearest'), ('second', 'Second'))):
                    image = Image.open(replay / f'{generated_index:04d}_{kind}.png').convert('RGB')
                    image_x = x + column * 36
                    objects.append(tile(f'{identifier}_{kind}', image, image_x, top + 5, 32))
                    objects.append(label(f'{identifier}_{kind}_label', heading, image_x + 16, top + 46,
                                         anchor='middle', size=7))
                ratio = float(record['ratio'])
                d1, d2 = float(record['d1']), float(record['d2'])
                objects.append(label(f'{identifier}_distances', f'{d1:.2f} / {d2:.2f} = {ratio:.3f}', x, top + 59, size=7))
                groups.append((f'n:{identifier}', objects))
                chosen.append(record)
    groups.append(('n:protocol', [label('protocol', 'Official generation seed 0 · pixel L2 distance ratio cutoff: 0.5',
                                         252, 199, anchor='middle', size=7)]))
    caption = ('The first two flagged and first two unflagged indices per requested class in the seed-0 pixel-neighbor replay. '
               'Each example shows a generated crop and its two closest rebuilt training references, which may be mirrored. '
               'The recorded first and second pixel L2 distances and their ratio appear below each triplet. '
               'The official evaluator flags ratios below 0.5. All tiles contain 32×32 RGB pixels.')
    inputs = (result, replay / 'neighbors.csv', replay / 'replay.json')
    save_assets(out, {'id': 'F7', 'kind': 'photo_grid', 'title': 'Memorization examples',
        'width': 504, 'height': 210, 'caption': caption, 'sources': {str(p): sha256(p) for p in inputs}}, groups, chosen)


def build(result, submission, out, replay=None):
    validated_result(result, 0, submission)
    if replay:
        nearest(result, replay, out)
    else:
        generated(result, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='Use the matching pixel-neighbor replay to build F7')
    args = parser.parse_args()
    build(args.result, args.submission, args.out, args.replay)


if __name__ == '__main__':
    main()

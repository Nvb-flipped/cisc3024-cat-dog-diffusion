"""Verify semantic object/text parity, XML references and physical figure dimensions."""
import argparse
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from src.common import sha256, write_json

SVG = 'http://www.w3.org/2000/svg'
DRAWING = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PRESENTATION = 'http://schemas.openxmlformats.org/presentationml/2006/main'


def validate(folder):
    folder = Path(folder)
    svg = ET.parse(folder / 'figure.svg').getroot()
    drawio_data = (folder / 'figure.drawio').read_bytes()
    if drawio_data != (folder / 'figure.xml').read_bytes():
        raise ValueError('Plain XML and draw.io source differ')
    drawio = ET.fromstring(drawio_data)
    if drawio.find('./diagram/mxGraphModel/root') is None:
        raise ValueError('Draw.io source must be uncompressed and readable')
    cells = drawio.findall('.//mxCell')
    ids = [cell.get('id') for cell in cells]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate draw.io object IDs')
    for cell in cells:
        for reference in ['source', 'target', 'parent']:
            if cell.get(reference) and cell.get(reference) not in ids:
                raise ValueError(f'Dangling draw.io {reference}')
    svg_text = [''.join(node.itertext()) for node in svg.iter(f'{{{SVG}}}text')]
    drawio_text = [cell.get('value') for cell in cells if cell.get('value')]
    if sorted(svg_text) != sorted(drawio_text):
        raise ValueError('Draw.io labels/values differ from the SVG')
    spec = json.loads((folder / 'figure.spec.json').read_text())
    objects = spec['objects']
    picture_count = spec.get('picture_count', 0)
    svg_pictures = list(svg.iter(f'{{{SVG}}}image'))
    drawio_pictures = [cell for cell in cells if 'shape=image;' in cell.get('style', '')]
    if len(svg_pictures) != picture_count or len(drawio_pictures) != picture_count:
        raise ValueError('Picture inventory differs across SVG, specification and draw.io')
    svg_ids = {element.get('id') for element in svg.iter()}
    if any(item['id'] not in svg_ids or item['id'] not in ids for item in objects):
        raise ValueError('Semantic IDs differ across SVG/spec/draw.io')
    with zipfile.ZipFile(folder / 'figure.pptx') as z:
        slide = ET.fromstring(z.read('ppt/slides/slide1.xml'))
        size = ET.fromstring(z.read('ppt/presentation.xml')).find(f'{{{PRESENTATION}}}sldSz')
        width_pt, height_pt = int(size.get('cx')) / 12700, int(size.get('cy')) / 12700
        pptx_text = [node.text or '' for node in slide.iter(f'{{{DRAWING}}}t')]
        if sorted(svg_text) != sorted(pptx_text):
            raise ValueError('Native PowerPoint labels/values differ from the SVG')
        if len(list(slide.iter(f'{{{PRESENTATION}}}pic'))) != picture_count:
            raise ValueError('The count of native PowerPoint pictures differs from the specification')
    if not math.isclose(width_pt, spec['width'], abs_tol=.001) or not math.isclose(height_pt, spec['height'], abs_tol=.001):
        raise ValueError('PowerPoint canvas differs from the canonical specification')
    with Image.open(folder / 'figure.png') as im:
        png = {'pixels': list(im.size), 'dpi': list(im.info.get('dpi', ())), 'mode': im.mode}
    if png['pixels'] != [round(spec['width'] / 72 * 300), round(spec['height'] / 72 * 300)]:
        raise ValueError('PNG physical dimensions differ from the requested 300-dpi canvas')
    application_checks = None
    application_path = folder / 'application_qa.json'
    if application_path.exists():
        application_checks = json.loads(application_path.read_text())
        if application_checks['reference_drawio_sha256'] != sha256(folder / 'figure.drawio'):
            raise ValueError('Native application evidence is stale for the current draw.io figure')
        powerpoint_check = application_checks.get('powerpoint_check', {})
        if powerpoint_check and powerpoint_check.get('reference_pptx_sha256') != sha256(folder / 'figure.pptx'):
            raise ValueError('Native application evidence is stale for the current PowerPoint figure')
    powerpoint_status = 'pending'
    drawio_status = 'pending'
    if application_checks:
        powerpoint_status = application_checks.get('powerpoint_check', {}).get('status', 'pending')
        if 'drawio_check' in application_checks:
            drawio_status = application_checks['drawio_check']['status']
        else:
            drawio_status = ('text/mark and connected-node edit-save-reopen tested'
                if application_checks.get('connection_check', {}).get('status') == 'passed'
                else 'text/mark edit-save-reopen tested; connected-node test pending')
    report = {'status': 'file_structure_and_semantic_parity_passed', 'svg_text_count': len(svg_text),
        'objects': len(objects), 'native_pictures': picture_count,
        'drawio_edges': sum(cell.get('edge') == '1' for cell in cells),
        'png': png, 'pptx_canvas_pt': [width_pt, height_pt],
        'actual_powerpoint_edit_save_render': powerpoint_status,
        'actual_drawio_edit_save_render': drawio_status,
        'application_checks': application_checks,
        'scope': 'File parity plus separately recorded native-operation evidence; final human design acceptance remains pending',
        'files': {name: sha256(folder / name) for name in ['figure.svg', 'figure.png', 'figure.pdf',
            'figure.pptx', 'figure.drawio', 'figure.xml', 'figure.spec.json', 'source.csv', 'source_summary.json']}}
    write_json(folder / 'export_qa.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    args = p.parse_args()
    print(validate(args.folder))


if __name__ == '__main__':
    main()

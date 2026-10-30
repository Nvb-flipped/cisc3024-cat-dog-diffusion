"""Convert the supported scientific SVG subset to readable native draw.io cells."""
import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import ImageFont
from src.common import write_json

SVG = 'http://www.w3.org/2000/svg'
NUMBER = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'
IDENTITY = (1, 0, 0, 1, 0, 0)


def multiply(m, n):
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a*A+c*B, b*A+d*B, a*C+c*D, b*C+d*D, a*E+c*F+e, b*E+d*F+f)


def transform(value):
    result = IDENTITY
    for name, arguments in re.findall(r'(\w+)\(([^)]*)\)', value or ''):
        numbers = [float(v) for v in re.findall(NUMBER, arguments)]
        if name == 'translate':
            matrix = (1, 0, 0, 1, numbers[0], numbers[1] if len(numbers) > 1 else 0)
        elif name == 'rotate':
            angle = math.radians(numbers[0])
            cx, cy = numbers[1:] if len(numbers) == 3 else (0, 0)
            cosine, sine = math.cos(angle), math.sin(angle)
            matrix = multiply((1, 0, 0, 1, cx, cy), multiply(
                (cosine, sine, -sine, cosine, 0, 0), (1, 0, 0, 1, -cx, -cy)))
        elif name == 'matrix' and len(numbers) == 6:
            matrix = tuple(numbers)
        else:
            raise ValueError(f'Unsupported SVG transform: {name}')
        result = multiply(result, matrix)
    return result


def point(matrix, x, y):
    a, b, c, d, e, f = matrix
    return a*x+c*y+e, b*x+d*y+f


def path_commands(data):
    tokens = re.findall(r'[A-Za-z]|' + NUMBER, data)
    result = []
    index = 0
    arity = {'M': 2, 'L': 2, 'C': 6, 'z': 0, 'Z': 0}
    while index < len(tokens):
        command = tokens[index]
        if command not in arity:
            raise ValueError(f'Unsupported path command: {command}')
        count = arity[command]
        values = [float(v) for v in tokens[index + 1:index + 1 + count]]
        result.append((command, values))
        index += count + 1
    return result


def svg_scene(svg_path):
    root = ET.parse(svg_path).getroot()
    scene = []
    counters = {}

    def walk(element, inherited, matrix, owner):
        tag = element.tag.rsplit('}', 1)[-1]
        if tag in ['metadata', 'defs', 'title', 'desc']:
            return
        style = dict(inherited)
        style.update({k: v for k, v in element.attrib.items()
                      if k in ['fill', 'stroke', 'stroke-width', 'font-size', 'font-family', 'font-weight', 'text-anchor']})
        style.update({k.strip(): v.strip() for k, v in (part.split(':', 1)
                      for part in element.get('style', '').split(';') if ':' in part)})
        matrix = multiply(matrix, transform(element.get('transform')))
        owner = element.get('id', owner)
        if tag in ['svg', 'g']:
            for child in element:
                walk(child, style, matrix, owner)
            return
        counters[owner] = counters.get(owner, 0) + 1
        identifier = f'{owner}_{counters[owner]}'
        if tag == 'text':
            size = float(style.get('font-size', 7))
            text = ''.join(element.itertext())
            font_path = 'C:\\Windows\\Fonts\\arialbd.ttf' if style.get('font-weight') in ['700', 'bold'] else 'C:\\Windows\\Fonts\\arial.ttf'
            font = ImageFont.truetype(font_path, round(size * 100))
            width, height = font.getlength(text) / 100 + 2*size, 2.2*size
            x, y = float(element.get('x', 0)), float(element.get('y', 0))
            anchor = style.get('text-anchor', 'start')
            left = x - {'start': 0, 'middle': width/2, 'end': width}.get(anchor, 0)
            cx, cy = point(matrix, left + width/2, y - .365*size)
            scene.append({'id': identifier, 'kind': 'text', 'text': text,
                'x': cx-width/2, 'y': cy-height/2, 'width': width, 'height': height,
                'rotation': math.degrees(math.atan2(matrix[1], matrix[0])),
                'font_size': size, 'anchor': anchor, 'bold': style.get('font-weight') in ['700', 'bold'],
                'color': style.get('fill', '#000000')})
        elif tag == 'path':
            commands = path_commands(element.get('d', ''))
            if any(command == 'C' for command, _ in commands):
                # Matplotlib's circle marker is eight cubic arcs centered at the symbol origin.
                endpoints = [values[-2:] for command, values in commands if command in ['M', 'C']]
                radii = [math.hypot(x, y) for x, y in endpoints]
                radius = radii[0]
                if len([c for c, _ in commands if c == 'C']) != 8 or max(abs(r-radius) for r in radii) > 1e-4:
                    raise ValueError('Unsupported curved primitive; do not rasterize it')
                cx, cy = point(matrix, 0, 0)
                scene.append({'id': identifier, 'kind': 'ellipse', 'x': cx-radius, 'y': cy-radius,
                              'width': radius*2, 'height': radius*2, 'style': style})
            else:
                vertices = [point(matrix, *values) for command, values in commands if command in ['M', 'L']]
                closed = commands[-1][0].lower() == 'z'
                if closed:
                    xs, ys = zip(*vertices)
                    if len(vertices) != 4 or len(set(xs)) != 2 or len(set(ys)) != 2:
                        raise ValueError('Only rectangular closed paths supported in this scientific chart exporter')
                    scene.append({'id': identifier, 'kind': 'rect', 'x': min(xs), 'y': min(ys),
                                  'width': max(xs)-min(xs), 'height': max(ys)-min(ys), 'style': style})
                else:
                    scene.append({'id': identifier, 'kind': 'line', 'points': vertices, 'style': style})
        else:
            raise ValueError(f'Unsupported SVG primitive: {tag}')

    walk(root, {}, IDENTITY, 'canvas')
    return [float(v) for v in root.get('viewBox').split()], scene


def drawio(svg_path, out):
    viewbox, scene = svg_scene(svg_path)
    scale = 96 / 72
    width, height = viewbox[2:]
    mxfile = ET.Element('mxfile', {'host': 'app.diagrams.net', 'compressed': 'false'})
    diagram = ET.SubElement(mxfile, 'diagram', {'id': 'F0', 'name': 'Diagnostic overfit'})
    model = ET.SubElement(diagram, 'mxGraphModel', {'dx': str(width*scale), 'dy': str(height*scale),
        'grid': '0', 'page': '1', 'pageScale': '1', 'pageWidth': str(width*scale), 'pageHeight': str(height*scale)})
    graph = ET.SubElement(model, 'root')
    ET.SubElement(graph, 'mxCell', {'id': '0'})
    ET.SubElement(graph, 'mxCell', {'id': '1', 'parent': '0'})

    def geometry(cell, x, y, w, h):
        return ET.SubElement(cell, 'mxGeometry', {'x': f'{x*scale:.6f}', 'y': f'{y*scale:.6f}',
            'width': f'{w*scale:.6f}', 'height': f'{h*scale:.6f}', 'as': 'geometry'})

    for item in scene:
        key, kind = item['id'], item['kind']
        if kind == 'text':
            style = ('text;html=0;whiteSpace=nowrap;overflow=visible;spacing=0;verticalAlign=middle;'
                f'align={ {"start":"left","middle":"center","end":"right"}[item["anchor"]]};'
                f'fontFamily=Arial;fontSize={item["font_size"]*scale:g};fontColor={item["color"]};'
                f'fontStyle={1 if item["bold"] else 0};rotation={item["rotation"]:g};')
            cell = ET.SubElement(graph, 'mxCell', {'id': key, 'value': item['text'], 'style': style,
                                                 'vertex': '1', 'parent': '1'})
            geometry(cell, item['x'], item['y'], item['width'], item['height'])
            continue
        style = item['style']
        fill = style.get('fill', '#000000') if kind != 'line' else 'none'
        stroke = style.get('stroke', 'none')
        stroke_width = float(style.get('stroke-width', 1)) * scale
        common = f'fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width:g};'
        if kind in ['ellipse', 'rect']:
            cell = ET.SubElement(graph, 'mxCell', {'id': key, 'value': '', 'parent': '1', 'vertex': '1',
                'style': ('ellipse;' if kind == 'ellipse' else 'rounded=0;') + common})
            geometry(cell, item['x'], item['y'], item['width'], item['height'])
        elif kind == 'line':
            vertices = item['points']
            for suffix, (x, y) in [('start', vertices[0]), ('end', vertices[-1])]:
                anchor = ET.SubElement(graph, 'mxCell', {'id': key+'-'+suffix, 'value': '', 'parent': '1',
                    'vertex': '1', 'style': 'opacity=0;fillColor=none;strokeColor=none;perimeter=none;'})
                geometry(anchor, x, y, 0, 0)
            cell = ET.SubElement(graph, 'mxCell', {'id': key, 'value': '', 'parent': '1', 'edge': '1',
                'source': key+'-start', 'target': key+'-end', 'style':
                'edgeStyle=none;curved=0;rounded=0;endArrow=none;startArrow=none;'
                'exitX=0.5;exitY=0.5;entryX=0.5;entryY=0.5;' + common})
            geom = ET.SubElement(cell, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
            points = ET.SubElement(geom, 'Array', {'as': 'points'})
            for x, y in vertices[1:-1]:
                ET.SubElement(points, 'mxPoint', {'x': f'{x*scale:.6f}', 'y': f'{y*scale:.6f}'})
    ET.indent(mxfile)
    data = ET.tostring(mxfile, encoding='utf-8', xml_declaration=True)
    out = Path(out)
    out.write_bytes(data)
    out.with_suffix('.xml').write_bytes(data)
    write_json(out.with_suffix('.scene.json'), {'units': 'points', 'canvas': [width, height], 'objects': scene,
        'source': str(svg_path), 'note': 'Draw.io is an editable vector drawing; underlying data remain in source.csv.'})
    return {'objects': len(scene), 'text': sum(s['kind'] == 'text' for s in scene), 'canvas_px': [width*scale, height*scale]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('svg', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    print(drawio(args.svg, args.out or args.svg.with_suffix('.drawio')))


if __name__ == '__main__':
    main()

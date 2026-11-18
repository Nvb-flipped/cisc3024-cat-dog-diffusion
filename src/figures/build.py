"""Build measured diagnostic plots with editable text and exact source-data copies."""
import argparse
import copy
import csv
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from src.common import ROOT, sha256, write_json

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', XLINK)


def normalized_svg(source, out, *, group_panels=False):
    """Expand Matplotlib symbol uses and explicit fonts for native-object conversion."""
    tree = ET.parse(source)
    root = tree.getroot()
    definitions = {element.get('id'): element for element in root.iter() if element.get('id')}
    for parent in list(root.iter()):
        for index, child in list(enumerate(parent)):
            if child.tag == f'{{{SVG}}}use':
                reference = child.get(f'{{{XLINK}}}href', '').lstrip('#')
                symbol = copy.deepcopy(definitions[reference])
                symbol.attrib.pop('id', None)
                x, y = float(child.get('x', 0)), float(child.get('y', 0))
                group = ET.Element(f'{{{SVG}}}g', {'transform': f'translate({x},{y})'})
                if child.get('style'):
                    group.set('style', child.get('style'))
                group.append(symbol)
                parent.remove(child)
                parent.insert(index, group)
    for element in root.iter():
        element.attrib.pop('clip-path', None)  # All plotted data/markers were placed within explicit axis limits.
        if element.tag == f'{{{SVG}}}text':
            style = element.get('style', '')
            match = re.search(r'font:\s*(?:(\d+|bold)\s+)?([\d.]+)px\s+[^;]+', style)
            if match:
                prefix = f'font-size:{match[2]};font-family:Arial;'
                if match[1] and match[1] not in ('400', 'normal'):
                    prefix += f'font-weight:{match[1]};'
                style = re.sub(r'font:[^;]+;?', prefix, style)
            style = re.sub(r'font-size:\s*([\d.]+)px', r'font-size:\1', style)
            size = re.search(r'font-size:\s*([\d.]+)', style)
            if size:
                element.set('font-size', size[1])
                style = re.sub(r'font-size:[^;]+;?', '', style)
            element.set('style', style)
    # Unused marker definitions carry <use>-free primitives but are not drawn.
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == f'{{{SVG}}}defs':
                parent.remove(child)
    root.set('font-family', 'Arial, Helvetica, sans-serif')
    root.set('font-size', '7')
    root.set('stroke-linecap', 'round')
    root.set('stroke-linejoin', 'round')
    if group_panels:
        canvas = next(child for child in root if child.get('id') == 'figure_1')
        if set(canvas.attrib) != {'id'}:
            raise ValueError('Plot canvas has inherited attributes that must be preserved')
        index = list(root).index(canvas)
        root.remove(canvas)
        for child in reversed(list(canvas)):
            root.insert(index, child)
    counters = {}
    def name_primitives(element, owner='canvas'):
        owner = element.get('id', owner)
        tag = element.tag.rsplit('}', 1)[-1]
        if tag in ['path', 'text', 'line', 'rect', 'circle', 'ellipse', 'polyline', 'polygon']:
            counters[owner] = counters.get(owner, 0) + 1
            element.set('id', f'{owner}_{counters[owner]}')
        for child in element:
            name_primitives(child, owner)
    name_primitives(root)
    tree.write(out, encoding='utf-8', xml_declaration=True)


def diagnostic_plot(entry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    out = Path(entry['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'figure.svg').exists():
        raise FileExistsError('Figure exists; use a fresh registry output for a revision')
    summary = json.loads(Path(entry['summary']).read_text())
    with Path(entry['data']).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    updates = [0] + [int(row['update']) for row in rows]
    noise = [summary['probe_mse_before']] + [float(row['probe_mse']) for row in rows]
    clean = [summary['reconstruction_mse_before']] + [float(row['reconstruction_mse']) for row in rows]
    style = {'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .6,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7, 2.65), dpi=72, facecolor='white')
        axes = [fig.add_axes([.105, .24, .345, .59]), fig.add_axes([.605, .24, .345, .59])]
        values = [noise, clean]
        colors = ['#0072B2', '#D55E00']
        titles = ['Noise prediction', 'Clean-image reconstruction']
        labels = ['Noise MSE', 'Image MSE (normalized pixels)']
        for index, (axis, series, color, title, ylabel) in enumerate(zip(axes, values, colors, titles, labels)):
            curve, = axis.plot(updates, series, '-o', color=color, linewidth=1,
                               markersize=2.7, markeredgewidth=.45)
            curve.set_gid(f'n:{"noise" if index == 0 else "reconstruction"}_curve')
            axis.set(xlim=(-50, 2050), ylim=(0, 1.2 if index == 0 else .6),
                     xticks=[0, 500, 1000, 1500, 2000], xlabel='Optimizer updates', ylabel=ylabel)
            axis.set_yticks([0, .3, .6, .9, 1.2] if index == 0 else [0, .15, .30, .45, .60])
            axis.tick_params(direction='out', length=2.5, width=.6, pad=2)
            axis.set_title(title, fontsize=7, pad=7)
            fig.text(.045 + index * .5, .865, chr(97 + index), fontsize=8, weight='bold')
            axis.text(.98, .90, f'{series[-1]:.4f} at update 2000', transform=axis.transAxes,
                      ha='right', va='top', fontsize=7)
        fig.text(.50, .035, 'Two pool images; fixed denoising probes; implementation diagnostic',
                 ha='center', fontsize=7)
        raw = out / 'matplotlib_reference.svg'
        fig.savefig(raw, format='svg')
        fig.savefig(out / 'figure.png', dpi=300)
        fig.savefig(out / 'figure.pdf')
        normalized_svg(raw, out / 'figure.svg')
        plt.close(fig)
    shutil.copy2(entry['data'], out / 'source.csv')
    shutil.copy2(entry['summary'], out / 'source_summary.json')
    caption = ('Fixed-budget integration diagnostic on two exact course-pool crops. '
        'The tiny randomly initialized U-Net is trained for 2,000 updates; these are not baseline or final-model results. '
        '(a) Fixed-probe noise-prediction MSE. (b) Reconstructed normalized-image MSE. '
        'Both points average the same two images at timesteps 250 and 750 with a fixed noise seed of 43. '
        'No smoothing or uncertainty intervals are applied; one run does not estimate training variability.')
    (out / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    write_json(out / 'figure.spec.json', {'id': entry['id'], 'kind': 'scientific-line-chart',
        'units': 'points', 'width': 504, 'height': 190.8, 'font': 'Arial',
        'font_file': font_manager.findfont('Arial', fallback_to_default=False),
        'source_files': {entry['data']: sha256(entry['data']), entry['summary']: sha256(entry['summary'])},
        'panels': [{'id': 'noise', 'x': updates, 'y': noise, 'ylim': [0, 1.2], 'color': colors[0]},
                   {'id': 'reconstruction', 'x': updates, 'y': clean, 'ylim': [0, .6], 'color': colors[1]}],
        'transformations': ['prepend the measured pre-update baseline to the logged fixed-probe values'],
        'caption': caption, 'alt_text': 'Noise and reconstruction errors decline over a fixed 2000-update, two-image integration check.',
        'status': 'Diagnostic evidence only; final data/model results remain pending.'})
    write_json(out / 'export_qa.json', {'matplotlib': matplotlib.__version__,
        'canvas_pt': [504, 190.8], 'png_dpi': 300, 'svg_text_editable': True,
        'native_pptx': 'pending', 'drawio': 'pending', 'human_final_size_review': 'pending',
        'publisher': 'Course project using Nature-style restraint; no journal-compliance claim'})
    print(out)


def verify_evaluation_table(rows):
    """Reconcile retained plot inputs to the official JSON without loading model weights."""
    for row in rows:
        path = (ROOT / row['source'].replace('\\', '/')).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != row['source_sha256']:
            raise ValueError(f'Official plot source does not match its receipt: {path}')
        result = json.loads(path.read_text(encoding='utf-8'))
        for field in ('accuracy', 'fid', 'recall', 'memorization_rate', 'num_samples'):
            if row[field] != result['metrics'][field]:
                raise ValueError(f'Plot value {field} differs from the official result: {path}')
        if row['valid'] != result['marks']['valid']:
            raise ValueError(f'Plot validity differs from the official result: {path}')


def checkpoint_rows(entry, manifest_hash):
    from src.evaluation.select import validated_result
    rows, sources = [], {}
    for series in entry['series']:
        for path, exported in series['results']:
            path, exported = Path(path), Path(exported)
            result = validated_result(path, 0, exported)
            lineage = json.loads((exported / 'provenance.json').read_text())
            sampler = json.loads((exported / 'inference.json').read_text())['sampling']
            if (lineage['training_seed'], lineage['weights_variant'], sampler['steps'], sampler['guidance'],
                lineage['manifest_sha256']) != (17, 'ema', 50, 2.5, manifest_hash):
                raise ValueError(f'Checkpoint plot requires the matched protocol: {path}')
            fields = ('accuracy', 'fid', 'recall', 'memorization_rate', 'num_samples')
            digest = sha256(path)
            rows.append({'model': series['name'], 'update': lineage['training_update'], 'generation_seed': 0,
                         **{field: result['metrics'][field] for field in fields}, 'valid': result['marks']['valid'],
                         'source': path.as_posix(), 'source_sha256': digest})
            sources[path.as_posix()] = digest
    rows.sort(key=lambda row: (row['model'], row['update']))
    return rows, sources


def checkpoint_metrics_plot(entry, out, summary_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    manifest_hash = sha256(entry['manifest'])
    if summary_path is None:
        rows, sources = checkpoint_rows(entry, manifest_hash)
    else:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary['manifest_sha256'] != manifest_hash:
            raise ValueError('Checkpoint metrics do not match the retained manifest')
        rows, sources = summary['rows'], summary['sources']
        verify_evaluation_table(rows)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'source.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panels = [('fid', 'FID', (60, 120), [60, 80, 100, 120], 1),
              ('recall', 'Recall (%)', (0, 40), [0, 10, 20, 30, 40], 100),
              ('memorization_rate', 'Memorization (%)', (-.3, 10), [0, 2, 4, 6, 8, 10], 100)]
    style = {'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': .6, 'svg.fonttype': 'none',
             'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7, 2.6), dpi=72, facecolor='white')
        for index, (metric, title, limits, ticks, scale) in enumerate(panels):
            axis = fig.add_axes([.086 + .316 * index, .23, .255, .59])
            axis.set_gid(f'n:{metric}')
            if metric == 'memorization_rate':
                axis.axhline(5, color='#858585', linewidth=.65, zorder=1)
                axis.text(30.35, 5.24, '5% limit', color='#575757', ha='right', va='bottom', fontsize=7)
            for series in entry['series']:
                points = [row for row in rows if row['model'] == series['name']]
                axis.plot([p['update'] / 1000 for p in points], [scale * p[metric] for p in points],
                          '-'+series['marker'], label=series['name'], color=series['color'], linewidth=1.1,
                          markersize=3.5, markeredgewidth=.55, zorder=2)
            axis.set(xlim=(19.5, 30.5), ylim=limits, xticks=[20, 22, 26, 30], yticks=ticks, title=title)
            axis.tick_params(direction='out', length=2.5, width=.6, pad=2)
            fig.text(.029 + .316 * index, .855, chr(97 + index), fontsize=9, weight='bold')
        handles, labels = axis.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .985), ncol=2,
                            frameon=False, handlelength=1.8, columnspacing=2.4, borderaxespad=0)
        legend.set_gid('legend')
        fig.text(.5, .115, 'Optimizer updates (thousands)', ha='center')
        fig.text(.5, .035, 'Training seed 17 · generation seed 0 · 2,000 samples per checkpoint',
                 ha='center', color='#404040', fontsize=7)
        raw = out / 'matplotlib_reference.svg'
        fig.savefig(raw, format='svg')
        fig.savefig(out / 'figure.png', dpi=300)
        fig.savefig(out / 'figure.pdf')
        normalized_svg(raw, out / 'figure.svg', group_panels=True)
        plt.close(fig)

    caption = ('Official development evaluations from matched EMA weights, 50-step DDIM, and guidance 2.5. '
               'Both U-Nets use training seed 17 and the same 1,197 exact crops. Each point represents one '
               '2,000-image evaluation on generation seed 0. Lines connect recorded checkpoints. '
               'The horizontal line shows the official 5% memorization limit.')
    (out / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    write_json(out / 'source_summary.json', {'protocol': 'EMA/DDIM50/CFG2.5, seed 0, 1000 per class',
               'manifest_sha256': manifest_hash, 'sources': sources, 'rows': rows})
    write_json(out / 'figure.spec.json', {'id': entry['id'], 'title': 'Checkpoint metrics',
               'kind': 'scientific-line-chart', 'units': 'points', 'width': 504, 'height': 187.2, 'font': 'Arial',
               'sources': sources, 'series': entry['series'], 'panels': [panel[0] for panel in panels],
               'caption': caption, 'status': 'development evidence'})
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', default='figures/registry.json')
    p.add_argument('--figure', help='Build just the figure with this ID')
    p.add_argument('--out', type=Path, help='Build one figure into a fresh output folder')
    p.add_argument('--from-source-summary', action='store_true',
                   help='Rebuild F4 or F5 from retained data and official JSON, with no private exports')
    args = p.parse_args()
    registry = json.loads(Path(args.registry).read_text())
    entries = [entry for entry in registry['figures'] if args.figure is None or entry['id'] == args.figure]
    if not entries:
        p.error(f'Unknown figure ID: {args.figure}')
    if (args.out or args.from_source_summary) and len(entries) != 1:
        p.error('Select one figure when supplying a new output or using a source summary')
    for entry in entries:
        original = Path(entry['output'])
        out = args.out or original
        summary_path = original / 'source_summary.json' if args.from_source_summary else None
        if summary_path and entry['kind'] not in ('checkpoint_metrics', 'sampling_sweep'):
            p.error('Only F4 and F5 support rebuilding from a source summary')
        if entry['kind'] == 'diagnostic_overfit':
            diagnostic_plot(entry | {'output': str(out)})
        elif entry['kind'] == 'checkpoint_metrics':
            checkpoint_metrics_plot(entry, out, summary_path)
        elif entry['kind'] == 'sampling_sweep':
            from src.figures.sampling_sweep import build
            build(Path(entry['source']).resolve(), out, summary_path)
        elif entry['kind'] == 'photo_grid':
            from src.figures.photo_grid import build
            replay = Path(entry['replay']) if 'replay' in entry else None
            build(Path(entry['result']), Path(entry['submission']), out, replay)
        else:
            raise ValueError(f'Unsupported figure kind: {entry["kind"]}')


if __name__ == '__main__':
    main()

"""Plot the complete official 500-image sampler screen for one checkpoint."""

import argparse
import csv
import itertools
import json
from pathlib import Path

from src.common import ROOT, sha256, write_json
from src.figures.build import normalized_svg, verify_evaluation_table


STEPS = (50, 100, 200)
GUIDANCE = (1.0, 1.5, 2.0, 2.5)
VARIANTS = ('ema', 'model')


def rows_from_receipts(source):
    from src.evaluation.select import validated_result

    rows, lineage_keys = [], set()
    for weights, steps, guidance in itertools.product(VARIANTS, STEPS, GUIDANCE):
        path = source / f'{weights}_s{steps}_g{guidance:g}' / 'seed_0.json'
        receipt = json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        command = receipt['command']
        exported = Path(command[command.index('--submission') + 1])
        result = validated_result(path, 0, exported, n_per_class=250)
        provenance = json.loads((exported / 'provenance.json').read_text(encoding='utf-8'))
        sampler = json.loads((exported / 'inference.json').read_text(encoding='utf-8'))['sampling']
        if (provenance['weights_variant'], sampler['steps'], sampler['guidance']) != (weights, steps, guidance):
            raise ValueError(f'Mismatched sampler receipt: {path}')
        lineage_keys.add(tuple(provenance[key] for key in ('training_update', 'training_seed',
                                                          'checkpoint_sha256', 'manifest_sha256')))
        keys = ('accuracy', 'fid', 'recall', 'memorization_rate', 'num_samples')
        rows.append({'weights': weights, 'steps': steps, 'guidance': guidance,
                     **{key: result['metrics'][key] for key in keys}, 'valid': result['marks']['valid'],
                     'source': path.relative_to(ROOT).as_posix(), 'source_sha256': sha256(path)})
    if len(lineage_keys) != 1:
        raise ValueError('Sampler plot requires one exact checkpoint and manifest')
    return rows, lineage_keys.pop()


def build(source, out, summary_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if summary_path is None:
        rows, (update, seed, checkpoint_hash, manifest_hash) = rows_from_receipts(source)
    else:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        update, seed = summary['update'], summary['training_seed']
        checkpoint_hash, manifest_hash = summary['checkpoint_sha256'], summary['manifest_sha256']
        rows = summary['rows']
        settings = [(row['weights'], row['steps'], row['guidance']) for row in rows]
        if settings != list(itertools.product(VARIANTS, STEPS, GUIDANCE)):
            raise ValueError('Retained sampler data must contain the complete screen in protocol order')
        verify_evaluation_table(rows)
    panels = [('accuracy', 'Accuracy (%)', (65, 100), [70, 80, 90, 100], 100),
              ('fid', 'FID', (70, 170), [80, 100, 120, 140, 160], 1),
              ('recall', 'Recall (%)', (20, 50), [20, 30, 40, 50], 100),
              ('memorization_rate', 'Memorization (%)', (-1, 60), [0, 20, 40, 60], 100)]
    out.mkdir(parents=True, exist_ok=False)
    style = {'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': .6, 'svg.fonttype': 'none',
             'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7, 4), facecolor='white')
        for index, (key, title, bounds, ticks, multiplier) in enumerate(panels):
            row, column = divmod(index, 2)
            left, bottom = .092 + .47 * column, .535 - .375 * row
            axis = fig.add_axes([left, bottom, .375, .275])
            axis.set_gid(f'n:{key}')
            if key == 'memorization_rate':
                axis.axhline(5, color='#858585', linewidth=.65, zorder=1)
                axis.text(1.01, 6.2, '5% limit', color='#575757', fontsize=7)
            for weights, color, name in [('ema', '#0072B2', 'EMA'), ('model', '#B34817', 'Raw')]:
                for steps, marker in zip(STEPS, ('o', 's', '|')):
                    series = [r for r in rows if r['weights'] == weights and r['steps'] == steps]
                    axis.plot(GUIDANCE, [multiplier * r[key] for r in series], color=color,
                              marker=marker, markersize=3.8, markeredgewidth=.8,
                              markerfacecolor=color if weights == 'model' else 'white',
                              linewidth=.8, label=f'{name} · {steps} steps', zorder=2)
            axis.set(xlim=(.92, 2.58), ylim=bounds, xticks=GUIDANCE, xticklabels=['1', '1.5', '2', '2.5'],
                     yticks=ticks, title=title)
            axis.tick_params(direction='out', length=2.5, width=.6, pad=2)
            fig.text(left - .056, bottom + .29, chr(97 + index), fontsize=9, weight='bold')
        handles, labels = axis.get_legend_handles_labels()
        legend_order = (0, 3, 1, 4, 2, 5)
        legend = fig.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
                            loc='upper center', bbox_to_anchor=(.5, .984), ncol=3,
                            frameon=False, handlelength=1.6, columnspacing=2.3, borderaxespad=0)
        legend.set_gid('legend')
        fig.text(.5, .058, 'Classifier-free guidance scale', ha='center')
        fig.text(.5, .024, f'{update:,} updates · training seed {seed} · generation seed 0 · 500 images per setting',
                 ha='center', color='#404040', fontsize=7)
        raw = out / 'matplotlib_reference.svg'
        fig.savefig(raw, format='svg')
        fig.savefig(out / 'figure.png', dpi=300)
        fig.savefig(out / 'figure.pdf')
        normalized_svg(raw, out / 'figure.svg', group_panels=True)
        plt.close(fig)
    with (out / 'source.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    caption = ('Official 500-image screening with seed 0 on one fixed full-width U-Net checkpoint. '
               'The 24 settings vary weight averaging, DDIM step count and classifier-free guidance. '
               'The gray line marks the 5% memorization limit. Small-sample FID is compared only within this screen.')
    (out / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    write_json(out / 'source_summary.json', {'checkpoint_sha256': checkpoint_hash, 'manifest_sha256': manifest_hash,
                                            'training_seed': seed, 'update': update, 'rows': rows})
    write_json(out / 'figure.spec.json', {'id': 'F5', 'title': 'Sampling tradeoffs', 'kind': 'scientific-line-chart',
               'units': 'points', 'width': 504, 'height': 288, 'font': 'Arial', 'caption': caption,
               'panels': [row[0] for row in panels], 'status': 'development screening evidence'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/experiments/optimization_v001_seed17/screening')
    parser.add_argument('--out', type=Path, default=ROOT / 'figures/F5_sampling_tradeoffs')
    args = parser.parse_args()
    build(args.source.resolve(), args.out)


if __name__ == '__main__':
    main()

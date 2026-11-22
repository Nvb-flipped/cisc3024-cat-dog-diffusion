"""Generate every number, table and figure of the report from the recorded result files.

    python -m scripts.build_report_assets

Writes report/numbers.tex (LaTeX macros and table bodies) and report/fig_*.pdf|png. Nothing is typed by hand: the report text only
uses these macros, so a number cannot drift from the evidence. The final model is configured in FINAL below.
"""
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path('report')
# ---- the submitted model -------------------------------------------------------------------------------------------------
FINAL = dict(
    label='v014, EMA 0.9995, 80k updates',
    official_dir=Path('results/final'),                                   # runs from submission/ itself (seeds 0/101/202/303/404)
    panel_dir=Path('results/experiments/v014_ema9995_flip/u080000_g275'),  # all further seeds of the same export
    official_seeds=(0, 101, 202, 303, 404),
    training_seconds=4661,   # receipt of checkpoints/v014_ema9995_flip_seed17/update_080000.json (two runs shared the GPU)
)
# ---- development progress (same evaluator; sampler differs where noted) -------------------------------------------------
PROGRESS = [
    ('v009', 1399, 'results/experiments/v009_linux_baseline', ['seed_0_cpu']),
    ('v010', 1929, 'results/experiments/v010_attn16/u048000_v009', ['seed_0']),
    ('v010 + mirror', 1929, 'results/experiments/v010_attn16_flip/u064000_g300', ['seed_0', 'seed_101', 'seed_202']),
    ('v011', 5536, 'results/experiments/v011_flip/u080000_g275', ['seed_0', 'seed_101', 'seed_202']),
    ('v012', 4610, 'results/experiments/v012_flip/u080000_g275', ['seed_0', 'seed_101', 'seed_202']),
    ('v013', 4922, 'results/experiments/v013_flip/u080000_g275', ['seed_0']),
    ('v014 (final)', 8029, None, ['seed_0', 'seed_101', 'seed_202']),
]
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def load(path):
    return json.loads(Path(path).read_text())


def metrics(path):
    d = load(path)
    return d['marks']['total'] if d['marks']['valid'] else 0, d['metrics'], d


def mean(values):
    return sum(values) / len(values)


def tex_row(cells):
    return ' & '.join(str(c) for c in cells) + r' \\'


def main():
    mac = {}
    # ---- official runs from submission/ -------------------------------------------------------------------------------------
    rows, official = [], []
    for s in FINAL['official_seeds']:
        marks, m, d = metrics(FINAL['official_dir'] / f'seed_{s}.json')
        official.append((s, marks, m, d))
        rows.append(tex_row([s, f"{m['accuracy']:.3f}", f"{m['fid']:.2f}", f"{m['kid'] * 1000:.1f}", f"{m['precision']:.3f}",
                             f"{m['recall']:.3f}", f"{m['memorization_rate'] * 100:.2f}\\%", marks]))
    mac['officialRows'] = '\n'.join(rows)
    res = official[0][3]['resources']
    ck = official[0][3]['checks']
    mac['chkDet'] = 'yes' if ck['deterministic'] else 'NO'
    mac['chkSeed'] = 'yes' if ck['seed_sensitive'] else 'NO'
    mac['chkMixed'], mac['chkSingle'] = f"{ck['mixed_batch_accuracy']:.2f}", f"{ck['single_class_batch_accuracy']:.2f}"
    mac['manifestRows'] = f"{official[0][3]['training_data']['stats']['valid_rows']:,}".replace(',', r'{,}')
    mac['params'] = f"{res['num_parameters']:,}".replace(',', r'{,}')
    mac['sizeMB'] = f"{res['submission_mb']:.1f}"
    mac['sampleSpeed'] = f"{st.median(o[3]['resources']['seconds_per_1000'] for o in official):.0f}"
    mac['trainHours'] = f"{FINAL['training_seconds'] / 3600:.1f}"
    # ---- all seeds of the same export (agent-chosen seeds plus the student's seed 31415) -------------------------------------
    panel = []
    for path in sorted(FINAL['panel_dir'].glob('seed_*.json')):
        if 'provenance' in path.name:
            continue
        marks, m, _ = metrics(path)
        panel.append((int(path.stem.split('_')[1]), marks, m))
    n = len(panel)
    n15 = sum(p[1] == 15 for p in panel)
    rec = [p[2]['recall'] for p in panel]
    mac.update(nSeeds=n, nFifteen=n15, nBelow=n - n15)
    for key, vals, fmt in (('recall', rec, '.3f'), ('fid', [p[2]['fid'] for p in panel], '.2f'),
                           ('acc', [p[2]['accuracy'] for p in panel], '.3f'), ('prec', [p[2]['precision'] for p in panel], '.3f')):
        mac[f'{key}Mean'] = format(mean(vals), fmt)
        mac[f'{key}Min'] = format(min(vals), fmt)
        mac[f'{key}Max'] = format(max(vals), fmt)
    mac['recallSD'] = f'{st.stdev(rec):.3f}'
    mac['memMax'] = f"{max(p[2]['memorization_rate'] for p in panel) * 100:.2f}"
    low = min(panel, key=lambda p: p[2]['recall'])
    mac['lowSeed'], mac['lowRecall'] = low[0], f"{low[2]['recall']:.4f}"
    z = (low[2]['recall'] - mean(rec)) / st.stdev(rec)
    mac['lowZ'] = f'{abs(z):.1f}'

    def wilson(k, n_, zz=1.96):
        p = k / n_
        d = 1 + zz * zz / n_
        c = p + zz * zz / (2 * n_)
        h = zz * math.sqrt(p * (1 - p) / n_ + zz * zz / (4 * n_ * n_))
        return (c - h) / d, (c + h) / d
    lo, hi = wilson(n15, n)
    mac['pFifteen'] = f'{100 * n15 / n:.1f}'
    mac['pFifteenLo'], mac['pFifteenHi'] = f'{100 * lo:.1f}', f'{100 * hi:.1f}'
    # ---- wide 10.7M model vs the final model on the same 15 seeds (replacement rule, decision log 75/78) ----------------------
    seeds15 = (0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414)
    def same15(folder):
        out = [metrics(Path(folder) / f'seed_{x}.json') for x in seeds15]
        return sum(o[0] == 15 for o in out), mean([o[1]['recall'] for o in out]), mean([o[1]['precision'] for o in out]), mean([o[1]['accuracy'] for o in out])
    fin = same15(FINAL['panel_dir'])
    wide = same15('results/experiments/v014_wide_ema9995_flip/u080000_g275')
    mac.update(wideAtFifteen=wide[0], wideRecall=f'{wide[1]:.3f}', widePrec=f'{wide[2]:.3f}', wideAcc=f'{wide[3]:.3f}',
               smallAtFifteen=fin[0], smallRecall=f'{fin[1]:.3f}', smallPrec=f'{fin[2]:.3f}', smallAcc=f'{fin[3]:.3f}')
    # ---- development progress table and figure -------------------------------------------------------------------------------
    prog_rows, pts = [], []
    for name, images, folder, seeds in PROGRESS:
        folder = Path(folder) if folder else FINAL['panel_dir']
        got = [metrics(folder / f'{s}.json') for s in seeds]
        marks = ' / '.join(str(g[0]) for g in got) + ''.join(' / --' for _ in range(3 - len(got)))
        ms = [g[1] for g in got]
        mem = mean([m['memorization_rate'] for m in ms]) * 100
        prog_rows.append(tex_row([name, f'{images:,}'.replace(',', r'{,}'), marks, f"{mean([m['accuracy'] for m in ms]):.3f}",
                                  f"{mean([m['fid'] for m in ms]):.1f}", f"{mean([m['recall'] for m in ms]):.3f}",
                                  f"{mean([m['precision'] for m in ms]):.3f}", f'{mem:.2f}\\%']))
        pts.append((name, images, mean([m['fid'] for m in ms]), mean([m['recall'] for m in ms])))
    mac['progressRows'] = '\n'.join(prog_rows)
    # ---- audit ---------------------------------------------------------------------------------------------------------------
    a = load('results/audit/breakdown.json')
    pc = lambda s: f"{100 * s['precision']:.1f}"                                       # noqa: E731
    ci = lambda s: f"{100 * s['wilson_95'][0]:.1f}--{100 * s['wilson_95'][1]:.1f}"      # noqa: E731
    mac.update(auditGood=a['overall']['good'], auditPct=pc(a['overall']), auditCI=ci(a['overall']),
               auditCat=f"{a['by_class']['cat']['good']}/{a['by_class']['cat']['n']}", auditCatPct=pc(a['by_class']['cat']),
               auditDog=f"{a['by_class']['dog']['good']}/{a['by_class']['dog']['n']}", auditDogPct=pc(a['by_class']['dog']),
               auditPersonal=f"{a['by_source']['personal']['good']}/{a['by_source']['personal']['n']}",
               auditAI=f"{a['by_source']['ai_found']['good']}/{a['by_source']['ai_found']['n']}",
               auditEars=a['bad_reasons']['ears cut'], auditBad=sum(a['bad_reasons'].values()),
               auditRecrop=f"{a['ears_cut_in_ai_recrop_rows']}/{a['ai_recrop_rows']['n']}")
    # ---- write the macros ----------------------------------------------------------------------------------------------------
    lines = ['% GENERATED by scripts/build_report_assets.py from results/*.json - do not edit']
    for k, v in mac.items():
        lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    (OUT / 'numbers.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # ---- figure 1: data growth vs FID and recall (two panels = two measures; one axis each) -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    offsets = {  # (dx, dy, ha) per panel; chosen so labels do not collide
        3: {'v009': (5, 4, 'left'), 'v010': (5, 4, 'left'), 'v010 + mirror': (5, 4, 'left'), 'v011': (5, 5, 'left'),
            'v012': (5, 4, 'left'), 'v013': (5, 4, 'left'), 'v014 (final)': (-6, 6, 'right')},
        2: {'v009': (5, -3, 'left'), 'v010': (5, -3, 'left'), 'v010 + mirror': (5, 4, 'left'), 'v011': (6, -3, 'left'),
            'v012': (-5, -10, 'right'), 'v013': (5, -1, 'left'), 'v014 (final)': (-5, 7, 'right')},
    }
    for ax, idx, label, line, text, tx, tha, tva in (
            (axes[0], 3, 'Recall (mean over listed seeds)', 0.40, 'Excellent: recall $\\geq$ 0.40', 0.02, 'left', 'bottom'),
            (axes[1], 2, 'FID (mean over listed seeds)', 50, 'Excellent: FID $\\leq$ 50', 0.98, 'right', 'bottom')):
        ax.axhline(line, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.text(tx, line, text, transform=ax.get_yaxis_transform(), ha=tha, va=tva, color=MUTED, fontsize=7)
        for p in pts:
            final = p[0].endswith('(final)')
            ax.plot(p[1], p[idx], 'o', ms=6 if final else 4.5, color=BLUE, mec='white', mew=0.8, zorder=3)
            dx, dy, ha = offsets[idx][p[0]]
            ax.annotate(p[0].replace(' (final)', ' final'), (p[1], p[idx]), xytext=(dx, dy), textcoords='offset points', ha=ha,
                        fontsize=6.5, color=INK, fontweight='bold' if final else 'normal')
        ax.set_xlabel('Training images in the manifest')
        ax.set_ylabel(label)
        ax.set_xlim(900, 9200)
        ax.grid(axis='y', color='#e7e6e2', lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0.30, 0.46)
    axes[1].set_ylim(44, 63)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / 'fig_progress.pdf')
    plt.close(fig)
    # ---- figure 2: recall of every evaluated seed -----------------------------------------------------------------------------
    order = sorted(panel, key=lambda p: p[2]['recall'])
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.axhline(0.40, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(len(order) * 0.99, 0.4008, 'Excellent: recall $\\geq$ 0.40', color=MUTED, fontsize=6.5, va='bottom', ha='right')
    ok = [(i, p[2]['recall']) for i, p in enumerate(order) if p[1] == 15]
    bad = [(i, p[2]['recall'], p[0]) for i, p in enumerate(order) if p[1] != 15]
    ax.plot([i for i, _ in ok], [r for _, r in ok], 'o', ms=3, color=BLUE, mec='white', mew=0.4, label=f'15/15 marks ({len(ok)} seeds)')
    ax.plot([i for i, _, _ in bad], [r for _, r, _ in bad], 'D', ms=5.5, color=ORANGE, mec='white', mew=0.6, label=f'14/15 marks ({len(bad)} seed)')
    for i, r, s in bad:
        ax.annotate(f'seed {s}: {r:.3f}', (i, r), xytext=(10, -9), textcoords='offset points', fontsize=6.5, color=INK, ha='left')
    ax.set_xlabel(f'Seeds, sorted by recall (n = {len(order)})')
    ax.set_ylabel('Recall')
    ax.set_ylim(0.385, 0.495)
    ax.grid(axis='y', color='#e7e6e2', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=6.5, frameon=False, handletextpad=0.3)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / 'fig_seeds.pdf')
    plt.close(fig)
    # ---- figure 3: audit result (precision by group with Wilson 95% intervals; error types) -----------------------------------
    groups = [('All 200', a['overall']), ('Cats', a['by_class']['cat']), ('Dogs', a['by_class']['dog']),
              ('My reviewed rows', a['by_source']['personal']), ('AI-found rows', a['by_source']['ai_found'])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw={'width_ratios': [1.25, 1]})
    for i, (name, g) in enumerate(groups):
        y = len(groups) - 1 - i
        lo_, hi_ = g['wilson_95']
        ax1.plot([100 * lo_, 100 * hi_], [y, y], color=MUTED, lw=1.2, solid_capstyle='butt', zorder=2)
        ax1.plot(100 * g['precision'], y, 'o', ms=6, color=BLUE, mec='white', mew=0.8, zorder=3)
        ax1.text(100.8, y, f"{g['good']}/{g['n']} = {100 * g['precision']:.1f}%", va='center', fontsize=6.5, color=INK)
    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels([n for n, _ in groups][::-1], fontsize=7)
    ax1.set_xlim(50, 118)
    ax1.set_xticks([50, 60, 70, 80, 90, 100])
    ax1.set_xlabel('Share of crops judged good (%), with 95% Wilson interval')
    ax1.grid(axis='x', color='#e7e6e2', lw=0.5)
    ax1.set_axisbelow(True)
    reasons = sorted(a['bad_reasons'].items(), key=lambda kv: kv[1])
    ax2.barh([k.capitalize() for k, _ in reasons], [v for _, v in reasons], color=BLUE, height=0.55)
    for i, (_, v) in enumerate(reasons):
        ax2.text(v + 0.6, i, str(v), va='center', fontsize=7, color=INK)
    ax2.set_xlim(0, 40)
    ax2.set_xlabel(f"Bad crops by reason (n = {sum(a['bad_reasons'].values())})")
    ax2.tick_params(axis='y', labelsize=7)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / 'fig_audit.pdf')
    plt.close(fig)
    # ---- sample grid: the official evaluator's own grid of the exact submission, seed 0 ---------------------------------------
    grid = Image.open(FINAL['official_dir'] / 'seed_0.png').convert('RGB')
    grid.resize((grid.width * 4, grid.height * 4), Image.NEAREST).save(OUT / 'fig_grid_seed0.png')
    print(f'{n} seeds, {n15} at 15; wrote {OUT / "numbers.tex"}, fig_progress.pdf, fig_seeds.pdf, fig_grid_seed0.png')


if __name__ == '__main__':
    main()

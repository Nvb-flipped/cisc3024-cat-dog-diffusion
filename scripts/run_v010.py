"""v010: all 50 personally reviewed batches, fresh training, official checkpoint screens.

Phases (run from the project root with the training venv):
  freeze                      freeze manifests/v010.csv from the personal ledgers, keeping v009's exact
                              training rows and byte-identical 63-row holdout, then verify cache parity
  diagnostic VARIANT          fixed 2,000-update single-batch overfit check for the variant's options
  train VARIANT               fresh seed-17 training (resumes from the newest verified checkpoint)
  screen VARIANT --updates U  export each EMA checkpoint with a sampler and run official seed 0
  confirm VARIANT --updates U --seeds 101 202 [303 404]
                              further official seeds for one frozen checkpoint/sampler

Every variant uses only exact manifest crops. Variants marked flip mirror whole crops left-right
with probability 0.5 during training; the official evaluator already counts mirror images as
copies in its memorization test (starter_kit/evaluate.py, "horizontal flips are a common
augmentation").
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.diagnostics import overfit
from src.training.optimize import evaluate

PARENT = Path('manifests/v009.csv')
AMENDMENTS = Path('results/plans/v010_input_amendments.json')
PREVIEW_RECEIPT = Path('results/curation/preview_portability_20261119.json')
MANIFEST = Path('manifests/v010.csv')
HOLDOUT = MANIFEST.with_suffix('.validation.csv')
BASE_CONFIG = Path('configs/runs/v009_attention16_32k.json')
# Exact sampler of the selected v009 34k development model.
SAMPLERS = {
    'v009': dict(weights='ema', steps=50, guidance=3.25, pag_scale=0.15, score_sharpness=1.03,
                 score_sigma=0.5, class_noise_max=0.1, freeu=True, freeu_strength=2.0),
}
VARIANTS = {
    # Matched v009 recipe on the larger reviewed set (data effect only).
    'v010_attn16': dict(changes={}, updates=48000, checkpoint_every=1000),
    # Same recipe plus whole-crop left-right mirroring of exact manifest crops.
    'v010_attn16_flip': dict(changes={'training': {'horizontal_flip': 0.5}}, updates=80000,
                             checkpoint_every=1000),
    # Wider U-Net (the earlier 10.7M baseline width) with mirroring.
    'v010_wide_flip': dict(changes={'model': {'channels': [48, 96, 144, 192]},
                                    'training': {'horizontal_flip': 0.5}},
                           updates=80000, checkpoint_every=1000),
}
TRAINING_SOURCES = ('src/training/train.py', 'src/training/step.py', 'src/training/data.py',
                    'src/training/validation.py', 'src/training/lineage.py', 'submission/model.py',
                    'submission/diffusion.py', 'submission/generate.py', 'src/release/export.py',
                    'src/evaluation/run.py', 'src/training/diagnostics.py')


def manifest_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return {tuple(row.values()) for row in csv.DictReader(stream)}


def freeze_v010():
    from src.curation.freeze import freeze
    if not MANIFEST.exists():
        print(freeze(read_config('configs/curation.yaml'), MANIFEST, preserve_split_from=PARENT,
                     preview_receipt=PREVIEW_RECEIPT), flush=True)
    if not manifest_rows(PARENT) < manifest_rows(MANIFEST):
        raise ValueError('v010 must contain every exact v009 training crop')
    if sha256(HOLDOUT) != sha256(PARENT.with_suffix('.validation.csv')):
        raise ValueError('v010 must keep the byte-identical v009 holdout')
    parity = Path('results/curation/v010_cache_parity.json')
    if not parity.exists():
        from src.training.data import load_manifest_data
        for path in (MANIFEST, HOLDOUT):
            load_manifest_data(path)  # builds the cache via the official rebuild
        subprocess.run([sys.executable, '-m', 'scripts.verify_manifest_cache', str(MANIFEST), str(HOLDOUT),
                        '--out', str(parity)], check=True)
    checks = read(parity)['checks']
    if not all(row['pixels_equal'] and row['labels_equal'] for row in checks):
        raise ValueError('Training cache differs from the official reconstruction')
    print({'manifest': str(MANIFEST), 'sha256': sha256(MANIFEST), 'training_rows': len(manifest_rows(MANIFEST)),
           'holdout_rows': len(manifest_rows(HOLDOUT)), 'parity': str(parity)}, flush=True)


def variant_config(name):
    settings = read(BASE_CONFIG)
    variant = VARIANTS[name]
    for section, values in variant['changes'].items():
        settings[section] = settings[section] | values
    settings['training'] = settings['training'] | {'updates': variant['updates'],
                                                   'checkpoint_every': variant['checkpoint_every']}
    config = Path('configs/runs') / f'{name}.json'
    if not config.exists():
        write_json(config, settings)
    if read(config) != settings:
        raise ValueError(f'Frozen variant config changed: {config}')
    return config, settings


def diagnostic_path(name):
    return Path('results/diagnostics') / name / 'result.json'


def run_diagnostic(name, device):
    _, settings = variant_config(name)
    out = diagnostic_path(name).parent
    if not diagnostic_path(name).exists():
        overfit(out=out, device=device, encoder_attention16=bool(settings['model'].get('encoder_attention16')),
                horizontal_flip=settings['training'].get('horizontal_flip', 0))
    result = read(diagnostic_path(name))
    print({'variant': name, 'passed': result['passed'], 'reduction': result['reduction']}, flush=True)
    if not result['passed']:
        raise ValueError('Fixed 2k diagnostic failed; investigate before training')


def plan_for(name):
    config, settings = variant_config(name)
    run = Path('checkpoints') / f'{name}_seed17'
    command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config),
               '--manifest', str(MANIFEST), '--validation-manifest', str(HOLDOUT), '--seed', '17',
               '--updates', str(settings['training']['updates']), '--out', str(run)]
    plan = Path('results/plans') / f'{name}.json'
    inputs = {str(p): sha256(p) for p in (Path(__file__).relative_to(ROOT), MANIFEST, HOLDOUT,
              MANIFEST.with_suffix('.provenance.json'), Path('results/curation/v010_cache_parity.json'),
              PREVIEW_RECEIPT,
              BASE_CONFIG, config, diagnostic_path(name), *map(Path, TRAINING_SOURCES))}
    spec = dict(inputs=inputs, command=[Path(command[0]).name, *command[1:]], variant=VARIANTS[name],
                samplers=SAMPLERS, seeds={'selection': [0, 101, 202], 'confirmation': [303, 404]},
                n_per_class=1000,
                rule='Screen EMA checkpoints on official seed 0 with the v009 sampler; reject any hard-check '
                     'violation; prefer the highest tier total, then the larger minimum normalized margin, '
                     'then lower memorization. Freeze the choice before seeds 303/404.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    frozen = read(plan)
    frozen.pop('created', None)
    # Recorded amendments: a frozen input may carry a newer hash only if the exact old->new pair
    # and its reason are listed (changes that cannot affect these runs, e.g. a new manifest origin).
    if AMENDMENTS.exists():
        for path, change in read(AMENDMENTS)['inputs'].items():
            if frozen['inputs'].get(path) == change['old'] and spec['inputs'].get(path) == change['new']:
                spec['inputs'][path] = change['old']
    if frozen != spec:
        raise ValueError(f'Frozen plan inputs changed: {plan}')
    return plan, run, command, settings


def train(name):
    if not read(diagnostic_path(name))['passed']:
        raise ValueError('Run the passing diagnostic first')
    plan, run, command, settings = plan_for(name)
    verify_starter()
    digest = sha256(MANIFEST)
    if not (run / 'completed.json').exists() or read(run / 'completed.json')['last_update'] < settings['training']['updates']:
        # A shared GPU can kill the job between saving a checkpoint and its receipt. Set aside
        # any checkpoint without a receipt (never delete) so the run resumes from the newest
        # verified one; small checkpoint intervals bound the loss to ~1k updates.
        for orphan in run.glob('update_*.pt'):
            if not orphan.with_suffix('.json').exists():
                quarantine = run / 'orphaned'
                quarantine.mkdir(exist_ok=True)
                orphan.replace(quarantine / f'{orphan.name}.{timestamp().replace(":", "")}')
        for partial in run.glob('*.tmp'):
            partial.unlink()  # incomplete atomic write; the previous checkpoint is intact
        saved = sorted(run.glob('update_*.json'))
        if not saved and (run / 'run.json').exists():
            # Interrupted before the first checkpoint: keep its metadata aside and start fresh.
            quarantine = run / 'orphaned' / timestamp().replace(':', '')
            quarantine.mkdir(parents=True)
            for leftover in [*run.glob('*.json'), *run.glob('*.csv')]:
                leftover.replace(quarantine / leftover.name)
        if saved:
            command += ['--resume', str(verified_checkpoint(run, read(saved[-1])['update'], digest))]
        subprocess.run(command, check=True)
    print({'completed': read(run / 'completed.json'), 'plan_sha256': sha256(plan)}, flush=True)


def export_for(name, run, update, sampler_name):
    checkpoint = verified_checkpoint(run, update, sha256(MANIFEST))
    export = Path('private_build') / name / f'u{update:06d}_{sampler_name}'
    if not export.exists():
        export_checkpoint(checkpoint, MANIFEST, export, **SAMPLERS[sampler_name])
    out = Path('results/experiments') / name / f'u{update:06d}_{sampler_name}'
    if not (out / 'interface.json').exists():
        write_json(out / 'interface.json', verify_interface(export, export) | {'solver': 'ddim'})
    return checkpoint, export, out


def run_seeds(name, updates, sampler_name, seeds):
    plan, run, _, _ = plan_for(name)
    runtime = {k: SAMPLERS[sampler_name][k] for k in ('weights', 'steps', 'guidance')}
    rows = []
    for update in updates:
        checkpoint, export, out = export_for(name, run, update, sampler_name)
        result = evaluate(checkpoint, MANIFEST, export, out, seeds=seeds, **runtime)
        write_json(out / f'summary_{"_".join(map(str, seeds))}.json',
                   dict(plan_sha256=sha256(plan), sampler=sampler_name, candidate=result))
        for row in result['evidence']:
            m = row['metrics']
            rows.append(dict(update=update, seed=row['seed'], marks=row['marks']['total'],
                             valid=row['marks']['valid'], accuracy=round(m['accuracy'], 4),
                             fid=round(m['fid'], 2), recall=round(m['recall'], 4),
                             precision=round(m['precision'], 4), memorization=m['memorization_rate']))
            print(rows[-1], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'diagnostic', 'train', 'screen', 'confirm'))
    parser.add_argument('variant', nargs='?', choices=sorted(VARIANTS))
    parser.add_argument('--updates', type=int, nargs='+')
    parser.add_argument('--sampler', default='v009', choices=sorted(SAMPLERS))
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--device', default='cuda', help='Diagnostic device (cpu keeps the GPU free)')
    args = parser.parse_args()
    if args.phase == 'freeze':
        return freeze_v010()
    if args.variant is None:
        parser.error('variant required')
    if args.phase == 'diagnostic':
        return run_diagnostic(args.variant, args.device)
    if args.phase == 'train':
        return train(args.variant)
    seeds = [0] if args.phase == 'screen' else args.seeds
    if not args.updates or not seeds:
        parser.error('--updates (and --seeds for confirm) required')
    run_seeds(args.variant, args.updates, args.sampler, seeds)


if __name__ == '__main__':
    main()

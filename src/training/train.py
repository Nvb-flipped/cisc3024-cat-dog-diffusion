"""Reproducible AdamW, BF16, gradient accumulation and EMA training."""
import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.common import ROOT, environment, read_config, sha256, timestamp, write_json
from src.training.data import load_manifest_data
from src.training.step import optimizer_step, update_ema
from src.training.validation import require_disjoint_sources, validation_loss
from src.training.lineage import reviewed_manifest_provenance
from submission.diffusion import cosine_alpha_bar
from submission.model import ConditionalUNet


def atomic_checkpoint(path, payload):
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists; preserve the original training lineage: {path}')
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume_destination(out, start_update, final_update):
    if final_update <= start_update:
        raise ValueError('Requested update budget must exceed the resumed checkpoint update')
    later = [path for path in out.glob('update_*.pt')
             if path.stem.removeprefix('update_').isdigit()
             and int(path.stem.removeprefix('update_')) > start_update]
    if later:
        raise ValueError('Run contains checkpoints after the resume point; choose a fresh --out for this branch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/ddpm.yaml')
    p.add_argument('--manifest', required=True)
    p.add_argument('--validation-manifest', type=Path, help='Disjoint pool holdout; defaults to the matching .validation.csv when present')
    p.add_argument('--pool', default='data/student')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--resume', type=Path)
    p.add_argument('--updates', type=int, help='Explicit budget override, recorded in run metadata')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    config = read_config(args.config)
    review_provenance_hash = reviewed_manifest_provenance(args.manifest)
    if args.updates is not None:
        config['training']['updates'] = args.updates
    settings = config['training']
    if settings['updates'] < 1 or settings['checkpoint_every'] < 1 or settings['log_every'] < 1:
        raise ValueError('Update, checkpoint and log intervals must be positive')
    torch.set_num_threads(settings.get('threads', 8))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; select --device cpu explicitly for diagnostics')
    images, labels, manifest_hash, stats = load_manifest_data(args.manifest, args.pool)
    if min(stats['cat'], stats['dog']) < 500:
        raise ValueError('Baseline requires at least 500 reviewed training heads per class')
    # Small 32x32 datasets fit on the local GPU; avoids multiprocessing/Windows loader overhead.
    images, labels = images.to(device), labels.to(device)
    validation_manifest = args.validation_manifest or Path(args.manifest).with_suffix('.validation.csv')
    validation = None
    if validation_manifest.exists():
        require_disjoint_sources(args.manifest, validation_manifest)
        held_images, held_labels, held_hash, held_stats = load_manifest_data(validation_manifest, args.pool)
        if held_stats['pool_content_sha256'] != stats['pool_content_sha256']:
            raise ValueError('Source pool changed while loading the holdout')
        validation = (held_images.to(device), held_labels.to(device))
    elif args.validation_manifest:
        raise FileNotFoundError(f'Requested validation manifest is missing: {validation_manifest}')
    model = ConditionalUNet(**config['model']).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 18_000_000:
        raise ValueError(f'{parameters} parameters exceeds engineering cap; freeze a reduced config first')
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 betas=tuple(settings['betas']), weight_decay=settings['weight_decay'])
    alpha_bar = cosine_alpha_bar(**config['diffusion']).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    effective, micro = settings['effective_batch'], settings['microbatch']
    if effective < 1 or micro < 1 or effective % micro:
        raise ValueError('effective_batch must be a positive multiple of microbatch')
    bf16 = settings['precision'] == 'bf16' and device.type == 'cuda'
    if bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 requested but not supported')
    if settings['precision'] not in ('bf16', 'fp32'):
        raise ValueError('Supported training precisions: bf16, fp32')
    if device.type == 'cpu' and settings['precision'] != 'fp32':
        raise ValueError('CPU training diagnostics must explicitly select fp32')
    out = args.out or Path('checkpoints') / f'ddpm_seed{args.seed}_{manifest_hash[:10]}'
    out.mkdir(parents=True, exist_ok=True)
    start_update = 0
    parent_hash = None
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state['manifest_sha256'] != manifest_hash:
            raise ValueError('Resume manifest mismatch: changed training rows require a fresh final run')
        if state.get('data_review_provenance_sha256') != review_provenance_hash:
            raise ValueError('Resume changes the frozen personal-review provenance')
        if state['stats']['pool_content_sha256'] != stats['pool_content_sha256']:
            raise ValueError('Source pool content changed since checkpoint')
        if state['seed'] != args.seed or state['config']['model'] != config['model']:
            raise ValueError('Resume seed/architecture mismatch')
        if state.get('validation_manifest_sha256') != (held_hash if validation else None):
            raise ValueError('Resume changes the held-out pool validation manifest')
        previous_settings = dict(state['config']['training'])
        new_settings = dict(settings)
        for mutable in ('updates', 'checkpoint_every', 'log_every'):
            previous_settings.pop(mutable, None)
            new_settings.pop(mutable, None)
        if previous_settings != new_settings or state['config']['diffusion'] != config['diffusion']:
            raise ValueError('Resume changes optimizer/objective settings')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator_state'].cpu())
        torch.set_rng_state(state['cpu_rng_state'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(), device)
        start_update = state['update']
        parent_hash = sha256(args.resume)
    elif (out / 'run.json').exists():
        raise ValueError(f'Run exists: {out}; use --resume or a fresh --out')
    validate_resume_destination(out, start_update, settings['updates'])
    started = timestamp()
    segment_id = f'{start_update:06d}_' + ''.join(c for c in started if c.isalnum())
    code_hashes = {name: sha256(ROOT / name) for name in ['src/training/train.py',
        'src/training/step.py', 'src/training/data.py', 'src/training/validation.py',
        'src/training/lineage.py', 'src/curation/pool_fingerprint.py',
        'submission/model.py', 'submission/diffusion.py']}
    metadata = {'config': config, 'config_sha256': sha256(args.config),
                'manifest': str(Path(args.manifest).resolve()), 'manifest_sha256': manifest_hash,
                'data_review_provenance_sha256': review_provenance_hash,
                'stats': stats, 'seed': args.seed, 'parameters': parameters,
                'environment': environment(), 'started': started, 'parent_checkpoint_sha256': parent_hash,
                'initial_update': start_update, 'precision_used': 'bf16' if bf16 else 'fp32',
                'segment_id': segment_id, 'source_code_sha256': code_hashes,
                'validation': {'manifest': str(validation_manifest.resolve()), 'sha256': held_hash,
                               'stats': held_stats, 'seed': 3024, 'timesteps': [50, 250, 500, 750],
                               'microbatch': 32} if validation else None,
                'command_args': vars(args) | {'out': str(out), 'resume': str(args.resume) if args.resume else None}}
    metadata['command_args'] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata['command_args'].items()}
    write_json(out / f'segment_{segment_id}.json', metadata)
    write_json(out / 'run.json', metadata)
    begin = time.perf_counter()
    model.train()
    log_path = out / f'loss_{segment_id}.csv'
    with log_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['update', 'loss', 'grad_norm', 'lr', 'elapsed_seconds'])
        if stream.tell() == 0:
            writer.writeheader()
        for update in range(start_update + 1, settings['updates'] + 1):
            loss, norm, lr = optimizer_step(model, optimizer, images, labels, alpha_bar,
                                            generator, settings, update)
            update_ema(ema, model, settings['ema_decay'])
            if update % settings['log_every'] == 0 or update == 1:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                row = dict(update=update, loss=loss.item(), grad_norm=norm.item(),
                           lr=lr, elapsed_seconds=time.perf_counter() - begin)
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
            if update % settings['checkpoint_every'] == 0 or update == settings['updates']:
                path = out / f'update_{update:06d}.pt'
                atomic_checkpoint(path, {'model': model.state_dict(), 'ema': ema.state_dict(),
                     'optimizer': optimizer.state_dict(), 'config': config, 'update': update,
                     'seed': args.seed, 'manifest_sha256': manifest_hash, 'stats': stats,
                     'data_review_provenance_sha256': review_provenance_hash,
                     'source_code_sha256': code_hashes, 'segment_id': segment_id,
                     'validation_manifest_sha256': held_hash if validation else None,
                     'generator_state': generator.get_state(), 'cpu_rng_state': torch.get_rng_state(),
                     'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None})
                write_json(path.with_suffix('.json'), {'sha256': sha256(path), 'update': update,
                           'manifest_sha256': manifest_hash, 'elapsed_seconds': time.perf_counter() - begin})
                if validation:
                    evaluation = {name: validation_loss(network, *validation, alpha_bar)
                                  for name, network in [('raw', model), ('ema', ema)]}
                    write_json(out / f'validation_{update:06d}.json', {'update': update,
                        'checkpoint_sha256': sha256(path), 'manifest_sha256': held_hash,
                        'training_manifest_sha256': manifest_hash, 'loss': evaluation})
    completion = {'ended': timestamp(), 'last_update': settings['updates'], 'segment_id': segment_id,
                  'segment_seconds': time.perf_counter() - begin, 'manifest_sha256': manifest_hash}
    write_json(out / f'completed_{segment_id}.json', completion)
    write_json(out / 'completed.json', completion)


if __name__ == '__main__':
    main()

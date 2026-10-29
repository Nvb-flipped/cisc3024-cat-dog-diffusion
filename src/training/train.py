"""Reproducible AdamW, BF16, gradient accumulation and EMA training."""
import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.common import environment, read_config, sha256, timestamp, write_json
from src.training.data import load_manifest_data
from submission.diffusion import add_noise, cosine_alpha_bar
from submission.model import ConditionalUNet


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/ddpm.yaml')
    p.add_argument('--manifest', required=True)
    p.add_argument('--pool', default='data/student')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--resume', type=Path)
    p.add_argument('--updates', type=int, help='Explicit budget override, recorded in run metadata')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    config = read_config(args.config)
    if args.updates is not None:
        config['training']['updates'] = args.updates
    settings = config['training']
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
    accumulation = effective // micro
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
        if state['seed'] != args.seed or state['config']['model'] != config['model']:
            raise ValueError('Resume seed/architecture mismatch')
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
    started = timestamp()
    metadata = {'config': config, 'config_sha256': sha256(args.config),
                'manifest': str(Path(args.manifest).resolve()), 'manifest_sha256': manifest_hash,
                'stats': stats, 'seed': args.seed, 'parameters': parameters,
                'environment': environment(), 'started': started, 'parent_checkpoint_sha256': parent_hash,
                'initial_update': start_update, 'precision_used': 'bf16' if bf16 else 'fp32',
                'command_args': vars(args) | {'out': str(out), 'resume': str(args.resume) if args.resume else None}}
    metadata['command_args'] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata['command_args'].items()}
    write_json(out / f'segment_{start_update:06d}.json', metadata)
    write_json(out / 'run.json', metadata)
    begin = time.perf_counter()
    model.train()
    log_path = out / 'loss.csv'
    with log_path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['update', 'loss', 'grad_norm', 'lr', 'elapsed_seconds'])
        if stream.tell() == 0:
            writer.writeheader()
        for update in range(start_update + 1, settings['updates'] + 1):
            lr = settings['learning_rate'] * min(update / max(settings['warmup'], 1), 1)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(accumulation):
                indices = torch.randint(len(labels), (micro,), generator=generator, device=device)
                x0, y = images[indices], labels[indices].clone()
                t = torch.randint(len(alpha_bar), (micro,), generator=generator, device=device)
                noise = torch.randn(x0.shape, generator=generator, device=device)
                y[torch.rand(micro, generator=generator, device=device) < settings['null_probability']] = 2
                xt = add_noise(x0, noise, t, alpha_bar)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    predicted = model(xt, t, y)
                    loss = F.mse_loss(predicted.float(), noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at update {update}')
                (loss / accumulation).backward()
                losses.append(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for target, value in zip(ema.parameters(), model.parameters()):
                    target.lerp_(value, 1 - settings['ema_decay'])
            if update % settings['log_every'] == 0 or update == 1:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                row = dict(update=update, loss=torch.stack(losses).mean().item(), grad_norm=norm.item(),
                           lr=lr, elapsed_seconds=time.perf_counter() - begin)
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
            if update % settings['checkpoint_every'] == 0 or update == settings['updates']:
                path = out / f'update_{update:06d}.pt'
                atomic_checkpoint(path, {'model': model.state_dict(), 'ema': ema.state_dict(),
                     'optimizer': optimizer.state_dict(), 'config': config, 'update': update,
                     'seed': args.seed, 'manifest_sha256': manifest_hash, 'stats': stats,
                     'generator_state': generator.get_state(), 'cpu_rng_state': torch.get_rng_state(),
                     'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None})
                write_json(path.with_suffix('.json'), {'sha256': sha256(path), 'update': update,
                           'manifest_sha256': manifest_hash, 'elapsed_seconds': time.perf_counter() - begin})
    write_json(out / 'completed.json', {'ended': timestamp(), 'last_update': settings['updates'],
               'segment_seconds': time.perf_counter() - begin, 'manifest_sha256': manifest_hash})


if __name__ == '__main__':
    main()

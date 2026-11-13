"""Fixed-budget pool-only overfit test and 500-update main-architecture benchmark."""
import argparse
import copy
import csv
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from src.common import environment, read_config, sha256, timestamp, write_json
from src.training.data import load_manifest_data
from src.training.step import optimizer_step, update_ema, validate_loss_weighting
from submission.diffusion import (add_noise, cosine_alpha_bar, reconstruct_clean,
                                  reconstruct_clean_from_velocity, velocity_target)
from submission.model import ConditionalUNet

OVERFIT_UPDATES = 2000
BENCHMARK_UPDATES = 500


def overfit(manifest='manifests/diagnostic_two_heads.csv', device='cuda', out='results/diagnostics/overfit',
            *, prediction_type='epsilon', min_snr_gamma=None, patch_size=1, patch_detail=False):
    if prediction_type not in ('epsilon', 'velocity'):
        raise ValueError('Unsupported denoising target')
    weighting = dict(prediction_type=prediction_type, min_snr_gamma=min_snr_gamma)
    validate_loss_weighting(weighting)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError('Diagnostic evidence exists; select a fresh --out for another attempt')
    torch.set_num_threads(8)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x, y, digest, stats = load_manifest_data(manifest)
    x, y = x.to(device), y.to(device)
    config = dict(channels=[16, 32, 48, 64], blocks_per_level=1,
                  conditioning_dim=64, attention_heads=4, dropout=0)
    if patch_size != 1:
        config.update(patch_size=patch_size, patch_detail=patch_detail)
    model = ConditionalUNet(**config).to(device)
    schedule = cosine_alpha_bar().to(device)
    generator = torch.Generator(device=device).manual_seed(29)
    probe_generator = torch.Generator(device='cpu').manual_seed(43)
    probe_noise = torch.randn(x.shape, generator=probe_generator).to(device)
    probe_t = torch.linspace(250, 750, len(x), device=device).long()
    probe = add_noise(x, probe_noise, probe_t, schedule)
    alpha = schedule[probe_t, None, None, None]
    target = probe_noise
    reconstruct = reconstruct_clean
    if prediction_type == 'velocity':
        target = velocity_target(x, probe_noise, alpha)
        reconstruct = reconstruct_clean_from_velocity

    @torch.no_grad()
    def measure():
        model.eval()
        prediction = model(probe, probe_t, y)
        clean = reconstruct(probe, prediction, alpha).clamp(-1, 1)
        return F.mse_loss(prediction, target).item(), F.mse_loss(clean, x).item(), clean

    before_mse, before_clean, before = measure()
    save_image(torch.cat([(x + 1) / 2, (before + 1) / 2]), out / 'before.png', nrow=len(x))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    settings = dict(effective_batch=len(x), microbatch=len(x), learning_rate=1e-3,
                    warmup=1, grad_clip=1.0, precision='fp32', null_probability=0,
                    **weighting)
    begin = time.perf_counter()
    losses = []
    for update in range(1, OVERFIT_UPDATES + 1):
        model.train()
        loss, norm, lr = optimizer_step(model, optimizer, x, y, schedule, generator, settings, update)
        if update % 100 == 0 or update == 1:
            probe_mse, clean_mse, _ = measure()
            row = {'update': update, 'loss': loss.item(), 'probe_mse': probe_mse,
                   'reconstruction_mse': clean_mse, 'seconds': time.perf_counter() - begin}
            losses.append(row)
            print(row, flush=True)
    after_mse, after_clean, after = measure()
    save_image(torch.cat([(x + 1) / 2, (after + 1) / 2]), out / 'after.png', nrow=len(x))
    with (out / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(losses[0]))
        writer.writeheader()
        writer.writerows(losses)
    record = {'purpose': 'integration correctness only, not generalization or final quality',
        'diagnostic_only': True, 'manifest_sha256': digest, 'stats': stats, 'model': config,
        'updates': OVERFIT_UPDATES, 'required_reduction': .80, **weighting,
        'probe_mse_before': before_mse, 'probe_mse_after': after_mse,
        'reduction': 1 - after_mse / before_mse, 'reconstruction_mse_before': before_clean,
        'reconstruction_mse_after': after_clean, 'seconds': time.perf_counter() - begin,
        'training_seed': 17, 'noise_seed': 29, 'probe_seed': 43,
        'probe_timesteps': probe_t.tolist(), 'environment': environment(), 'completed': timestamp()}
    if patch_size != 1:
        record['model_source_sha256'] = sha256('submission/model.py')
    record['passed'] = record['reduction'] >= .80 and after_clean < before_clean
    write_json(out / 'result.json', record)
    if not record['passed']:
        raise AssertionError('Fixed 2000-update diagnostic failed; investigate without weakening the criterion')
    return record


def benchmark(config_path='configs/ddpm.yaml', manifest='manifests/diagnostic_two_heads.csv',
              out='results/diagnostics/benchmark'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError('Benchmark exists; select a fresh --out')
    config = read_config(config_path)
    settings = config['training']
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    x, y, digest, stats = load_manifest_data(manifest)
    x, y = x.cuda(), y.cuda()
    model = ConditionalUNet(**config['model']).cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
        betas=tuple(settings['betas']), weight_decay=settings['weight_decay'])
    generator = torch.Generator(device='cuda').manual_seed(18)
    schedule = cosine_alpha_bar(**config['diffusion']).cuda()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    records = []
    for update in range(1, BENCHMARK_UPDATES + 1):
        loss, norm, lr = optimizer_step(model, optimizer, x, y, schedule, generator, settings, update)
        update_ema(ema, model, settings['ema_decay'])
        if update % 50 == 0 or update == 1:
            torch.cuda.synchronize()
            row = {'update': update, 'loss': loss.item(), 'seconds': time.perf_counter() - begin}
            records.append(row)
            print(row, flush=True)
    seconds = time.perf_counter() - begin
    # Ignore first 50 warmup/kernel-initialization updates when projecting compute time.
    steady = (records[-1]['seconds'] - records[1]['seconds']) / (BENCHMARK_UPDATES - 50)
    record = {'purpose': 'compute benchmark on two manifest rows, not a quality baseline',
        'diagnostic_only': True, 'config': config, 'config_sha256': sha256(config_path),
        'manifest_sha256': digest, 'updates': BENCHMARK_UPDATES, 'seconds': seconds,
        'steady_seconds_per_update': steady, 'estimated_50k_hours': steady * 50000 / 3600,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'records': records,
        'environment': environment(), 'weights_retained': False}
    write_json(out / 'result.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['overfit', 'benchmark'])
    p.add_argument('--manifest', default='manifests/diagnostic_two_heads.csv')
    p.add_argument('--config', default='configs/ddpm.yaml')
    p.add_argument('--out')
    p.add_argument('--prediction-type', choices=['epsilon', 'velocity'], default='epsilon')
    p.add_argument('--min-snr-gamma', type=float)
    p.add_argument('--patch-size', type=int, choices=[1, 4, 8], default=1)
    p.add_argument('--patch-detail', action='store_true')
    args = p.parse_args()
    if args.mode == 'overfit':
        print(overfit(args.manifest, out=args.out or 'results/diagnostics/overfit',
                      prediction_type=args.prediction_type, min_snr_gamma=args.min_snr_gamma,
                      patch_size=args.patch_size, patch_detail=args.patch_detail))
    else:
        print(benchmark(args.config, args.manifest, args.out or 'results/diagnostics/benchmark'))


if __name__ == '__main__':
    main()

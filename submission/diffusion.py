"""Cosine DDPM objective and deterministic DDIM sampling (eta=0)."""
import math
import torch


def cosine_alpha_bar(timesteps=1000, cosine_offset=0.008):
    if timesteps < 2 or cosine_offset < 0:
        raise ValueError('Invalid schedule')
    times = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos((times / timesteps + cosine_offset) / (1 + cosine_offset) * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - beta, dim=0).float()


def add_noise(clean, noise, timesteps, alpha_bar):
    a = alpha_bar[timesteps].reshape(-1, 1, 1, 1)
    return a.sqrt() * clean + (1 - a).sqrt() * noise


def reconstruct_clean(noisy, noise_prediction, a):
    return (noisy - (1 - a).sqrt() * noise_prediction) / a.sqrt()


def ddim_indices(timesteps, steps):
    if not 1 <= steps <= timesteps:
        raise ValueError('sampling steps must be in [1, timesteps]')
    return torch.linspace(timesteps - 1, 0, steps).round().long().tolist()


@torch.no_grad()
def sample(model, labels, seed, *, timesteps=1000, cosine_offset=0.008,
           steps=100, guidance=1.5, microbatch=32, execution_batch=32):
    if model.training:
        raise ValueError('Sampling requires model.eval()')
    if labels.ndim != 1 or labels.dtype != torch.long or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('labels must be a one-dimensional LongTensor containing 0/1')
    if microbatch < 1 or execution_batch < 1 or not math.isfinite(guidance):
        raise ValueError('Invalid sampling configuration')
    device = next(model.parameters()).device
    schedule = cosine_alpha_bar(timesteps, cosine_offset).to(device)
    indices = ddim_indices(timesteps, steps)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    # One CPU draw per output position: independent of labels and microbatch boundaries.
    noise = torch.stack([torch.randn(3, 32, 32, generator=generator) for _ in range(len(labels))]) \
        if len(labels) else torch.empty(0, 3, 32, 32)
    outputs = []
    chunk_size = min(microbatch, execution_batch)
    for start in range(0, len(labels), chunk_size):
        x = noise[start:start + chunk_size].to(device)
        y = labels[start:start + chunk_size].to(device)
        count = len(y)
        # Keep convolution/attention shapes fixed even if logical chunk boundaries change.
        # Near the cosine schedule's noisy endpoint, tiny arithmetic differences can grow.
        if count < execution_batch:
            x = torch.cat([x, x.new_zeros(execution_batch - count, 3, 32, 32)])
            y = torch.cat([y, y.new_full((execution_batch - count,), 2)])
        for index, t in enumerate(indices):
            time = torch.full((len(y),), t, device=device, dtype=torch.long)
            if guidance == 1:
                prediction = model(x, time, y)
            else:
                unconditional = model(x, time, torch.full_like(y, 2))
                prediction = unconditional + guidance * (model(x, time, y) - unconditional)
            a = schedule[t]
            clean = reconstruct_clean(x, prediction, a).clamp(-1, 1)
            # Recompute epsilon after clipping x0, matching the documented clipped DDIM convention.
            corrected_noise = (x - a.sqrt() * clean) / (1 - a).sqrt()
            previous = indices[index + 1] if index + 1 < len(indices) else -1
            if previous < 0:
                x = clean
            else:
                a_previous = schedule[previous]
                x = a_previous.sqrt() * clean + (1 - a_previous).sqrt() * corrected_noise
        outputs.append((x[:count].float() + 1).mul(0.5).clamp(0, 1))
    return torch.cat(outputs) if outputs else torch.empty(0, 3, 32, 32, device=device)

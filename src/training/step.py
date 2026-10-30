"""The same optimizer update is used by training and hardware diagnostics."""
import torch
from torch.nn import functional as F
from submission.diffusion import add_noise


def optimizer_step(model, optimizer, images, labels, alpha_bar, generator, settings, update):
    device = images.device
    effective, micro = settings['effective_batch'], settings['microbatch']
    accumulation = effective // micro
    bf16 = settings['precision'] == 'bf16' and device.type == 'cuda'
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
        drop = torch.rand(micro, generator=generator, device=device) < settings['null_probability']
        y[drop] = 2
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
    return torch.stack(losses).mean(), norm, lr


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, value in zip(ema.parameters(), model.parameters()):
        target.lerp_(value, 1 - decay)

"""The same optimizer update is used by training and hardware diagnostics."""
import math

import torch
from torch.nn import functional as F
from submission.diffusion import add_noise, reconstruct_clean, velocity_target
from submission.edm import LOG_SIGMA_MEAN, LOG_SIGMA_STD, noise_condition, training_pair
from submission.flow import flow_pair


def validate_loss_weighting(settings):
    if type(settings.get('appearance_sampling', False)) is not bool:
        raise ValueError('appearance_sampling must be a boolean')
    gamma = settings.get('min_snr_gamma')
    focus = settings.get('timestep_sampling_focus')
    if focus is not None and (type(focus) not in (int, float) or not 0 < focus < 1
                              or gamma is not None or settings.get('prediction_type', 'epsilon') != 'epsilon'):
        raise ValueError('Timestep focus must be between zero and one for unweighted epsilon prediction')
    perturbation = settings.get('noise_input_perturbation', 0)
    if type(perturbation) not in (int, float) or not math.isfinite(perturbation) or perturbation < 0:
        raise ValueError('noise_input_perturbation must be nonnegative and finite')
    if perturbation and settings.get('prediction_type', 'epsilon') != 'epsilon':
        raise ValueError('Noise input perturbation requires epsilon prediction')
    coarse = settings.get('coarse_epsilon_weight', 0)
    if type(coarse) not in (int, float) or not math.isfinite(coarse) or coarse < 0:
        raise ValueError('coarse_epsilon_weight must be nonnegative and finite')
    if coarse and (gamma is not None or settings.get('prediction_type', 'epsilon') != 'epsilon'):
        raise ValueError('Coarse epsilon loss requires unweighted epsilon prediction')
    if gamma is None:
        return
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('min_snr_gamma must be positive and finite')
    if settings.get('prediction_type', 'epsilon') != 'epsilon':
        raise ValueError('Min-SNR weighting is implemented for the epsilon target')


def epsilon_min_snr_weights(alpha_bar, gamma):
    # min(SNR, gamma) / SNR with SNR = alpha_bar / (1 - alpha_bar).
    return (gamma * (1 - alpha_bar) / alpha_bar.clamp_min(torch.finfo(alpha_bar.dtype).tiny)).clamp_max(1)


def coarse_epsilon_mse(error):
    # Nonoverlapping 4x4 averages reduce independent noise variance by 16.
    return F.avg_pool2d(error, kernel_size=4).square().mean() * 16


def timestep_sampling_probabilities(length, focus, device):
    """Discretize a broad Gaussian with a 25% uniform component over all timesteps."""
    normalized = torch.linspace(0, 1, length, device=device)
    gaussian = torch.softmax(-.5 * ((normalized - focus) / .22).square(), dim=0)
    return .75 * gaussian + .25 / length


def optimizer_step(model, optimizer, images, labels, alpha_bar, generator, settings, update, *, image_weights=None):
    device = images.device
    effective, micro = settings['effective_batch'], settings['microbatch']
    accumulation = effective // micro
    bf16 = settings['precision'] == 'bf16' and device.type == 'cuda'
    prediction_type = settings.get('prediction_type', 'epsilon')
    gamma = settings.get('min_snr_gamma')
    coarse = settings.get('coarse_epsilon_weight', 0)
    perturbation = settings.get('noise_input_perturbation', 0)
    focus = settings.get('timestep_sampling_focus')
    timestep_probabilities = None if focus is None else timestep_sampling_probabilities(len(alpha_bar), focus, device)
    lr = settings['learning_rate'] * min(update / max(settings['warmup'], 1), 1)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(accumulation):
        if image_weights is None:
            indices = torch.randint(len(labels), (micro,), generator=generator, device=device)
        else:
            indices = torch.multinomial(image_weights, micro, replacement=True, generator=generator)
        x0, y = images[indices], labels[indices].clone()
        if prediction_type == 'edm':
            log_sigma = torch.randn((micro, 1, 1, 1), generator=generator, device=device)
            sigma = (log_sigma * LOG_SIGMA_STD + LOG_SIGMA_MEAN).exp()
            t = noise_condition(sigma)
        elif timestep_probabilities is not None:
            t = torch.multinomial(timestep_probabilities, micro, replacement=True, generator=generator)
        else:
            t = torch.randint(len(alpha_bar), (micro,), generator=generator, device=device)
        noise = torch.randn(x0.shape, generator=generator, device=device)
        drop = torch.rand(micro, generator=generator, device=device) < settings['null_probability']
        y[drop] = 2
        if prediction_type == 'edm':
            xt, target, _, _, _ = training_pair(x0, noise, sigma)
        elif prediction_type == 'flow':
            xt, target = flow_pair(x0, noise, t, len(alpha_bar))
        else:
            input_noise = noise
            if perturbation:
                independent_noise = torch.randn(x0.shape, generator=generator, device=device)
                input_noise = noise + perturbation * independent_noise
            xt = add_noise(x0, input_noise, t, alpha_bar)
            target = velocity_target(x0, noise, alpha_bar[t, None, None, None]) \
                if prediction_type == 'velocity' else noise
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            use_hint = getattr(model, 'self_condition', False) and torch.rand(
                (), generator=generator, device=device).item() < .5
            if use_hint:
                with torch.no_grad():
                    hint = reconstruct_clean(xt, model(xt, t, y).float(), alpha_bar[t, None, None, None]).clamp(-1, 1)
                predicted = model(xt, t, y, hint)
            else:
                predicted = model(xt, t, y)
            if gamma is None:
                loss = F.mse_loss(predicted.float(), target)
                if coarse:
                    loss = (loss + coarse * coarse_epsilon_mse(predicted.float() - target)) / (1 + coarse)
            else:
                errors = F.mse_loss(predicted.float(), target, reduction='none').mean(dim=(1, 2, 3))
                loss = (errors * epsilon_min_snr_weights(alpha_bar[t], gamma)).mean()
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

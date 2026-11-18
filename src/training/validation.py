"""Fixed-noise loss on the separately held-out pool crops; never a quality metric."""
import csv
from pathlib import Path
import torch
from submission.diffusion import add_noise, velocity_target
from submission.edm import noise_condition, training_pair
from submission.flow import flow_pair


def require_disjoint_sources(training_manifest, validation_manifest):
    def identifiers(path):
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            return {row['pool_id'].strip() for row in csv.DictReader(stream)}
    if identifiers(training_manifest) & identifiers(validation_manifest):
        raise ValueError('Training and validation manifests share pool sources')


@torch.no_grad()
def validation_loss(model, images, labels, alpha_bar, *, seed=3024,
                    timesteps=(50, 250, 500, 750), microbatch=32, prediction_type='epsilon'):
    if not len(labels) or images.shape != (len(labels), 3, 32, 32):
        raise ValueError('Validation requires nonempty exact 32x32 RGB crops')
    if not timesteps or any(not 0 <= t < len(alpha_bar) for t in timesteps) or microbatch < 1 or \
            prediction_type not in ('epsilon', 'velocity', 'edm', 'flow'):
        raise ValueError('Invalid fixed validation protocol')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    original_mode = model.training
    model.eval()
    records = []
    try:
        for timestep in timesteps:
            noise = torch.randn(images.shape, generator=generator, device='cpu').to(images.device)
            losses = []
            for start in range(0, len(labels), microbatch):
                clean = images[start:start+microbatch]
                noise_batch = noise[start:start+microbatch]
                times = torch.full((len(clean),), timestep, device=images.device, dtype=torch.long)
                alpha = alpha_bar[timestep]
                if prediction_type == 'edm':
                    sigma = ((1 - alpha) / alpha).sqrt().expand(len(clean), 1, 1, 1)
                    noisy, target, _, _, _ = training_pair(clean, noise_batch, sigma)
                    times = noise_condition(sigma)
                elif prediction_type == 'flow':
                    noisy, target = flow_pair(clean, noise_batch, times, len(alpha_bar))
                else:
                    noisy = add_noise(clean, noise_batch, times, alpha_bar)
                    target = noise_batch if prediction_type == 'epsilon' else velocity_target(
                        clean, noise_batch, alpha)
                prediction = model(noisy, times, labels[start:start+microbatch])
                losses.append((prediction.float() - target).square().mean(dim=(1, 2, 3)).cpu())
            loss = torch.cat(losses)
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Nonfinite held-out denoising loss')
            records.append({'timestep': timestep, 'mse': loss.mean().item(),
                'by_class': {str(label): loss[labels.cpu() == label].mean().item()
                             if (labels == label).any() else None for label in [0, 1]}})
    finally:
        model.train(original_mode)
    return {'images': len(labels), 'seed': seed, 'microbatch': microbatch, 'prediction_type': prediction_type,
            'mean_mse': sum(row['mse'] for row in records) / len(records), 'records': records,
            'scope': 'Fixed-noise diagnostic on disjoint held-out pool crops. Not FID, accuracy, generalization certification, or official quality evidence.'}

"""Estimate within-species coverage weights from exact training crop pixels."""

import torch
from torch.nn import functional as F


def appearance_sampling_weights(images, labels):
    """Give locally uncommon visual features more training exposure.

    Fit the density estimate inside each species and preserve its total sampling
    mass. Colour and low-resolution luminance describe appearance, not breeds.
    """
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32) or labels.shape != (len(images),):
        raise ValueError('Expected one species label per exact RGB crop')
    if images.device.type != 'cpu' or labels.device.type != 'cpu':
        raise ValueError('Compute the density weights on the CPU before moving training tensors')
    grid = F.avg_pool2d(images.float(), kernel_size=4)
    luminance = .299 * grid[:, 0] + .587 * grid[:, 1] + .114 * grid[:, 2]
    layout = (luminance - luminance.mean(dim=(1, 2), keepdim=True))[:, 1:7, 1:7].flatten(1) / 4
    centre_colour = grid[:, :, 2:6, 2:6].mean(dim=(2, 3))
    features = torch.cat((layout, centre_colour), dim=1)
    weights = torch.ones(len(labels), dtype=torch.float32)
    for label in (0, 1):
        members = labels == label
        count = int(members.sum())
        if count < 2:
            continue
        distance = torch.cdist(features[members], features[members])
        distance.fill_diagonal_(float('inf'))
        local = distance.topk(min(16, count - 1), largest=False).values.mean(dim=1)
        median = local.median().clamp_min(1e-6)
        relative = (local / median).pow(1.5).clamp(.60, 1.55)
        weights[members] = relative / relative.mean()
    return weights

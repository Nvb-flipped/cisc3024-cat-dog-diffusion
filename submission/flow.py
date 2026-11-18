"""Linear flow matching primitives for noise-to-pixel sampling."""

import torch


def flow_pair(clean, noise, timesteps, total_timesteps):
    fraction = timesteps.to(dtype=clean.dtype).view(-1, 1, 1, 1) / (total_timesteps - 1)
    velocity = noise - clean
    return clean + fraction * velocity, velocity


def reconstruct_flow(image, velocity, timesteps, total_timesteps):
    fraction = timesteps.to(dtype=image.dtype).view(-1, 1, 1, 1) / (total_timesteps - 1)
    return image - fraction * velocity


def heun_flow(model, image, labels, indices, timesteps, guidance):
    """Integrate the same velocity field from Gaussian noise to clean pixels."""
    null_labels = torch.full_like(labels, 2)

    def predict(value, index):
        time = torch.full_like(labels, index)
        conditional = model(value, time, labels)
        if guidance == 1:
            return conditional
        unconditional = model(value, time, null_labels)
        return unconditional + guidance * (conditional - unconditional)

    velocity = predict(image, indices[0])
    for current, following in zip(indices, indices[1:]):
        delta = (following - current) / (timesteps - 1)
        predictor = image + delta * velocity
        correction = predict(predictor, following)
        image = image + (.5 * delta) * (velocity + correction)
        if following:
            velocity = predict(image, following)
    return image

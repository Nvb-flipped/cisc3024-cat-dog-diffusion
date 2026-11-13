"""EDM noise conditioning and deterministic Heun integration."""

import torch


SIGMA_DATA = 0.5
LOG_SIGMA_MEAN = -1.2
LOG_SIGMA_STD = 1.2


def precondition(sigma):
    variance = sigma.square() + SIGMA_DATA ** 2
    scale_in = variance.rsqrt()
    return scale_in, SIGMA_DATA ** 2 / variance, sigma * SIGMA_DATA * scale_in


def training_pair(clean, noise, sigma):
    scale_in, skip, scale_out = precondition(sigma)
    noisy = clean + sigma * noise
    # Algebraically (clean - skip * noisy) / scale_out, without subtraction at small sigma.
    target = (sigma / SIGMA_DATA * clean - SIGMA_DATA * noise) * scale_in
    return noisy * scale_in, target, noisy, skip, scale_out


def noise_condition(sigma):
    return sigma.flatten().log() / 4


def heun_sigmas(steps, device):
    if steps < 2:
        raise ValueError('EDM requires at least two positive noise levels')
    endpoints = torch.linspace(80 ** (1 / 7), 0.002 ** (1 / 7), steps,
                               device=device, dtype=torch.float64).pow(7)
    return torch.cat((endpoints, endpoints.new_zeros(1)))


def heun_sample(model, noise, labels, sigmas, guidance):
    def denoise(state, sigma):
        sigma = sigma.float()
        scale_in, skip, scale_out = precondition(sigma)
        image = scale_in * state.float()
        condition = noise_condition(sigma).expand(len(labels))
        prediction = model(image, condition, labels)
        if guidance != 1:
            unconditioned = model(image, condition, torch.full_like(labels, 2))
            prediction = unconditioned + guidance * (prediction - unconditioned)
        return (skip * state.float() + scale_out * prediction).double()

    state = noise.double() * sigmas[0]
    for index, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        clean = denoise(state, sigma)
        if index == len(sigmas) - 2:
            return clean
        slope = (state - clean) / sigma
        delta = next_sigma - sigma
        trial = state + delta * slope
        next_slope = (trial - denoise(trial, next_sigma)) / next_sigma
        state = state + delta * (slope + next_slope) / 2

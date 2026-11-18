"""Offline API for a fixed mixture of two exact-pool diffusion fits."""

import json
from pathlib import Path

import torch

if __package__:
    from .diffusion import sample
    from .model import ConditionalUNet
else:
    from diffusion import sample
    from model import ConditionalUNet

HERE = Path(__file__).resolve().parent


def load_model(device: torch.device) -> torch.nn.Module:
    config = json.loads((HERE / 'inference.json').read_text(encoding='utf-8'))
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.ModuleList(ConditionalUNet(**config['model']) for _ in range(2))
    model.load_state_dict(torch.load(HERE / 'weights.pt', map_location='cpu', weights_only=True), strict=True)
    model.sampling_config = config['sampling']
    return model.to(device).eval()


@torch.no_grad()
def generate(model: torch.nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    if labels.ndim != 1 or labels.dtype != torch.long or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('labels must be a one-dimensional LongTensor containing 0/1')
    device = next(model.parameters()).device
    result = torch.empty((len(labels), 3, 32, 32), device=device, dtype=torch.float32)
    router = torch.Generator(device='cpu').manual_seed(int(seed) ^ 0x503BDCACE09272CB)
    members = torch.randint(0, 2, (len(labels),), generator=router)
    for index, net in enumerate(model):
        positions = torch.where(members == index)[0]
        if positions.numel():
            branch_seed = int(seed) ^ (index * 0x2DFCA1B398E05481)
            branch = sample(net, labels[positions.to(labels.device)], branch_seed, **model.sampling_config)
            result.index_copy_(0, positions.to(device), branch)
    return result

"""Self-contained inference for the two-checkpoint autoguidance experiment."""

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


class AutoguidedDenoiser(torch.nn.Module):
    def __init__(self, model_config, scale):
        super().__init__()
        self.main = ConditionalUNet(**model_config)
        self.guide = ConditionalUNet(**model_config)
        self.scale = scale

    def forward(self, x, timesteps, labels):
        prediction = self.main(x, timesteps, labels)
        weaker = self.guide(x, timesteps, labels)
        return prediction + self.scale * (prediction - weaker)


def load_model(device: torch.device) -> torch.nn.Module:
    config = json.loads((HERE / 'inference.json').read_text(encoding='utf-8'))
    with torch.random.fork_rng(devices=[]):
        model = AutoguidedDenoiser(config['model'], config['autoguidance_scale'])
    state = torch.load(HERE / 'weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.sampling_config = config['sampling']
    return model.to(device).eval()


@torch.no_grad()
def generate(model: torch.nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    return sample(model, labels, seed, **model.sampling_config)

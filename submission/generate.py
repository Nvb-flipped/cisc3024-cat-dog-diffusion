"""Official self-contained, offline generation entry points."""
import json
from pathlib import Path
import torch

try:
    from .model import ConditionalUNet
    from .diffusion import sample
except ImportError:  # Official evaluator imports generate.py as a standalone module.
    from model import ConditionalUNet
    from diffusion import sample

HERE = Path(__file__).resolve().parent


def load_model(device: torch.device) -> torch.nn.Module:
    config = json.loads((HERE / 'inference.json').read_text(encoding='utf-8'))
    # Parameter initialization must not consume the caller's CPU/global CUDA RNG streams.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalUNet(**config['model'])
    state = torch.load(HERE / 'weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.sampling_config = config['sampling']
    return model.to(device).eval()


@torch.no_grad()
def generate(model: torch.nn.Module, labels: torch.Tensor, seed: int) -> torch.Tensor:
    return sample(model, labels, seed, **model.sampling_config)

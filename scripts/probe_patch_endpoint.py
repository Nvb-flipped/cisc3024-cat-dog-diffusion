"""Compare the trained denoisers at the noisy DDIM endpoint on the pool holdout."""

import argparse
import json
from pathlib import Path

import torch

from src.common import sha256, write_json
from src.release.check import check_files
from src.training.data import load_manifest_data
from submission.diffusion import add_noise, cosine_alpha_bar, reconstruct_clean
from submission.model import ConditionalUNet


@torch.inference_mode()
def probe(exports, manifest):
    torch.set_num_threads(4)
    clean, labels, holdout_sha, _ = load_manifest_data(manifest)
    generator = torch.Generator(device='cpu').manual_seed(3024)
    noise = torch.randn(clean.shape, generator=generator)
    schedule = cosine_alpha_bar()
    records = []
    for folder in exports:
        files = check_files(folder)['file_sha256']
        config = json.loads((folder / 'inference.json').read_text(encoding='utf-8'))
        model = ConditionalUNet(**config['model']).eval()
        model.load_state_dict(torch.load(folder / 'weights.pt', weights_only=True, map_location='cpu'))
        guidance = config['sampling']['guidance']
        results = []
        for timestep in (50, 250, 500, 750, 975, 990, 999):
            t = torch.full_like(labels, timestep)
            x = add_noise(clean, noise, t, schedule)
            prediction = []
            for start in range(0, len(labels), 16):
                group = slice(start, start + 16)
                conditional = model(x[group], t[group], labels[group])
                null = model(x[group], t[group], torch.full_like(labels[group], 2))
                prediction.append(null + guidance * (conditional - null))
            prediction = torch.cat(prediction)
            estimate = reconstruct_clean(x, prediction, schedule[timestep])
            results.append({'timestep': timestep,
                            'guided_noise_mse': (prediction - noise).square().mean().item(),
                            'clipped_clean_mse': (estimate.clamp(-1, 1) - clean).square().mean().item(),
                            'estimate_outside_range': estimate.abs().gt(1).float().mean().item()})
        records.append({'export': str(folder), 'weights_sha256': files['weights.pt'],
                        'inference_sha256': files['inference.json'], 'guidance': guidance,
                        'records': results})
    return {'purpose': 'Single-step denoiser diagnostic on held-out pool crops',
            'manifest_sha256': holdout_sha, 'images': len(labels), 'probe_seed': 3024,
            'source_sha256': sha256(__file__), 'exports': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exports', nargs='+', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('manifests/v002.validation.csv'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Use a fresh endpoint receipt: {args.out}')
    write_json(args.out, probe(args.exports, args.manifest))

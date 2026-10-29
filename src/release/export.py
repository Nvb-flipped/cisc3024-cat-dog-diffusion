"""Export exactly one trained checkpoint and its exact immutable manifest."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from src.common import ROOT, sha256, timestamp, write_json
from submission.model import ConditionalUNet

INFERENCE_FILES = ['__init__.py', 'generate.py', 'model.py', 'diffusion.py']


def export_checkpoint(checkpoint, manifest, out, *, weights='ema', steps=None, guidance=None, microbatch=None):
    checkpoint, manifest, out = Path(checkpoint), Path(manifest), Path(out)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    digest = sha256(manifest)
    if state['manifest_sha256'] != digest:
        raise ValueError('Checkpoint/manifest mismatch; final model must list all and only its training rows')
    model = ConditionalUNet(**state['config']['model'])
    model.load_state_dict(state[weights], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 18_000_000:
        raise ValueError('Model exceeds 18M engineering ceiling')
    if (out / 'weights.pt').exists() or (out / 'manifest.csv').exists():
        raise FileExistsError('Candidate export already exists; use a new immutable output directory')
    sampling = dict(state['config']['sampling']) | state['config']['diffusion']
    for key, value in [('steps', steps), ('guidance', guidance), ('microbatch', microbatch)]:
        if value is not None:
            sampling[key] = value
    if not 1 <= sampling['steps'] <= sampling['timesteps'] or sampling['microbatch'] < 1:
        raise ValueError('Invalid sampler configuration')
    out.mkdir(parents=True, exist_ok=True)
    for name in INFERENCE_FILES:
        source = ROOT / 'submission' / name
        dest = out / name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    shutil.copy2(manifest, out / 'manifest.csv')
    torch.save({name: tensor.detach().cpu() for name, tensor in state[weights].items()}, out / 'weights.pt')
    write_json(out / 'inference.json', {'model': state['config']['model'], 'sampling': sampling})
    write_json(out / 'provenance.json', {'checkpoint_sha256': sha256(checkpoint),
        'manifest_sha256': digest, 'weights_sha256': sha256(out / 'weights.pt'),
        'config_sha256': sha256(out / 'inference.json'), 'weights_variant': weights,
        'training_update': state['update'], 'training_seed': state['seed'],
        'parameters': parameters, 'exported': timestamp(),
        'inference_source_sha256': {name: sha256(out / name) for name in INFERENCE_FILES}})
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    if size > 90_000_000:
        raise ValueError(f'Export exceeds internal 90,000,000-byte ceiling: {size}')
    return {'path': str(out), 'parameters': parameters, 'bytes': size,
            'manifest_sha256': digest, 'weights': weights, 'sampling': sampling}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path, help='Selected update_XXXXXX.pt file (no implicit selection)')
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--out', default='submission', type=Path)
    p.add_argument('--weights', choices=['ema', 'model'], default='ema')
    p.add_argument('--steps', type=int)
    p.add_argument('--guidance', type=float)
    p.add_argument('--microbatch', type=int)
    args = p.parse_args()
    print(export_checkpoint(args.run, args.manifest, args.out, weights=args.weights,
                            steps=args.steps, guidance=args.guidance, microbatch=args.microbatch))


if __name__ == '__main__':
    main()

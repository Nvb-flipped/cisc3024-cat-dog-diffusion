"""Check actual exported inference in a fresh offline process and clean directory."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from src.common import ROOT, sha256, write_json


def check_files(folder):
    folder = Path(folder).resolve()
    required = ['generate.py', 'model.py', 'diffusion.py', 'weights.pt', 'inference.json', 'manifest.csv', 'provenance.json']
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise ValueError(f'Missing export files: {missing}')
    config = json.loads((folder / 'inference.json').read_text(encoding='utf-8'))
    if config.get('sampling', {}).get('prediction_type') == 'edm' and not (folder / 'edm.py').is_file():
        raise ValueError('EDM export is missing edm.py')
    if config.get('sampling', {}).get('prediction_type') == 'flow' and not (folder / 'flow.py').is_file():
        raise ValueError('Flow export is missing flow.py')
    files = [f for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    size = sum(f.stat().st_size for f in files)
    forbidden = [str(f) for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.zip', '.npz', '.npy', '.safetensors']]
    weights = [f for f in files if f.suffix.lower() in ['.pt', '.pth']]
    if forbidden or len(weights) != 1 or size > 90_000_000:
        raise ValueError(f'Export resource/allowlist failure: bytes={size}, weights={len(weights)}, forbidden={forbidden}')
    provenance = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    for name, expected in [('manifest.csv', provenance['manifest_sha256']),
                           ('weights.pt', provenance['weights_sha256']),
                           ('inference.json', provenance['config_sha256'])]:
        if sha256(folder / name) != expected:
            raise ValueError(f'Export provenance mismatch: {name}')
    for name, expected in provenance['inference_source_sha256'].items():
        if sha256(folder / name) != expected:
            raise ValueError(f'Inference code changed after export: {name}')
    return {'submission_bytes': size, 'files': [str(f.relative_to(folder)) for f in files],
            'file_sha256': {str(f.relative_to(folder)).replace('\\', '/'): sha256(f) for f in files}}


def child_check(folder, device, counts):
    import importlib.util
    import random
    import socket
    import time
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    def no_network(*args, **kwargs):
        raise RuntimeError('Network disabled during inference check')
    socket.socket.connect = no_network
    socket.create_connection = no_network
    # Make project source unavailable to the imported inference package.
    project_root = ROOT.resolve()
    sys.path[:] = [str(folder)] + [entry for entry in sys.path
        if Path(entry or '.').resolve() != project_root]
    for name in list(sys.modules):
        if name == 'src' or name.startswith('src.'):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location('candidate_generate', folder / 'generate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    if device == 'cuda':
        torch.cuda.init()
        cuda_rng = torch.cuda.get_rng_state().clone()
    start = time.perf_counter()
    model = module.load_model(torch.device(device))
    parameters = sum(p.numel() for p in model.parameters())
    if model.training or parameters > 20_000_000:
        raise ValueError('Model train/eval mode or parameter budget failure')
    records = []
    for count in counts:
        labels = (torch.arange(count, device=device) % 2).long()
        begin = time.perf_counter()
        first = module.generate(model, labels, 3024)
        if first.shape != (count, 3, 32, 32) or not first.is_floating_point():
            raise ValueError('Output shape/dtype failure')
        if not torch.isfinite(first).all() or not (first >= 0).all() or not (first <= 1).all():
            raise ValueError('Output finite/range failure')
        repeat = module.generate(model, labels, 3024)
        if not torch.equal(first, repeat):
            raise ValueError('Generation is not exactly repeatable on this device')
        other_seed = module.generate(model, labels, 3025)
        if count and torch.equal(first, other_seed):
            raise ValueError('Seed does not change outputs')
        if count > 1:
            changed = labels.clone()
            changed[1:] = 1 - changed[1:]
            other_labels = module.generate(model, changed, 3024)
            if not torch.equal(first[0], other_labels[0]):
                raise ValueError('A target depends on other examples labels')
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise ValueError('Caller CPU RNG changed')
        if device == 'cuda' and not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
            raise ValueError('Caller CUDA RNG changed')
        if random.getstate() != python_rng:
            raise ValueError('Caller Python RNG changed')
        after_numpy = np.random.get_state()
        if not np.array_equal(numpy_rng[1], after_numpy[1]) or numpy_rng[2:] != after_numpy[2:]:
            raise ValueError('Caller NumPy RNG changed')
        records.append({'count': count, 'checks_seconds': time.perf_counter() - begin,
                        'sample_sha256': __import__('hashlib').sha256(first.detach().cpu().numpy().tobytes()).hexdigest()})
    write_json(Path(os.environ['PROJECT_CHECK_OUTPUT']), {'device': device, 'parameters': parameters,
        'offline': True, 'clean_working_directory': str(Path.cwd()), 'records': records,
        'seconds': time.perf_counter() - start})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission', default='submission', type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--counts', default='1,2,20,50,500')
    p.add_argument('--out', default='results/release/offline_check.json', type=Path)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    folder = args.submission.resolve()
    counts = [int(v) for v in args.counts.split(',')]
    if args.child:
        return child_check(folder, args.device, counts)
    resource = check_files(folder)
    clean = ROOT / 'tmp/offline_check'
    clean.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    child_env = dict(os.environ) | {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'PROJECT_CHECK_OUTPUT': str(out), 'PYTHONPATH': str(ROOT)}
    command = [sys.executable, '-m', 'src.release.check', '--child', '--submission', str(folder),
               '--device', args.device, '--counts', args.counts]
    subprocess.run(command, cwd=clean, env=child_env, check=True)
    result = json.loads(out.read_text()) | resource
    write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

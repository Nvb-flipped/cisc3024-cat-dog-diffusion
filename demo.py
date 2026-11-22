"""Offline demo of the submitted cat/dog head generator. Runs on Windows, macOS and Linux.

    python demo.py --check          verify the interface on this computer (seconds on a GPU, many minutes on a CPU)
    python demo.py --seed 0         generate one 16-image grid (8 cats, 8 dogs) for seed 0
    python demo.py                  interactive: type a seed for each grid, q to quit

Needs only Python 3.10 or newer, torch and pillow (see requirements-demo.txt). No network, no dataset and no GPU are needed:
an NVIDIA GPU is used when torch can see one, otherwise the CPU. The model is the one in submission/ (its weights hash is
printed). The same seed gives the same image on the same computer; another CPU or GPU may differ in the last bits, so the
printed image hash can differ between computers.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

try:
    import torch
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # a short instruction instead of a traceback
    sys.exit(f'Missing Python package "{exc.name}". Install the demo requirements, for example:\n'
             '    python -m pip install torch pillow\nsee README.md, section "Run the demo".')

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE / 'submission'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_generator(folder):
    """Import submission/generate.py the way the official evaluator does (as a standalone module)."""
    spec = importlib.util.spec_from_file_location('demo_generate', folder / 'generate.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def pick_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        sys.exit('No CUDA GPU is visible to torch. Run without --device, or with --device cpu.')
    if name == 'cuda' or (name == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda')
    return torch.device('cpu')  # Apple "mps" is deliberately not used: the sampler needs FFT operations that it may lack


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_grid(images, seed, output):
    tile, gutter, margin, heading = 128, 8, 16, 30
    width = 2 * margin + 4 * tile + 3 * gutter
    height = 2 * margin + 4 * tile + 2 * gutter + 2 * heading
    grid = Image.new('RGB', (width, height), '#f7f8fa')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # older Pillow without a size argument
        font = ImageFont.load_default()
    for group, name in enumerate(('Cat', 'Dog')):
        top = margin + group * (2 * tile + gutter + heading)
        draw.text((margin, top + 2), f'{name} - seed {seed}', fill='#182235', font=font)
        for index in range(8):
            rgb = (images[group * 8 + index].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
            preview = Image.fromarray(rgb).resize((tile, tile), Image.NEAREST)
            x = margin + (index % 4) * (tile + gutter)
            y = top + heading + (index // 4) * (tile + gutter)
            grid.paste(preview, (x, y))
    grid.save(output)


def run_check(module, model, device):
    """Interface checks from the course template, on this computer. Returns True if every check passes."""
    labels = torch.tensor([0, 1, 0, 1], device=device)
    steps = [('seed 5, labels [0,1,0,1]', labels, 5), ('seed 5 again', labels, 5), ('seed 6', labels, 6),
             ('seed 5, labels [0,0,0,0]', torch.tensor([0, 0, 0, 0], device=device), 5)]
    outputs = []
    for number, (name, y, seed) in enumerate(steps, 1):
        start = time.perf_counter()
        outputs.append(module.generate(model, y, seed))
        sync(device)
        print(f'  generated {number}/{len(steps)} ({name}) in {time.perf_counter() - start:.1f} s', flush=True)
    a, b, other_seed, other_labels = outputs
    tolerance = 1e-5
    results = [
        ('same seed gives the same images', (a - b).abs().max().item() <= tolerance, f'max difference {(a - b).abs().max().item():.1e}'),
        ('a different seed changes every image', bool(((a - other_seed).abs().flatten(1).mean(1) > 0.01).all()),
         f'mean change per image {[round(v, 2) for v in (a - other_seed).abs().flatten(1).mean(1).tolist()]}'),
        ('image 0 does not depend on the other labels', (a[0] - other_labels[0]).abs().max().item() <= tolerance,
         f'max difference {(a[0] - other_labels[0]).abs().max().item():.1e}'),
        ('output is [N,3,32,32], finite, within [0,1]',
         tuple(a.shape) == (4, 3, 32, 32) and bool(torch.isfinite(a).all()) and a.min().item() >= 0 and a.max().item() <= 1,
         f'shape {tuple(a.shape)}, range [{a.min().item():.2f}, {a.max().item():.2f}]'),
    ]
    for name, passed, detail in results:
        print(f'  [{"PASS" if passed else "FAIL"}] {name} ({detail})')
    return all(passed for _, passed, _ in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--submission', type=Path, default=SUBMISSION, help='folder with generate.py and weights.pt')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seed', type=int, help='generate once and exit (omit to be asked for seeds)')
    parser.add_argument('--check', action='store_true', help='run the interface self-check and exit')
    parser.add_argument('--out', type=Path, default=Path('demo_output'), help='folder for the PNG grids')
    args = parser.parse_args()

    folder = args.submission.resolve()
    if not (folder / 'generate.py').is_file() or not (folder / 'weights.pt').is_file():
        sys.exit(f'{folder} must contain generate.py and weights.pt (run this script from the unzipped project folder).')
    weights_hash = sha256(folder / 'weights.pt')
    recorded = json.loads((folder / 'provenance.json').read_text(encoding='utf-8')).get('weights_sha256')
    if recorded and recorded != weights_hash:
        sys.exit('weights.pt does not match the hash recorded in provenance.json: the file is damaged or was changed.')
    device = pick_device(args.device)
    print(f'Python {sys.version.split()[0]}, torch {torch.__version__}, device {device}', flush=True)
    if device.type == 'cpu':
        print('Note: running on the CPU. Generating a 16-image grid takes minutes here (about half a second on an NVIDIA GPU);\n'
              '      the results are the same, only slower.', flush=True)
    print('Loading the model ...', flush=True)
    module = load_generator(folder)
    model = module.load_model(device)
    labels = torch.arange(2, device=device).repeat_interleave(8)
    module.generate(model, labels[:1], 0)  # warm-up
    sync(device)
    print(f'Model ready. Weights SHA-256 {weights_hash}', flush=True)

    if args.check:
        print('Interface self-check:')
        ok = run_check(module, model, device)
        print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
        sys.exit(0 if ok else 1)

    args.out.mkdir(parents=True, exist_ok=True)
    while True:
        seed = args.seed
        if seed is None:
            try:
                response = input('Enter a seed, or q to quit [0]: ').strip()
            except EOFError:
                break
            if response.lower() == 'q':
                break
            try:
                seed = int(response or '0')
            except ValueError:
                print('Enter an integer seed.')
                continue
        sync(device)
        start = time.perf_counter()
        images = module.generate(model, labels, seed).cpu()
        sync(device)
        seconds = time.perf_counter() - start
        output = args.out / f'seed_{seed}.png'
        save_grid(images, seed, output)
        print(f'{seconds:.2f} s for 16 images on {device.type}. {output.resolve()}  image SHA-256 {sha256(output)}', flush=True)
        if args.seed is not None:
            break


if __name__ == '__main__':
    main()

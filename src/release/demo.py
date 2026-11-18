"""Preload the selected offline export and generate a 16-image demo for each seed."""

import argparse
import importlib.util
import sys
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageDraw, ImageFont

from src.common import sha256
from src.release.check import check_files


def load_generator(folder):
    spec = importlib.util.spec_from_file_location('demo_generate', folder / 'generate.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def save_grid(images, seed, output):
    tile_size, gutter, margin, heading = 128, 8, 16, 30
    width = 2 * margin + 4 * tile_size + 3 * gutter
    height = 2 * margin + 4 * tile_size + 2 * gutter + 2 * heading
    grid = Image.new('RGB', (width, height), '#f7f8fa')
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=16)
    for group, name in enumerate(('Cat', 'Dog')):
        y = margin + group * (2 * tile_size + gutter + heading)
        draw.text((margin, y + 2), f'{name} · seed {seed}', fill='#182235', font=font)
        for index in range(8):
            tile = images[group * 8 + index]
            rgb = (tile.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
            preview = Image.fromarray(rgb).resize((tile_size, tile_size), Image.Resampling.NEAREST)
            left = margin + (index % 4) * (tile_size + gutter)
            top = y + heading + (index // 4) * (tile_size + gutter)
            grid.paste(preview, (left, top))
    grid.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, default=Path('submission'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, help='Generate once. Omit to change seeds without reloading.')
    parser.add_argument('--out', type=Path, default=Path('private_build/demo'))
    args = parser.parse_args()
    folder = args.submission.resolve()
    files = check_files(folder)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    module = load_generator(folder)
    model = module.load_model(device)
    labels = torch.arange(2, device=device).repeat_interleave(8)
    module.generate(model, labels[:1], 0)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f'Model ready on {device}. Weights SHA-256 {files["file_sha256"]["weights.pt"]}.', flush=True)

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
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = perf_counter()
        images = module.generate(model, labels, seed).cpu()
        seconds = perf_counter() - start
        output = args.out / f'seed_{seed}.png'
        save_grid(images, seed, output)
        print(f'{seconds:.3f} s for 16 images. {output.resolve()} · SHA-256 {sha256(output)}', flush=True)
        if args.seed is not None:
            break


if __name__ == '__main__':
    main()

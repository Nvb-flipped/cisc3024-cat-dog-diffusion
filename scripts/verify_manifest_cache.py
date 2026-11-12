"""Compare an existing training cache with an independent official pool rebuild."""

import argparse
from pathlib import Path

import torch

from src.common import sha256, timestamp, write_json
from src.training.data import Pool, check_manifest, load_manifest_data, rebuild


def check_cache(manifest, pool_path, cache_dir):
    cache = cache_dir / f'v2_{sha256(manifest)}.npz'
    if not cache.is_file():
        raise FileNotFoundError(f'No existing cache for {manifest}: {cache}')
    cached_images, cached_labels, digest, stats = load_manifest_data(manifest, pool_path, cache_dir)
    pool = Pool(pool_path)
    checked = check_manifest(manifest, pool)
    if checked['errors']:
        raise ValueError(f'Official manifest check failed: {checked["errors"]}')
    official_images, official_labels = rebuild(checked['rows'], pool)
    expected = official_images.mul(2).sub(1)
    if not torch.equal(cached_labels, official_labels) or not torch.equal(cached_images, expected):
        raise ValueError(f'Cached labels or pixels differ from the official rebuild: {manifest}')
    return {'manifest': str(manifest), 'manifest_sha256': digest, 'rows': len(official_labels),
            'cat': stats['cat'], 'dog': stats['dog'], 'cache_sha256': sha256(cache),
            'pool_content_sha256': stats['pool_content_sha256'],
            'labels_equal': True, 'pixels_equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', type=Path, nargs='+')
    parser.add_argument('--pool', type=Path, default=Path('data/student'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Preserve the existing cache parity receipt: {args.out}')
    torch.set_num_threads(4)
    receipts = [check_cache(path, args.pool, args.cache_dir) for path in args.manifests]
    write_json(args.out, {'checked': timestamp(), 'rebuild': 'starter_kit.catdog_eval.data.rebuild',
                          'checks': receipts})
    print(f'Exact official reconstruction for {sum(row["rows"] for row in receipts)} cached crops')


if __name__ == '__main__':
    main()

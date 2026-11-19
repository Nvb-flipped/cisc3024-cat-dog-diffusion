import numpy as np

from src.curation.deduplicate import dhash, families, popcount


def crops(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)


def test_exact_mirror_and_near_copies_share_a_family_but_distinct_images_do_not():
    base = crops(4)
    near = np.clip(base[0].astype(int) + 1, 0, 255).astype(np.uint8)  # tiny recompression-like shift
    stack = np.stack([base[0], base[0], base[0][:, ::-1], near, base[1], base[2], base[3]])
    roots = families(stack, chunk=3)
    assert roots[0] == roots[1] == roots[2] == roots[3]
    assert len({roots[0], roots[4], roots[5], roots[6]}) == 4


def test_dhash_is_resolution_robust_and_popcount_counts_bits():
    image = np.repeat(np.repeat(crops(1, 3)[:, :8, :8], 4, axis=1), 4, axis=2)  # smooth blocky image
    brighter = np.clip(image.astype(int) + 10, 0, 255).astype(np.uint8)
    assert popcount(dhash(image) ^ dhash(brighter))[0] == 0
    assert popcount(np.array([np.uint64(0b1011)]))[0] == 3


def test_hash_collision_without_pixel_similarity_is_not_a_duplicate():
    import csv
    from pathlib import Path
    import pytest
    from src.common import sha256
    manifest = Path('manifests/v010.csv')
    cache = Path(f'data/cache/v2_{sha256(manifest)}.npz') if manifest.exists() else None
    if cache is None or not cache.exists():
        pytest.skip('local v010 cache not available')
    rows = list(csv.DictReader(manifest.open()))
    pixels = np.load(cache)['x']
    index = {(r['pool_id'], r['x0']): i for i, r in enumerate(rows)}
    pair = pixels[[index[('s038524', '82')], index[('s063930', '51')]]]  # dog vs cat, white studio
    assert popcount(dhash(pair[:1]) ^ dhash(pair[1:]))[0] <= 4
    roots = families(pair)
    assert roots[0] != roots[1]

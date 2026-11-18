"""Check that coarse appearance clusters reproduce on disjoint pool samples."""

import json
from pathlib import Path

import numpy as np

from src.common import sha256, write_json


def squared_distances(points, centres):
    return ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)


def kmeans(points, rng, clusters=3):
    best_centres, best_error = None, float('inf')
    for _ in range(16):
        centres = points[rng.choice(len(points), size=clusters, replace=False)]
        for _ in range(60):
            assignments = squared_distances(points, centres).argmin(axis=1)
            groups = [points[assignments == index] for index in range(clusters)]
            if any(len(group) == 0 for group in groups):
                break
            updated = np.stack([group.mean(axis=0) for group in groups])
            if np.array_equal(updated, centres):
                break
            centres = updated
        error = float(squared_distances(points, centres).min(axis=1).sum())
        if error < best_error:
            best_centres, best_error = centres, error
    return best_centres


def adjusted_rand(first, second, clusters=3):
    table = np.bincount(first * clusters + second, minlength=clusters ** 2).reshape(clusters, clusters)
    pairs = table * (table - 1) / 2
    row_counts, column_counts = table.sum(axis=1), table.sum(axis=0)
    row_pairs = (row_counts * (row_counts - 1) / 2).sum()
    column_pairs = (column_counts * (column_counts - 1) / 2).sum()
    expected = row_pairs * column_pairs / (len(first) * (len(first) - 1) / 2)
    denominator = (row_pairs + column_pairs) / 2 - expected
    return float((pairs.sum() - expected) / denominator) if denominator else 0.0


def main():
    manifest = Path('manifests/v009.csv')
    digest = sha256(manifest)
    cache = Path('data/cache') / f'v2_{digest}.npz'
    parity_path = Path('results/curation/v009_cache_parity.json')
    receipt = json.loads(parity_path.read_text(encoding='utf-8'))['checks'][0]
    if (receipt['manifest_sha256'] != digest or receipt['cache_sha256'] != sha256(cache)
            or not receipt['pixels_equal'] or not receipt['labels_equal']):
        raise ValueError('Only the verified exact training pixels are eligible')
    with np.load(cache, allow_pickle=False) as payload:
        images, labels = payload['x'].astype(np.float32) / 255, payload['y']
    grid = images.reshape(-1, 8, 4, 8, 4, 3).mean(axis=(2, 4))
    luminance = grid @ np.array([.299, .587, .114], dtype=np.float32)
    layout = (luminance - luminance.mean(axis=(1, 2), keepdims=True))[:, 1:7, 1:7].reshape(-1, 36) / 4
    colour = grid[:, 2:6, 2:6, :].mean(axis=(1, 2))
    features = np.concatenate([layout, colour], axis=1)
    comparisons = []
    for label in (0, 1):
        points = features[labels == label]
        for seed in (17, 29, 43):
            rng = np.random.default_rng(seed)
            left, right, check = np.array_split(points[rng.permutation(len(points))], 3)
            first, second = [squared_distances(check, kmeans(part, rng)).argmin(axis=1)
                             for part in (left, right)]
            min_fraction = min(float(np.bincount(index, minlength=3).min()) / len(check)
                               for index in (first, second))
            score = adjusted_rand(first, second)
            comparisons.append(dict(label=label, seed=seed, adjusted_rand=score,
                                    minimum_fraction=min_fraction, passes=score >= .75 and min_fraction >= .20))
    output = dict(manifest_sha256=digest, verified_cache_sha256=receipt['cache_sha256'],
                  source_sha256=sha256(Path(__file__)), numpy_version=np.__version__,
                  comparisons=comparisons, passes=all(row['passes'] for row in comparisons))
    write_json(Path('results/diagnostics/v009_appearance_modes.json'), output)
    print(output)


if __name__ == '__main__':
    main()

"""Immutable manifest-to-cache reconstruction using the official pool reader."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from src.common import ROOT, sha256, write_json

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, rebuild
from catdog_eval.manifest import check_manifest


def load_manifest_data(manifest, pool_path='data/student', cache_dir='data/cache'):
    manifest = Path(manifest)
    digest = sha256(manifest)
    pool = Pool(pool_path)
    checked = check_manifest(manifest, pool)
    if checked['errors']:
        raise ValueError(f'Invalid manifest: {checked["errors"]}')
    if not checked['rows']:
        raise ValueError('Empty training manifest')
    index_hash = sha256(Path(pool_path) / 'index.csv')
    source_files = sorted(Path(pool_path).glob('*.zip')) + sorted((Path(pool_path) / 'images').glob('*.jpg'))
    source_hashes = {str(path.relative_to(pool_path)): sha256(path) for path in source_files}
    source_hashes['index.csv'] = index_hash
    pool_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    checked['stats']['pool_content_sha256'] = pool_hash
    cache = Path(cache_dir) / f'v2_{digest}.npz'
    if cache.exists():
        sidecar = cache.with_suffix('.json')
        if not sidecar.exists() or json.loads(sidecar.read_text())['cache_sha256'] != sha256(cache):
            raise ValueError('Cache content hash mismatch; rebuild from the immutable manifest')
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored['manifest_sha256']) != digest or str(stored['pool_content_sha256']) != pool_hash:
                raise ValueError('Cache provenance mismatch')
            pixels, labels = stored['x'].copy(), stored['y'].copy()
        if pixels.shape != (len(checked['rows']), 32, 32, 3) or pixels.dtype != np.uint8:
            raise ValueError('Malformed image cache')
        expected_labels = np.array([r['label'] for r in checked['rows']], dtype=np.int64)
        if not np.array_equal(labels, expected_labels):
            raise ValueError('Cache label mismatch')
        images = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div(255)
        labels = torch.from_numpy(labels).long()
    else:
        images, labels = rebuild(checked['rows'], pool)
        pixels = images.mul(255).round().byte().permute(0, 2, 3, 1).numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=pixels, y=labels.numpy(),
                            manifest_sha256=digest, pool_content_sha256=pool_hash)
        write_json(cache.with_suffix('.json'), {'manifest_sha256': digest,
                   'pool_content_sha256': pool_hash, 'source_hashes': source_hashes, 'stats': checked['stats'],
                   'cache_sha256': sha256(cache), 'method': 'official catdog_eval.data.rebuild'})
    # Exact manifest pixels, no transformations or image augmentation.
    return images.mul(2).sub(1).contiguous(), labels, digest, checked['stats']

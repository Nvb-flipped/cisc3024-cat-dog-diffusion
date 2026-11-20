"""Immutable manifest-to-cache reconstruction using the official pool reader."""
import sys
import json
from pathlib import Path
import numpy as np
import torch
from src.common import ROOT, sha256, write_json
from src.curation.pool_fingerprint import fingerprint

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
    pool_hash, source_hashes = fingerprint(pool_path)
    provenance_path = manifest.with_suffix('.provenance.json')
    if not provenance_path.exists() and manifest.stem.endswith('.validation'):
        provenance_path = manifest.with_name(manifest.stem.removesuffix('.validation') + '.provenance.json')
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get('origin') in ['student_reviewed_manifest', 'subset_of_student_reviewed_manifest',
                                        'ai_assisted_reviewed_manifest']:
            if digest not in {record['sha256'] for record in provenance.get('manifests', [])}:
                raise ValueError('Manifest bytes differ from the frozen personal-review lineage')
            if provenance.get('pool_content_sha256') != pool_hash:
                raise ValueError('Source pool differs from the frozen personal-review manifest lineage')
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

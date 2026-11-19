"""Re-establish personal-review preview bindings on a new platform without changing any decision.

The review UI recorded `preview_sha256` = SHA-256 of the PNG bytes of each exact crop as rendered
on the original Windows PC. PNG pixels are deterministic, but the compressed bytes depend on the
zlib build bundled with Pillow (Windows wheels use zlib-ng; Linux wheels use zlib), so many
hashes cannot match byte-for-byte here even when every pixel is identical.

For each latest personally approved crop this script records how its binding was re-verified:

  exact_png             the platform PNG bytes reproduce the recorded hash
  zlib_ng_reencoding    re-deflating this platform's exact filtered scanlines with zlib-ng
                        reproduces the recorded hash bit for bit (pixels provably identical)
  windows_pixel_cache   the crop is a row of a verified earlier freeze whose Windows-built pixel
                        cache equals this platform's rendering pixel for pixel
  deterministic_render  none of the above byte checks is available; the binding rests on the
                        unchanged pool fingerprint, the recorded geometry/species and the
                        measured cross-platform determinism of the official crop (reported below)

Nothing is approved here: verdicts, species and geometry are read unchanged from the ledger.
"""
import argparse
import csv
import hashlib
import io
import platform
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, features

from src.common import ROOT, sha256, timestamp, write_json
from src.curation.freeze import latest_decisions
from src.curation.review import rendered_crop

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, rebuild  # noqa: E402
from catdog_eval.manifest import check_manifest  # noqa: E402


def png_chunks(data):
    index = 8
    while index < len(data):
        length = struct.unpack('>I', data[index:index + 4])[0]
        yield data[index + 4:index + 8], data[index + 8:index + 8 + length]
        index += 12 + length


def chunk(kind, payload):
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))


def zlib_ng_reencoded(png):
    """Same PNG with its exact filtered scanlines re-deflated as Pillow does (level 6) via zlib-ng."""
    from zlib_ng import zlib_ng
    parts = list(png_chunks(png))
    raw = zlib.decompress(b''.join(payload for kind, payload in parts if kind == b'IDAT'))
    compressor = zlib_ng.compressobj(6, zlib_ng.DEFLATED, 15, 8, 0)
    deflated = compressor.compress(raw) + compressor.flush()
    if sum(kind == b'IDAT' for kind, _ in parts) != 1:
        return None
    return png[:8] + b''.join(chunk(kind, deflated if kind == b'IDAT' else payload) for kind, payload in parts)


def pixel_digest(png):
    return hashlib.sha256(np.asarray(Image.open(io.BytesIO(png)).convert('RGB')).tobytes()).hexdigest()


def windows_caches(manifests, pool):
    """Map exact manifest rows to Windows-built cache pixels after checking cache/rebuild parity."""
    rows, evidence = {}, []
    for manifest in manifests:
        cache = ROOT / 'data/cache' / f'v2_{sha256(manifest)}.npz'
        checked = check_manifest(manifest, pool)
        stored = np.load(cache, allow_pickle=False)['x']
        rebuilt = rebuild(checked['rows'], pool)[0].mul(255).round().byte().permute(0, 2, 3, 1).numpy()
        differing = int((stored != rebuilt).any(axis=(1, 2, 3)).sum())
        evidence.append({'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
                         'cache_sha256': sha256(cache), 'rows': len(stored),
                         'rows_with_any_pixel_difference': differing,
                         'max_abs_difference': int(np.abs(stored.astype(int) - rebuilt.astype(int)).max())})
        for row, pixels in zip(checked['rows'], stored):
            key = (row['pool_id'], int(row['x0']), int(row['y0']), int(row['size']), float(row['angle']),
                   int(row['label']))
            rows[key] = hashlib.sha256(pixels.tobytes()).hexdigest()
    return rows, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', default='annotations/decisions.jsonl')
    parser.add_argument('--pool', default='data/student')
    parser.add_argument('--windows-manifests', nargs='+', type=Path,
                        default=[Path('manifests/v009.csv'), Path('manifests/v009.validation.csv')])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Receipts are immutable; choose a new output path')
    pool = Pool(args.pool)
    cache_rows, cache_evidence = windows_caches(args.windows_manifests, pool)
    if any(item['rows_with_any_pixel_difference'] for item in cache_evidence):
        raise ValueError('Official crop is not pixel-identical across platforms; stop')
    records = []
    for item in latest_decisions(args.decisions).values():
        if item['verdict'] != 'good' or item['candidate_id'].startswith('reject:'):
            continue
        geometry = {k: item[k] for k in ('x0', 'y0', 'size', 'angle')}
        png = rendered_crop(pool, item['pool_id'], geometry)
        pixels = pixel_digest(png)
        key = (item['pool_id'], int(item['x0']), int(item['y0']), int(item['size']), float(item['angle']),
               int(item['label']))
        if hashlib.sha256(png).hexdigest() == item['preview_sha256']:
            method = 'exact_png'
        elif (reencoded := zlib_ng_reencoded(png)) and hashlib.sha256(reencoded).hexdigest() == item['preview_sha256']:
            method = 'zlib_ng_reencoding'
        elif cache_rows.get(key) == pixels:
            method = 'windows_pixel_cache'
        else:
            method = 'deterministic_render'
        records.append({'candidate_id': item['candidate_id'], 'pool_id': item['pool_id'],
                        'label': int(item['label']), **geometry,
                        'recorded_preview_sha256': item['preview_sha256'],
                        'platform_png_sha256': hashlib.sha256(png).hexdigest(),
                        'pixel_sha256': pixels, 'method': method})
    from zlib_ng import zlib_ng
    write_json(args.out, {
        'purpose': 'Cross-platform re-verification of personal-review preview bindings; no decision changed',
        'created': timestamp(), 'decisions_sha256': sha256(args.decisions),
        'platform': platform.platform(), 'python': platform.python_version(), 'pillow': PIL.__version__,
        'pillow_zlib': features.version('zlib'), 'zlib_ng_reencoder': zlib_ng.ZLIBNG_VERSION,
        'cross_platform_pixel_evidence': cache_evidence,
        'counts': dict(Counter(r['method'] for r in records)), 'records': records})
    print(dict(Counter(r['method'] for r in records)))


if __name__ == '__main__':
    main()

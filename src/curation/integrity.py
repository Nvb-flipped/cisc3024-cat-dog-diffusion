"""Decode and hash every pool image without expanding the image shards."""
import argparse
import csv
import hashlib
import io
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from src.common import write_json


def scan_shard(path, index):
    rows = []
    with zipfile.ZipFile(path) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            pool_id = Path(entry.filename).stem
            row = {'pool_id': pool_id, 'shard': path.name, 'member': entry.filename,
                   'status': 'ok', 'reason': '', 'width': '', 'height': '',
                   'source_sha256': '', 'rgb_sha256': '', 'dhash': ''}
            try:
                if pool_id not in index:
                    raise ValueError('unindexed_member')
                raw = z.read(entry)  # ZipFile verifies the decompressed entry CRC.
                row['source_sha256'] = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    row['width'], row['height'] = im.size
                    if im.size != index[pool_id]:
                        raise ValueError(f'dimension_mismatch:{im.size}!={index[pool_id]}')
                    rgb = im.convert('RGB')  # No EXIF transpose: coordinates are as delivered.
                    row['rgb_sha256'] = hashlib.sha256(
                        f'{rgb.width},{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
                    pixels = list(rgb.convert('L').resize((9, 8)).getdata())
                    bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1]
                            for y in range(8) for x in range(8)]
                    row['dhash'] = f'{sum(int(b) << i for i, b in enumerate(bits)):016x}'
            except Exception as exc:
                row['status'], row['reason'] = 'bad', f'{type(exc).__name__}:{exc}'
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', default='data/student', type=Path)
    p.add_argument('--out', default='results/curation/integrity.csv', type=Path)
    p.add_argument('--workers', default=8, type=int)
    args = p.parse_args()
    with (args.pool / 'index.csv').open(newline='', encoding='utf-8') as f:
        raw_index = list(csv.DictReader(f))
    index = {r['pool_id']: (int(r['width']), int(r['height'])) for r in raw_index}
    if len(index) != len(raw_index):
        raise ValueError('Duplicate pool IDs in index')
    start = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_shard, shard, index): shard
                   for shard in sorted(args.pool.glob('*.zip'))}
        for future in as_completed(futures):
            batch = future.result()
            rows.extend(batch)
            print(f'{futures[future].name}: {len(batch)} images; total {len(rows)}', flush=True)
    counts = Counter(r['pool_id'] for r in rows)
    missing = sorted(set(index) - set(counts))
    duplicate_ids = sorted(k for k, v in counts.items() if v > 1)
    rows.sort(key=lambda r: (r['pool_id'], r['shard']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    good = [r for r in rows if r['status'] == 'ok']
    pixel_counts = Counter(r['rgb_sha256'] for r in good)
    summary = {'indexed_ids': len(index), 'entries': len(rows), 'decoded_ok': len(good),
               'bad': len(rows) - len(good), 'missing_ids': missing,
               'duplicate_ids': duplicate_ids,
               'exact_pixel_duplicate_excess': sum(n - 1 for n in pixel_counts.values()),
               'elapsed_seconds': time.perf_counter() - start,
               'pool_index_sha256': hashlib.sha256((args.pool / 'index.csv').read_bytes()).hexdigest(),
               'scope': 'pool only; CRC, full decode, delivered dimensions, RGB hashes; no EXIF reorientation'}
    write_json(args.out.with_suffix('.json'), summary)
    print(summary, flush=True)
    if missing or duplicate_ids or summary['bad']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

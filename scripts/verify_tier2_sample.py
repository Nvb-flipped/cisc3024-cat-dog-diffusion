"""SOP step 2: spot-check that older-run (Tier 2) deletions are really held by the migration backup.

    GH_TOKEN_FILE=/dev/shm/tok python -m scripts.verify_tier2_sample --manifest /tmp/snapshot-manifest.json --remote

Draws a seeded random sample from the Tier-2 deletion list. For each file it (1) hashes the local file and looks that hash up in
the backup manifest (path, size and SHA-256 must all agree) and, with --remote, (2) reads the member's BYTES out of the
release archive over HTTP range requests (no full 1.5 GB download), hashes them and compares with the local hash.
"""
import argparse
import hashlib
import io
import json
import os
import random
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import requests

from scripts.backup_kept_set import KEEP_EVERY, REPO, SKIP, session
from scripts.prune_checkpoints import file_sha256, plan


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a release asset; signed download URLs are re-resolved when they expire."""

    def __init__(self, sess, asset):
        self.sess, self.asset, self.size, self.pos, self.url, self.requests = sess, asset, asset['size'], 0, None, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _resolve(self):
        r = self.sess.get(self.asset['url'], headers={'Accept': 'application/octet-stream'}, allow_redirects=False)
        if r.status_code != 302:
            raise RuntimeError(f'asset download did not redirect: {r.status_code}')
        self.url = r.headers['Location']

    def readinto(self, buffer):
        want = min(len(buffer), self.size - self.pos)
        if want <= 0:
            return 0
        for attempt in range(4):
            if self.url is None:
                self._resolve()
            r = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + want - 1}'}, timeout=120)
            self.requests += 1
            if r.status_code in (200, 206) and len(r.content) == want:
                buffer[:want] = r.content
                self.pos += want
                return want
            self.url = None                       # expired or refused: resolve a fresh signed URL and retry
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f'range read failed at {self.pos}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True)
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--seed', type=int, default=2026112101)
    p.add_argument('--remote', action='store_true')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    by_sha = {e['sha256']: e for e in manifest['files']}
    _, ledger = plan(Namespace(keep_every=KEEP_EVERY, skip=SKIP, live_minutes=45, legacy_backup_manifest=args.manifest))
    tier2 = [e for e in ledger if 'backup_archive' in e]
    print(f'Tier-2 deletion list: {len(tier2)} files, {sum(e["bytes"] for e in tier2) / 1e9:.1f} GB; sample seed {args.seed}')
    sample = random.Random(args.seed).sample(tier2, args.n)
    sess = session() if args.remote else None
    assets = {a['name']: a for a in json.loads(Path('/tmp/release_assets.json').read_text())} if args.remote else {}
    all_ok = True
    for i, entry in enumerate(sample, 1):
        path = Path(entry['path'])
        local = file_sha256(path)
        hit = by_sha.get(local)
        ok = bool(hit) and hit['path'] == entry['path'] and hit['size'] == entry['bytes'] and hit['sha256'] == local
        print(f'\n[{i}] {entry["path"]}\n    local size {entry["bytes"]:,} B, local sha256 {local}'
              f'\n    manifest: {"path/size/hash AGREE" if ok else "MISMATCH"}'
              f'{" in " + hit["archive"] if hit else ""}')
        remote_ok = None
        if args.remote and hit:
            meta = sess.get(f'https://api.github.com/repos/{REPO}/releases/assets/{assets[hit["archive"]]["id"]}').json()
            raw = HttpRangeFile(sess, meta)
            with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=8 << 20)) as archive:
                digest, started = hashlib.sha256(), time.time()
                with archive.open(hit['path']) as member:
                    for block in iter(lambda: member.read(1 << 22), b''):
                        digest.update(block)
            remote_ok = digest.hexdigest() == local
            print(f'    release bytes: sha256 {digest.hexdigest()} -> {"IDENTICAL to local file" if remote_ok else "DIFFERENT"}'
                  f' ({raw.requests} range requests, {time.time() - started:.0f}s)')
        all_ok &= ok and (remote_ok is not False)
    print('\nRESULT:', 'all sampled files verified' if all_ok else 'PROBLEM FOUND - do not delete Tier 2')


if __name__ == '__main__':
    main()

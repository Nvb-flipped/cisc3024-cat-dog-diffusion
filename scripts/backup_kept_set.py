"""Back up the final model and the checkpoints that SURVIVE pruning to a private GitHub release (SOP step 1).

    python -m scripts.backup_kept_set build  --out private_build/linux_era_backup_20261121
    GH_TOKEN_FILE=/dev/shm/tok python -m scripts.backup_kept_set upload --out ... --tag linux-era-backup-2026-11-21
    GH_TOKEN_FILE=/dev/shm/tok python -m scripts.backup_kept_set verify --out ... --tag ...

build   packs submission/ (the final model) and every file that stays in the Linux-era run folders after
        `scripts.prune_checkpoints` into stored (uncompressed) ZIP parts below the 2 GiB asset limit, hashes every
        file while writing, then re-reads each archive and checks every member against the manifest.
upload  creates the private release (if needed), uploads each asset with retries and compares GitHub's own
        SHA-256 for the asset with the local SHA-256.
verify  downloads the small final-model asset again and checks the weights inside it against the local file.
The GitHub token is read from the file named by GH_TOKEN_FILE (never from the command line or the repository).
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import requests

from scripts.prune_checkpoints import CKPT, LINUX_ERA, plan
from src.common import timestamp, write_json

REPO = 'Nvb-flipped/Pattern-Recognition-Project'
PART_LIMIT = 1_400_000_000
KEEP_EVERY = ['v014_ema9995_flip=10000', 'v014_flip=20000']
SKIP = ['v014_wide_ema9995_flip', 'v014_mid_ema9995_flip']          # runs still training; backed up later
FINAL_ZIP = 'final-model-v014-ema9995-80k.zip'
CHUNK = 1 << 24


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def kept_files():
    args = Namespace(keep_every=KEEP_EVERY, skip=SKIP, live_minutes=45, legacy_backup_manifest=None)
    rows, ledger = plan(args)
    delete = {Path(e['path']).as_posix() for e in ledger}
    skipped = {r[0] for r in rows if r[5].startswith('SKIPPED')}
    kept = []
    for directory in sorted(p for p in CKPT.iterdir() if p.is_dir() and LINUX_ERA.match(p.name)):
        if directory.name.removesuffix('_seed17') in skipped:
            continue
        kept += [f for f in sorted(directory.rglob('*')) if f.is_file() and f.as_posix() not in delete]
    return kept, sorted(skipped)


def write_zip(path, items):
    """items: [(source Path, archive name)] -> stored ZIP; returns manifest rows with the hash of every member."""
    rows = []
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, name in items:
            info = zipfile.ZipInfo.from_file(source, name)
            info.compress_type = zipfile.ZIP_STORED
            digest = hashlib.sha256()
            with archive.open(info, 'w', force_zip64=True) as target, open(source, 'rb') as stream:
                for block in iter(lambda: stream.read(CHUNK), b''):
                    digest.update(block)
                    target.write(block)
            rows.append({'path': name, 'size': source.stat().st_size, 'sha256': digest.hexdigest(), 'archive': path.name})
    return rows


def verify_zip(path, rows):
    expected = {r['path']: r for r in rows}
    with zipfile.ZipFile(path) as archive:
        names = [i.filename for i in archive.infolist()]
        if sorted(names) != sorted(expected):
            raise ValueError(f'{path.name}: member list differs from the manifest')
        for info in archive.infolist():
            digest = hashlib.sha256()
            with archive.open(info) as member:           # reading to the end also checks the stored CRC-32
                for block in iter(lambda: member.read(CHUNK), b''):
                    digest.update(block)
            if digest.hexdigest() != expected[info.filename]['sha256'] or info.file_size != expected[info.filename]['size']:
                raise ValueError(f'{path.name}: {info.filename} differs from the manifest')


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    kept, skipped = kept_files()
    manifest = {'created': timestamp(), 'purpose': 'Backup of the final model and of the Linux-era checkpoints kept after pruning',
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'excluded_runs_still_training': skipped, 'archives': [], 'files': []}
    final_items = [(f, f.as_posix()) for f in sorted(Path('submission').iterdir()) if f.is_file()]
    groups, current, size = [], [], 0
    for f in kept:
        if current and size + f.stat().st_size > PART_LIMIT:
            groups.append(current)
            current, size = [], 0
        current.append(f)
        size += f.stat().st_size
    if current:
        groups.append(current)
    jobs = [(out / FINAL_ZIP, final_items)] + [(out / f'kept-set.part{i:03d}.zip', [(f, f.as_posix()) for f in g])
                                                for i, g in enumerate(groups, 1)]
    for path, items in jobs:
        started = time.time()
        rows = write_zip(path, items)
        verify_zip(path, rows)
        manifest['files'] += rows
        manifest['archives'].append({'name': path.name, 'size': path.stat().st_size, 'sha256': sha256_file(path), 'members': len(rows)})
        print(f"{path.name}: {len(rows)} files, {path.stat().st_size / 1e9:.2f} GB, verified member by member ({time.time() - started:.0f}s)", flush=True)
    write_json(out / 'kept-set-manifest.json', manifest)
    print('files', len(manifest['files']), 'bytes', sum(r['size'] for r in manifest['files']))


def session():
    s = requests.Session()
    s.headers.update({'Authorization': 'Bearer ' + Path(os.environ['GH_TOKEN_FILE']).read_text().strip(),
                      'Accept': 'application/vnd.github+json'})
    return s


def get_release(s, tag, create):
    r = s.get(f'https://api.github.com/repos/{REPO}/releases/tags/{tag}')
    if r.status_code == 200:
        return r.json()
    if not create:
        raise SystemExit(f'release {tag} not found')
    body = ('Private backup made BEFORE pruning intermediate checkpoints (21 Nov 2026). Contains the final model '
            '(submission/ of v014 EMA 0.9995, 80k updates) and the Linux-era (v010-v014) checkpoints and receipts that remain '
            'after pruning, as stored ZIP parts. kept-set-manifest.json lists every file with its SHA-256. '
            'Older runs are in release migration-2026-11-19. Keep this repository private.')
    r = s.post(f'https://api.github.com/repos/{REPO}/releases', json={
        'tag_name': tag, 'target_commitish': 'main', 'name': 'Linux-era checkpoint backup (kept set + final model), 2026-11-21',
        'body': body, 'draft': False, 'prerelease': False})
    r.raise_for_status()
    return r.json()


def remote_assets(s, release_id):
    r = s.get(f'https://api.github.com/repos/{REPO}/releases/{release_id}/assets', params={'per_page': 100})
    r.raise_for_status()
    return {a['name']: a for a in r.json()}


def upload(out, tag):
    s = session()
    release = get_release(s, tag, create=True)
    rid = release['id']
    files = sorted(p for p in out.iterdir() if p.suffix in ('.zip', '.json'))
    for path in files:
        local = sha256_file(path)
        for attempt in range(1, 6):
            existing = remote_assets(s, rid).get(path.name)
            if existing and existing['state'] == 'uploaded' and existing.get('digest') == f'sha256:{local}' and existing['size'] == path.stat().st_size:
                print(f'{path.name}: already uploaded and verified', flush=True)
                break
            if existing:
                s.delete(f"https://api.github.com/repos/{REPO}/releases/assets/{existing['id']}")
            started = time.time()
            ctype = 'application/zip' if path.suffix == '.zip' else 'application/json'
            try:
                with open(path, 'rb') as stream:
                    r = s.post(f'https://uploads.github.com/repos/{REPO}/releases/{rid}/assets', params={'name': path.name},
                               headers={'Content-Type': ctype}, data=stream, timeout=(30, 7200))
                ok = r.status_code == 201
            except requests.RequestException as exc:
                ok, r = False, exc
            rate = path.stat().st_size / max(time.time() - started, 1) / 1e6
            print(f'{path.name}: attempt {attempt}, {"ok" if ok else "FAILED " + str(getattr(r, "status_code", r))[:80]}, {rate:.1f} MB/s', flush=True)
            if ok:
                remote = remote_assets(s, rid)[path.name]
                if remote.get('digest') == f'sha256:{local}' and remote['size'] == path.stat().st_size:
                    break
                print(f'{path.name}: remote digest/size mismatch, retrying', flush=True)
            time.sleep(10 * attempt)
        else:
            raise SystemExit(f'giving up on {path.name}')
    print('upload complete:', release['html_url'])


def verify(out, tag):
    """Download the final-model asset again and compare the weights inside it with the local file."""
    s = session()
    release = get_release(s, tag, create=False)
    asset = remote_assets(s, release['id'])[FINAL_ZIP]
    r = s.get(asset['url'], headers={'Accept': 'application/octet-stream'}, stream=True)
    r.raise_for_status()
    target = out / 'redownloaded_final_model.zip'
    with open(target, 'wb') as f:
        for block in r.iter_content(CHUNK):
            f.write(block)
    with zipfile.ZipFile(target) as archive:
        inside = hashlib.sha256(archive.read('submission/weights.pt')).hexdigest()
    result = {'downloaded_asset_sha256': sha256_file(target), 'github_digest': asset['digest'],
              'weights_inside_downloaded_zip_sha256': inside, 'local_submission_weights_sha256': sha256_file(Path('submission/weights.pt'))}
    target.unlink()
    print(json.dumps(result, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('build', 'upload', 'verify'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tag', default='linux-era-backup-2026-11-21')
    args = p.parse_args()
    {'build': lambda: build(args.out), 'upload': lambda: upload(args.out, args.tag), 'verify': lambda: verify(args.out, args.tag)}[args.action]()


if __name__ == '__main__':
    main()

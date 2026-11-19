"""Stage a migration restore from whichever snapshot archives are already present.

Same safety rules as restore_migration.py (archive SHA-256, safe paths, per-file hash
verification, never overwriting a different existing file), but it restores only from the
archives named on the command line, so large downloads can be restored incrementally.
Existing files whose bytes differ from the snapshot (newer Git work) are kept and listed.
Run restore_migration.py afterwards, once every archive is present, for full parity.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', required=True, help='Archive names to restore from')
    args = parser.parse_args()
    target = args.target.resolve()
    manifest = json.loads((args.archives / 'snapshot-manifest.json').read_text(encoding='utf-8'))
    archives = {item['name']: item for item in manifest['archives']}
    records = {r['path']: r for r in manifest['files']}
    kept_different, restored, present = [], 0, 0
    for name in args.parts:
        item = archives[name]
        path = args.archives / name
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError(f'Invalid archive: {path}')
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                record = records.get(member.filename)
                if not record or record['archive'] != name:
                    raise ValueError(f'Unlisted archive entry: {member.filename}')
                pure = PurePosixPath(member.filename)
                destination = (target / member.filename).resolve()
                if pure.is_absolute() or '..' in pure.parts or not destination.is_relative_to(target):
                    raise ValueError('Unsafe snapshot path')
                if destination.exists():
                    if sha(destination) == record['sha256']:
                        present += 1
                    else:
                        kept_different.append(member.filename)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + '.migration_tmp')
                with archive.open(member) as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                if sha(temporary) != record['sha256']:
                    temporary.unlink()
                    raise ValueError(f'Restored file hash mismatch: {member.filename}')
                temporary.replace(destination)
                restored += 1
        print(f'Restored from {name}', flush=True)
    print(json.dumps({'restored': restored, 'already_identical': present,
                      'kept_existing_different': kept_different}, indent=1))


if __name__ == '__main__':
    main()

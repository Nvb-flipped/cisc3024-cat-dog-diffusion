"""Restore a verified private migration snapshot using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archives',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--apply',action='store_true',help='Restore missing files; refuse conflicting existing files')
    args=parser.parse_args();target=args.target.resolve()
    manifest=json.loads((args.archives/'snapshot-manifest.json').read_text(encoding='utf-8'))
    records={r['path']:r for r in manifest['files']}
    for item in manifest['archives']:
        path=args.archives/item['name']
        if path.stat().st_size!=item['size'] or sha(path)!=item['sha256']:raise ValueError(f'Invalid archive: {path}')
        print('Verified archive '+path.name,flush=True)
    for relative, record in records.items():
        pure=PurePosixPath(relative)
        destination=(target/relative).resolve()
        if pure.is_absolute() or '..' in pure.parts or not destination.is_relative_to(target):raise ValueError('Unsafe snapshot path')
        if destination.exists() and (not destination.is_file() or sha(destination)!=record['sha256']):
            raise ValueError(f'Existing file differs; preserve it and restore into a clean clone: {destination}')
    if not args.apply:return print('Archives verified; use --apply to restore')
    for item in manifest['archives']:
        with zipfile.ZipFile(args.archives/item['name']) as archive:
            for member in archive.infolist():
                record=records.get(member.filename)
                if not record or record['archive']!=item['name']:raise ValueError('Unlisted archive entry')
                destination=(target/member.filename).resolve()
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    temporary=destination.with_name(destination.name+'.migration_tmp')
                    with archive.open(member) as source, temporary.open('xb') as output:shutil.copyfileobj(source,output,8*1024*1024)
                    if sha(temporary)!=record['sha256']:raise ValueError('Restored file hash mismatch')
                    temporary.replace(destination)
        print('Restored '+item['name'],flush=True)
    for relative,record in records.items():
        if sha(target/relative)!=record['sha256']:raise ValueError(f'Final parity failed: {relative}')
    print(f'COMPLETE: {len(records)} files verified. Recreate virtual environments and run the handoff checks.')


if __name__=='__main__':main()

"""Reproduce local layout, hash course inputs, and verify immutable starter code."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_starter():
    errors = []
    with zipfile.ZipFile(ROOT / 'Course_Project_Resources/starter_kit.zip') as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            dest = ROOT / item.filename
            if not dest.exists() or hashlib.sha256(z.read(item)).hexdigest() != sha256(dest):
                errors.append(item.filename)
    if errors:
        raise RuntimeError(f'Official files differ: {errors}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    verify_starter()
    if args.verify_only:
        print('Official starter kit matches archive exactly.')
        return
    target = ROOT / 'data/student'
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    sources = sorted((ROOT / 'student').glob('*'))
    sources += sorted((ROOT / 'Course_Project_Resources').glob('*.zip'))
    sources += [ROOT / 'project_handout.pdf', ROOT / 'Project Briefing.pptx']
    for source in sources:
        if not source.is_file():
            continue
        digest = sha256(source)
        hashes[str(source.relative_to(ROOT)).replace('\\', '/')] = {
            'sha256': digest, 'bytes': source.stat().st_size}
        if source.parent.name == 'student':
            dest = target / source.name
            if dest.exists():
                if sha256(dest) != digest:
                    raise RuntimeError(f'Existing data copy differs: {dest}')
            else:
                shutil.copy2(source, dest)
                if sha256(dest) != digest:
                    raise RuntimeError(f'Copy verification failed: {dest}')
        print(source.name, source.stat().st_size, flush=True)
    out = ROOT / 'results/setup/source_hashes.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print('Inputs copied and verified; image ZIP shards remain compressed.')


if __name__ == '__main__':
    main()

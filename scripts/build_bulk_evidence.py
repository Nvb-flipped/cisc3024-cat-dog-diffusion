"""Build the separate bulk-evidence asset and the manifest that lets the course ZIP stay under the 100 MB upload limit.

    python -m scripts.build_bulk_evidence --out private_build/bulk_curation_logs.zip

The files listed in src.release.package.BULK_OMITTED (raw per-image detector output and the candidate/screening tables derived
from it) are the largest curation intermediates. They are not in the course ZIP; this script packs them into one asset for the
GitHub release and writes results/release/omitted_bulk_files.json (path, bytes, SHA-256 of every file, and of the asset), which
is part of the ZIP. The asset is built once and never overwritten, so its recorded hash stays valid.
"""
import argparse
import json
import zipfile
from pathlib import Path

from src.common import ROOT, sha256
from src.release.package import BULK_MANIFEST, BULK_OMITTED, DEADLINE_EPOCH

NOTE = """Raw curation intermediates of the CISC3024 cat/dog project (the files omitted from the course ZIP to respect its
100 MB limit). Extract this archive inside the unzipped project folder: the paths are relative to the project root.
Every file's SHA-256 is listed in results/release/omitted_bulk_files.json inside the course ZIP.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'private_build/bulk_curation_logs.zip')
    parser.add_argument('--release', default='final-submission')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'{out} exists: the asset is built once so that its recorded hash stays valid')
    files = []
    for relative in BULK_OMITTED:
        path = ROOT / relative
        if path.stat().st_mtime > DEADLINE_EPOCH:
            raise SystemExit(f'{relative} is dated after the deadline')
        files.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        bundle.writestr('README.txt', NOTE)
        for item in files:
            bundle.write(ROOT / item['path'], item['path'])
    with zipfile.ZipFile(out) as bundle:
        assert bundle.testzip() is None
    record = {'reason': 'The course ZIP must stay under the 100 MB upload limit; these are the largest raw curation intermediates.',
              'asset': out.name, 'asset_sha256': sha256(out), 'asset_bytes': out.stat().st_size,
              'published_with_release': args.release,
              'restore': 'unzip the asset inside the project root; every path below is relative to it',
              'files': files}
    (ROOT / BULK_MANIFEST).write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}, indent=1), [f['path'] for f in files])


if __name__ == '__main__':
    main()

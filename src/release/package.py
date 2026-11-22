"""Build the course ZIP from an explicit allowlist after structural release checks."""
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from src.common import ROOT, sha256, timestamp, write_json
from src.release.readiness import readiness

ROOT_FILES = {'README.md', 'DATA_CARD.md', 'AGENTS.md', 'AGENT_LOG.md',
              'requirements.txt', 'requirements-lock.txt', 'pytest.ini', '.gitattributes',
              'THIRD_PARTY_NOTICES.md', 'PROJECT_PLAN.md', 'REQUIREMENTS_TRACE.md', 'STUDENT_REVIEW_QUEUE.md',
              'demo.py', 'requirements-demo.txt', 'EXPERIMENTS.md'}
SOURCE_ROOTS = {'src', 'scripts', 'tests', 'configs', 'manifests', 'annotations', 'logs', 'results', 'figures', 'report', 'student_checks'}
SOURCE_EXTENSIONS = {'.py', '.js', '.html', '.css', '.yaml', '.yml', '.json', '.jsonl', '.csv',
                     '.txt', '.log', '.md', '.tex', '.bib', '.sty', '.cls', '.xml', '.drawio'}
INFERENCE_FILES = {'__init__.py', 'generate.py', 'model.py', 'diffusion.py', 'edm.py', 'flow.py', 'weights.pt',
                   'inference.json', 'manifest.csv', 'provenance.json'}
PRESENTATION_EXTENSIONS = {'.pptx', '.pdf', '.tex', '.sty', '.cls', '.json', '.csv', '.txt', '.md', '.bib', '.mjs'}
# Course deadline: Thursday 26 November 2026 23:55 Asia/Shanghai = 15:55 UTC. Every file in the archive must be older.
DEADLINE_EPOCH = 1795708500  # 2026-11-26T15:55:00Z
FORBIDDEN_PARTS = {'data', 'student', '.venvs', '.git', '.agents', '.codex', 'models',
                   'checkpoints', 'private_build', 'tmp', '__pycache__', '.pytest_tmp', '.pytest_cache'}


def allowed(relative):
    relative = Path(relative)
    parts = relative.parts
    if not parts or relative.is_absolute() or '..' in parts or any(p in FORBIDDEN_PARTS for p in parts):
        return False
    if len(parts) == 1:
        return relative.name in ROOT_FILES
    if parts[0] == 'submission':
        return len(parts) == 2 and relative.name in INFERENCE_FILES
    if parts[0] == 'audit':
        return len(parts) == 2 and relative.name in {'audit.csv', 'provenance.json', 'summary.json'}
    if parts[0] == 'presentation':
        return relative.suffix.lower() in PRESENTATION_EXTENSIONS
    if parts[0] in SOURCE_ROOTS:
        return relative.suffix.lower() in SOURCE_EXTENSIONS
    return False


def selected_files(root):
    root = Path(root).resolve()
    files = []
    candidates = []
    for directory, directories, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        directories[:] = sorted(name for name in directories if name not in FORBIDDEN_PARTS
                                and (parent != root or name in SOURCE_ROOTS | {'submission', 'audit', 'presentation'}))
        for name in directories:
            if (parent / name).is_symlink() or (parent / name).resolve() != (parent / name).absolute():
                raise ValueError(f'Archive directory is a filesystem link: {parent / name}')
        candidates.extend(parent / name for name in names)
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if not allowed(relative) or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Archive member escapes project root: {relative}')
        if path.suffix.lower() in ['.xml', '.drawio', '.html', '.json', '.jsonl']:
            text = path.read_text(encoding='utf-8', errors='strict').lower()
            if 'data:image' in text or '<image' in text or 'image=data:' in text:
                continue  # Keep photographic source drawings only in the complete local package.
        files.append(path)
    return files


def build_archive(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Archive exists; keep upload artifacts immutable')
    checks = readiness(root)
    if not checks['structural_checks_passed']:
        raise ValueError('Release is not ready for packaging: ' + '; '.join(checks['issues']))
    files = selected_files(root)
    history = subprocess.check_output(['git', 'log', '--date=iso-strict', '--pretty=format:%h %ad %s'],
                                     cwd=root, text=True)
    hashes = {str(path.relative_to(root)).replace('\\', '/'): sha256(path) for path in files}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, str(path.relative_to(root)).replace('\\', '/'))
        z.writestr('GIT_HISTORY.txt', history + '\n')
        z.writestr('ARTIFACT_HASHES.json', json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        late = [i.filename for i in z.infolist() if time.mktime(tuple(i.date_time) + (0, 0, -1)) > DEADLINE_EPOCH]
        if late:
            raise ValueError(f'Archive members dated after the submission deadline: {late[:5]}')
        for item in z.namelist():
            if item not in ['GIT_HISTORY.txt', 'ARTIFACT_HASHES.json'] and not allowed(Path(item)):
                raise ValueError(f'Forbidden archive member: {item}')
        for name, expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f'Archive content hash mismatch: {name}')
    temporary.replace(out)
    receipt = {'archive': str(out), 'sha256': sha256(out), 'bytes': out.stat().st_size,
        'created': timestamp(), 'members': len(hashes) + 2,
        'report_uploaded_separately': str(root / 'report/report.pdf'),
        'scope': 'Local archive only; no course upload performed. Clean-room execution and human final review remain required.'}
    write_json(root / 'results/release/package_receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=ROOT, type=Path)
    p.add_argument('--out', default='private_build/project_submission.zip', type=Path)
    p.add_argument('--list', action='store_true', help='Review the exact allowlist without creating an archive')
    args = p.parse_args()
    if args.list:
        for path in selected_files(args.root):
            print(path.relative_to(args.root))
    else:
        print(build_archive(args.root, args.out))


if __name__ == '__main__':
    main()

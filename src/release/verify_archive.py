"""Extract the actual course ZIP and run its inference with separately installed tools/data."""
import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from src.common import ROOT, sha256, timestamp, write_json
from src.release.package import allowed, check_archive_size
from scripts.setup_inputs import verify_starter

EXTRA_MEMBERS = {'ARTIFACT_HASHES.json', 'GIT_HISTORY.txt'}


def validate_members(archive):
    """Check all members and their hashes before extracting or executing anything."""
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError('Duplicate or case-colliding archive members')
        for info in bundle.infolist():
            name = info.filename
            relative = PurePosixPath(name)
            if '\\' in name or relative.is_absolute() or '..' in relative.parts or ':' in name or \
               stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f'Unsafe archive member: {name}')
            if name not in EXTRA_MEMBERS and not allowed(Path(name)):
                raise ValueError(f'Archive contains a forbidden member: {name}')
        hashes = json.loads(bundle.read('ARTIFACT_HASHES.json'))
        if set(hashes) != set(names) - EXTRA_MEMBERS:
            raise ValueError('Archive hash inventory does not match its actual members')
        if bundle.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        for name, expected in hashes.items():
            content = bundle.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Archived content hash mismatch: {name}')
            if Path(name).suffix.lower() in ['.xml', '.drawio', '.html', '.json', '.jsonl']:
                text = content.decode('utf-8').lower()
                if 'data:image' in text or '<image' in text or 'image=data:' in text:
                    raise ValueError(f'Embedded image in archived text: {name}')
        return hashes


def verify_archive(archive, extract_to, *, course_root=ROOT, devices=('cpu', 'cuda'),
                   counts='1,2,20,50,500', official_seed=None, expected_sha256=None):
    archive, extract_to, course_root = map(lambda path: Path(path).resolve(), [archive, extract_to, course_root])
    digest = sha256(archive)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('Archive differs from the expected upload artifact hash')
    if extract_to.exists():
        raise FileExistsError('Clean-room directory must be new; preserve earlier verification evidence')
    check_archive_size(archive)
    validate_members(archive)
    extract_to.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extract_to)
    evidence_folder = extract_to / 'private_build/archive_verification'
    evidence_folder.mkdir(parents=True)
    evidence = {'archive': str(archive), 'archive_sha256': digest, 'extracted_root': str(extract_to),
                'started': timestamp(), 'devices': {}, 'official_evaluation': None,
                'scope': 'Archive execution check, not a new human audit or hidden-test result.'}
    for device in devices:
        result_path = evidence_folder / f'offline_{device}.json'
        command = [sys.executable, '-m', 'src.release.check', '--submission', str(extract_to / 'submission'),
                   '--device', device, '--counts', counts, '--out', str(result_path)]
        with (evidence_folder / f'offline_{device}.log').open('x', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=extract_to, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Clean-room {device} inference failed; inspect {evidence_folder}')
        record = json.loads(result_path.read_text())
        if record['file_sha256'] != {name.removeprefix('submission/'): value
                                     for name, value in validate_members(archive).items() if name.startswith('submission/')}:
            raise ValueError('Extracted inference evidence does not match archived submission files')
        evidence['devices'][device] = {'result_sha256': sha256(result_path), 'command': command}
    if official_seed is not None:
        # The supplied kit and evaluator weights are separate installations, not ZIP members.
        verify_starter(course_root)
        shutil.copytree(course_root / 'starter_kit', extract_to / 'starter_kit',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        resource_folder = extract_to / 'Course_Project_Resources'
        resource_folder.mkdir()
        shutil.copy2(course_root / 'Course_Project_Resources/starter_kit.zip', resource_folder / 'starter_kit.zip')
        result_path = evidence_folder / f'official_seed_{official_seed}.json'
        command = [sys.executable, '-m', 'src.evaluation.run', '--submission', str(extract_to / 'submission'),
            '--out', str(result_path), '--data-root', str(course_root / 'data'),
            '--metric-cache', str(course_root / 'models/evaluator'), '--seed', str(official_seed)]
        with (evidence_folder / 'official.log').open('x', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=extract_to, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Clean-room official evaluation failed; inspect {evidence_folder}')
        evidence['official_evaluation'] = {'result_sha256': sha256(result_path), 'command': command}
    evidence['finished'] = timestamp()
    write_json(evidence_folder / 'receipt.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--extract-to', required=True, type=Path)
    parser.add_argument('--course-root', default=ROOT, type=Path)
    parser.add_argument('--devices', default='cpu,cuda')
    parser.add_argument('--counts', default='1,2,20,50,500')
    parser.add_argument('--official-seed', type=int, help='Rerun the full unchanged official evaluator after extraction')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive, args.extract_to, course_root=args.course_root,
        devices=args.devices.split(','), counts=args.counts, official_seed=args.official_seed,
        expected_sha256=args.expected_sha256), indent=2))


if __name__ == '__main__':
    main()

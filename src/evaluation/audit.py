"""Prepare the unchanged official audit safely and bind judgments to its exact sample."""
import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from src.common import ROOT, sha256, timestamp, write_json
from src.curation.calibration import wilson


def official_audit():
    spec = importlib.util.spec_from_file_location('course_official_audit', ROOT / 'starter_kit/audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(row):
    return {'pool_id': row['pool_id'].strip(), 'x0': int(row['x0']), 'y0': int(row['y0']),
            'size': int(row['size']), 'angle': float(row['angle'] or 0), 'label': int(row['label'])}


def expected_sample(manifest, student_id):
    if not str(student_id).strip():
        raise ValueError('Actual student ID is required for the official sample')
    with Path(manifest).open(newline='', encoding='utf-8-sig') as f:
        rows = [canonical(row) for row in csv.DictReader(f)]
    if len(rows) < 200:
        raise ValueError('The official audit requires at least 200 final manifest rows')
    return official_audit().sample_rows(rows, str(student_id), 200)


def verify_sample(path, manifest, student_id):
    """Recompute the official selection; metadata alone cannot establish sample identity."""
    path, manifest = Path(path), Path(manifest)
    provenance = json.loads(path.with_name('provenance.json').read_text(encoding='utf-8'))
    if provenance['manifest_sha256'] != sha256(manifest) or provenance['student_id'] != str(student_id):
        raise ValueError('Audit is stale for this final manifest or student ID')
    if provenance['official_tool_sha256'] != sha256(ROOT / 'starter_kit/audit.py'):
        raise ValueError('Official audit tool changed after sample preparation')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    expected = expected_sample(manifest, student_id)
    if len(rows) != 200 or [canonical(row) for row in rows] != expected:
        raise ValueError('Audit rows differ from the exact official 200-row sample')
    if [int(row['index']) for row in rows] != list(range(200)):
        raise ValueError('Audit indices differ from the official ordering')
    return rows


def prepare(manifest, student_id, data_root, out):
    manifest, out = Path(manifest).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Audit directory already exists; preserve personal judgments and use a new version')
    expected_sample(manifest, student_id)
    digest = sha256(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='audit_prepare_', dir=out.parent) as temporary:
        staged = Path(temporary) / 'audit'
        command = [sys.executable, str(ROOT / 'starter_kit/audit.py'), '--manifest', str(manifest),
                   '--student-id', str(student_id), '--n', '200', '--data-root', str(Path(data_root).resolve()),
                   '--out', str(staged)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Unchanged official audit failed: ' + result.stdout + result.stderr)
        if sha256(manifest) != digest:
            raise ValueError('Manifest changed during audit preparation')
        write_json(staged / 'provenance.json', {'student_id': str(student_id), 'manifest_sha256': digest,
            'official_tool_sha256': sha256(ROOT / 'starter_kit/audit.py'), 'created': timestamp(),
            'command': command, 'initial_csv_sha256': sha256(staged / 'audit.csv'),
            'sheet_sha256': {p.name: sha256(p) for p in sorted(staged.glob('sheet_*.png'))},
            'ownership': 'Verdicts and bad-row reasons must be personally supplied by the student.'})
        verify_sample(staged / 'audit.csv', manifest, student_id)
        shutil.move(str(staged), str(out))
    return out / 'audit.csv'


def summarize(path, manifest, student_id):
    rows = verify_sample(path, manifest, student_id)
    for row in rows:
        verdict = row['verdict'].strip().lower()
        if verdict not in ['good', 'bad'] or (verdict == 'bad' and not row['reason'].strip()):
            raise ValueError('All 200 personal verdicts and every bad-row reason are required')
    def statistics(subset):
        count = len(subset)
        good = sum(row['verdict'].strip().lower() == 'good' for row in subset)
        return {'n': count, 'good': good, 'precision': good / count if count else None,
                'wilson_95': wilson(good, count) if count else None}
    return {'status': 'complete_csv_and_exact_official_sample', 'audit_sha256': sha256(path),
            'manifest_sha256': sha256(manifest), 'overall': statistics(rows),
            'by_class': {name: statistics([r for r in rows if int(r['label']) == label])
                         for label, name in [(0, 'cat'), (1, 'dog')]},
            'scope': 'Checks sample identity and CSV completeness; authentic personal judgment remains the student responsibility.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'summarize'])
    parser.add_argument('--manifest', default='submission/manifest.csv', type=Path)
    parser.add_argument('--student-id', required=True)
    parser.add_argument('--data-root', default='data', type=Path)
    parser.add_argument('--out', default='audit', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.manifest, args.student_id, args.data_root, args.out))
    else:
        report = summarize(args.out / 'audit.csv', args.manifest, args.student_id)
        write_json(args.out / 'summary.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

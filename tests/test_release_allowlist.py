import csv
from pathlib import Path
import pytest
from src.release.package import (BULK_OMITTED, MAX_ARCHIVE_BYTES, allowed, check_archive_size, check_bulk_manifest, selected_files)
from src.release.readiness import audit_check


@pytest.mark.parametrize('name', ['data/student/index.csv', 'student/index.csv', '.git/config',
    'checkpoints/run.pt', 'submission/alternate.pt', 'submission/image.png', 'results/grid.png',
    'figures/example.svg', 'report/fig_audit.pdf', 'report/report_draft.pdf', 'report/old/report.pdf', '.venvs/training/anything.py', '../secret.txt',
    'results/release/package_receipt.json', *BULK_OMITTED])
def test_course_archive_excludes_prohibited_material(name):
    assert not allowed(name)


@pytest.mark.parametrize('name', ['submission/weights.pt', 'submission/generate.py', 'audit/audit.csv',
    'src/training/train.py', 'figures/F0/figure.drawio', 'figures/F0/source.csv',
    'presentation/presentation.pptx', 'presentation/presentation.pdf', 'report/report.pdf', 'report/report.tex', 'README.md', 'demo.py',
    'requirements-demo.txt', 'EXPERIMENTS.md', 'results/release/omitted_bulk_files.json',
    'results/curation/full/duplicate_candidates.csv'])
def test_course_archive_retains_required_material(name):
    assert allowed(name)


def test_embedded_photos_are_not_smuggled_in_xml(tmp_path):
    figures = tmp_path / 'figures'
    figures.mkdir()
    (figures / 'photo.drawio').write_text('<mxfile>data:image/png;base64,fixture</mxfile>')
    (figures / 'vector.drawio').write_text('<mxfile><diagram/></mxfile>')
    assert [p.name for p in selected_files(tmp_path)] == ['vector.drawio']


def test_audit_completeness_requires_every_bad_reason(tmp_path):
    path = tmp_path / 'audit.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['verdict', 'reason'])
        writer.writeheader()
        writer.writerows([{'verdict': 'good', 'reason': ''} for _ in range(199)] + [{'verdict': 'bad', 'reason': ''}])
    with pytest.raises(ValueError, match='lacks reason'):
        audit_check(path)
    text = path.read_text()
    path.write_text(text.replace('bad,', 'bad,fixture reason'))
    assert audit_check(path)['rows'] == 200


def test_archive_deadline_is_26_november_2026_2355_shanghai():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.release.package import DEADLINE_EPOCH
    zone = ZoneInfo('Asia/Shanghai')
    assert datetime.fromtimestamp(DEADLINE_EPOCH, zone) == datetime(2026, 11, 26, 23, 55, tzinfo=zone)


def test_archive_size_ceiling(tmp_path):
    assert MAX_ARCHIVE_BYTES == 100_000_000
    path = tmp_path / 'archive.zip'
    path.write_bytes(b'x' * 10)
    assert check_archive_size(path, limit=10) == 10
    with pytest.raises(ValueError, match='above the 9-byte upload limit'):
        check_archive_size(path, limit=9)


def test_omitted_bulk_files_are_listed_and_unchanged():
    record = check_bulk_manifest(Path(__file__).resolve().parents[1])
    assert sorted(item['path'] for item in record['files']) == sorted(BULK_OMITTED)
    assert record['asset_bytes'] > 0 and len(record['asset_sha256']) == 64

"""Synthetic audit fixtures verify identity checks without claiming human review."""
import csv
import pytest
from src.common import sha256, write_json, ROOT
from src.evaluation.audit import expected_sample, verify_sample, summarize, prepare


def fixture_audit(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    fields = ['pool_id', 'x0', 'y0', 'size', 'angle', 'label']
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(dict(pool_id=f'fixture{i:04d}', x0=0, y0=0, size=64, angle=0, label=i % 2)
                         for i in range(320))
    path = tmp_path / 'audit.csv'
    rows = expected_sample(manifest, 'fixture-id')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, ['index'] + fields + ['verdict', 'reason'])
        writer.writeheader()
        writer.writerows(dict(index=i, **row, verdict='good', reason='synthetic fixture')
                         for i, row in enumerate(rows))
    write_json(tmp_path / 'provenance.json', {'student_id': 'fixture-id',
        'manifest_sha256': sha256(manifest), 'official_tool_sha256': sha256(ROOT / 'starter_kit/audit.py')})
    return path, manifest


def test_recomputed_official_sample_and_summary(tmp_path):
    path, manifest = fixture_audit(tmp_path)
    assert len(verify_sample(path, manifest, 'fixture-id')) == 200
    result = summarize(path, manifest, 'fixture-id')
    assert result['overall']['precision'] == 1
    assert sum(s['n'] for s in result['by_class'].values()) == 200


def test_edited_audit_geometry_cannot_keep_old_provenance(tmp_path):
    path, manifest = fixture_audit(tmp_path)
    path.write_text(path.read_text().replace(',0,0,64,', ',1,0,64,', 1))
    with pytest.raises(ValueError, match='exact official'):
        verify_sample(path, manifest, 'fixture-id')


def test_changed_manifest_or_student_invalidates_audit(tmp_path):
    path, manifest = fixture_audit(tmp_path)
    with pytest.raises(ValueError, match='stale'):
        verify_sample(path, manifest, 'different-id')
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='stale'):
        verify_sample(path, manifest, 'fixture-id')


def test_prepare_never_overwrites_personal_decisions(tmp_path):
    out = tmp_path / 'existing_audit'
    out.mkdir()
    (out / 'audit.csv').write_text('preserve fixture')
    with pytest.raises(FileExistsError, match='preserve personal'):
        prepare('unused.csv', 'fixture-id', 'unused', out)
    assert (out / 'audit.csv').read_text() == 'preserve fixture'

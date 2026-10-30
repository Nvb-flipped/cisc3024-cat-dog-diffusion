"""Synthetic geometry-only manifests test the declared data-quantity ablation."""
import csv
import json
import pytest
from src.common import sha256, write_json
from src.curation.subset import balanced_subset, COLUMNS


class MetadataPool:
    def __contains__(self, identifier):
        return identifier.startswith('synthetic_')

    def size(self, identifier):
        return 120, 120


def parent_bundle(tmp_path, per_class=600):
    parent = tmp_path / 'parent.csv'
    rows = [dict(pool_id=f'synthetic_{label}_{index:04d}', x0=20, y0=20, size=60, angle=0, label=label)
            for label in [0, 1] for index in range(per_class)]
    held = parent.with_suffix('.validation.csv')
    for path, subset in [(parent, rows), (held, [dict(rows[0], pool_id='synthetic_holdout')])]:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, COLUMNS)
            writer.writeheader()
            writer.writerows(subset)
    write_json(parent.with_suffix('.provenance.json'), {'origin': 'student_reviewed_manifest',
        'pool_content_sha256': 'synthetic-pool-fingerprint',
        'scope': 'SYNTHETIC UNIT FIXTURE; not an actual student review',
        'manifests': [{'sha256': sha256(parent)}, {'sha256': sha256(held)}],
        'candidate_ids': [f'fixture-{i}' for i in range(len(rows))]})
    return parent


def test_subset_is_balanced_deterministic_and_keeps_exact_holdout_bytes(tmp_path):
    parent = parent_bundle(tmp_path)
    a, b = tmp_path / 'subset_a.csv', tmp_path / 'subset_b.csv'
    balanced_subset(parent, a, MetadataPool())
    balanced_subset(parent, b, MetadataPool())
    assert a.read_bytes() == b.read_bytes()
    assert a.with_suffix('.validation.csv').read_bytes() == parent.with_suffix('.validation.csv').read_bytes()
    with a.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1000 and sum(row['label'] == '0' for row in rows) == 500
    assert len({row['pool_id'] for row in rows}) == 1000
    provenance = json.loads(a.with_suffix('.provenance.json').read_text())
    assert len(provenance['candidate_ids']) == 1000 and provenance['fresh_initialization_required']


def test_no_size_contrast_is_omitted_instead_of_duplicating_rows(tmp_path):
    parent = parent_bundle(tmp_path, 500)
    with pytest.raises(ValueError, match='No meaningful quantity contrast'):
        balanced_subset(parent, tmp_path / 'subset.csv', MetadataPool())


def test_changed_parent_or_holdout_refuses_subset(tmp_path):
    parent = parent_bundle(tmp_path)
    held = parent.with_suffix('.validation.csv')
    held.write_text(held.read_text() + '\n')
    with pytest.raises(ValueError, match='holdout changed'):
        balanced_subset(parent, tmp_path / 'subset.csv', MetadataPool())
    parent.write_text(parent.read_text() + '\n')
    with pytest.raises(ValueError, match='parent manifest changed'):
        balanced_subset(parent, tmp_path / 'subset.csv', MetadataPool())

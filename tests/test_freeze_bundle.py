"""A failed official geometry check must not leave a half-published manifest."""
import csv
import hashlib
import json
import importlib
import pytest

freeze_module = importlib.import_module('src.curation.freeze')


def inputs(tmp_path, monkeypatch):
    decisions = tmp_path / 'decisions.jsonl'
    rows = [dict(candidate_id=f'fixture{i}', pool_id=f'pool{i}', label=i % 2,
                 verdict='good', origin='student_browser_review', reviewer='Automated fixture only',
                 timestamp='fixture timestamp', criteria=dict.fromkeys(freeze_module.CRITERIA, True),
                 head_width=42, occupancy=.70, x0=20, y0=20, size=60, angle=0,
                 preview_sha256=hashlib.sha256(f'fixture pixels pool{i}'.encode()).hexdigest()) for i in range(24)]
    decisions.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    groups = tmp_path / 'groups.csv'
    with groups.open('w', newline='') as f:
        writer = csv.DictWriter(f, ['candidate_id','group_id','keep','reviewer','timestamp','origin','crop_review_sha256'])
        writer.writeheader()
        writer.writerows(dict(candidate_id=row['candidate_id'], group_id=f'family{i}', keep='yes',
                              reviewer='Automated fixture only', timestamp='fixture timestamp',
                              origin='student_duplicate_review', crop_review_sha256=freeze_module.crop_review_identity(row))
                         for i, row in enumerate(rows))
    monkeypatch.setattr(freeze_module, 'Pool', lambda _: object())
    monkeypatch.setattr(freeze_module, 'fingerprint', lambda _: ('synthetic-pool-fingerprint', {}))
    monkeypatch.setattr(freeze_module, 'rendered_crop', lambda pool, identifier, row: f'fixture pixels {identifier}'.encode())
    return dict(pool='fixture', decisions=str(decisions), validation_fraction=.05, seed=3024), groups


def test_second_manifest_validation_failure_publishes_nothing(tmp_path, monkeypatch):
    config, groups = inputs(tmp_path, monkeypatch)
    calls = []
    def check(path, pool):
        calls.append(path)
        return {'errors': ['synthetic failure'] if len(calls) == 2 else [], 'stats': {}, 'warnings': []}
    monkeypatch.setattr(freeze_module, 'check_manifest', check)
    out = tmp_path / 'manifests/v001.csv'
    with pytest.raises(ValueError, match='Official manifest check failed'):
        freeze_module.freeze(config, out, groups_path=groups)
    assert list(out.parent.iterdir()) == []


def test_success_publishes_consistent_bundle_and_refuses_existing_provenance(tmp_path, monkeypatch):
    config, groups = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(freeze_module, 'check_manifest', lambda *args: {'errors': [], 'stats': {}, 'warnings': []})
    out = tmp_path / 'manifests/v001.csv'
    frozen = freeze_module.freeze(config, out, groups_path=groups)
    provenance = json.loads(out.with_suffix('.provenance.json').read_text())
    assert provenance['manifests'] == frozen
    for item in frozen:
        assert freeze_module.sha256(item['path']) == item['sha256']
    # Even provenance by itself reserves the immutable version.
    out.unlink()
    out.with_suffix('.validation.csv').unlink()
    with pytest.raises(ValueError, match='immutable'):
        freeze_module.freeze(config, out, groups_path=groups)


def test_rounded_legacy_occupancy_cannot_mask_out_of_range_width(tmp_path, monkeypatch):
    config, groups = inputs(tmp_path, monkeypatch)
    decisions = tmp_path / 'decisions.jsonl'
    rows = [json.loads(line) for line in decisions.read_text().splitlines()]
    rows[0].update(head_width=48.2, occupancy=.80)
    decisions.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    out = tmp_path / 'manifests/v001.csv'
    with pytest.raises(ValueError, match='Recheck source width or crop occupancy for fixture0'):
        freeze_module.freeze(config, out, groups_path=groups)
    assert not out.exists()


def test_expansion_keeps_exact_parent_rows_and_holdout(tmp_path, monkeypatch):
    config, groups = inputs(tmp_path, monkeypatch)

    def check(path, pool):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ('x0', 'y0', 'size', 'label'):
                row[key] = int(row[key])
            row['angle'] = float(row['angle'])
        return {'errors': [], 'rows': rows, 'stats': {}, 'warnings': []}

    monkeypatch.setattr(freeze_module, 'check_manifest', check)
    decisions = tmp_path / 'decisions.jsonl'
    records = decisions.read_text().splitlines()
    decisions.write_text('\n'.join(records[:-2]) + '\n')
    parent, expanded = tmp_path / 'parent.csv', tmp_path / 'expanded.csv'
    freeze_module.freeze(config, parent, groups_path=groups)
    decisions.write_text('\n'.join(records) + '\n')
    freeze_module.freeze(config, expanded, groups_path=groups, preserve_split_from=parent)
    assert parent.with_suffix('.validation.csv').read_bytes() == expanded.with_suffix('.validation.csv').read_bytes()
    assert set(expanded.read_text().splitlines()) - set(parent.read_text().splitlines()) == {
        'pool22,20,20,60,0,0', 'pool23,20,20,60,0,1'}
    assert set(parent.read_text().splitlines()).issubset(expanded.read_text().splitlines())
    decisions.write_text('\n'.join(records[1:]) + '\n')
    with pytest.raises(ValueError, match='Parent row changed or is no longer kept'):
        freeze_module.freeze(config, tmp_path / 'invalid.csv', groups_path=groups, preserve_split_from=parent)

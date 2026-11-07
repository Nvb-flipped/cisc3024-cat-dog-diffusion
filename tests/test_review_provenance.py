import hashlib
import json
from PIL import Image
import pytest
from src.curation.review import CRITERIA, ReviewStore, parse_geometry, rendered_crop
from src.curation.duplicates import HammingTree
from src.curation.freeze import split_by_group


class FixturePool:
    def image(self, pool_id):
        return Image.new('RGB', (120, 120), (80, 120, 160))


@pytest.fixture
def store(tmp_path):
    candidates = tmp_path / 'candidates.csv'
    candidates.write_text('candidate_id,pool_id,label,x0,y0,size,angle\nfixture,source,0,20,20,60,0\n')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'proposed_candidate_ids': ['fixture'], 'rejected_pool_ids': ['rejected']}))
    return ReviewStore(candidates, selection, tmp_path / 'test_only_decisions.jsonl')


def decision():
    geometry = {'x0': 20, 'y0': 20, 'size': 60, 'angle': 0}
    return {'candidate_id': 'fixture', 'reviewer': 'Automated fixture, not a student check',
            'verdict': 'good', 'reason': '', **geometry, 'label': 0, 'head_width': 42,
            'occupancy': 0.70, 'criteria': dict.fromkeys(CRITERIA, True),
            'preview_sha256': hashlib.sha256(rendered_crop(FixturePool(), 'source', geometry)).hexdigest()}


def test_review_requires_explicit_complete_checks_and_current_preview(store):
    data = decision()
    event = store.validate(data, FixturePool())
    assert event['origin'] == 'student_browser_review'
    store.save(event)
    assert store.items('calibration')[0]['prior']['verdict'] == 'good'
    for field, value in [('reviewer', ''), ('preview_sha256', 'stale'),
                         ('head_width', 31), ('occupancy', 0.9), ('criteria', {})]:
        invalid = decision() | {field: value}
        with pytest.raises(ValueError):
            store.validate(invalid, FixturePool())


def test_bad_requires_concrete_reason_and_log_preserves_revisions(store):
    data = decision() | {'verdict': 'bad'}
    with pytest.raises(ValueError):
        store.validate(data, FixturePool())
    good = store.validate(decision(), FixturePool())
    bad = store.validate(data | {'reason': 'Fixture reason, not an actual image judgment'}, FixturePool())
    store.save(good)
    store.save(bad)
    assert len(store.path.read_text().splitlines()) == 2
    assert store.latest['fixture']['verdict'] == 'bad'


def test_draft_requires_personal_confirmation_and_retains_measurement_origin(store):
    data = decision()
    del data['occupancy']
    data['head_width'] = 44
    data['head_width_basis'] = 'agent_draft_confirmed'
    store.drafts['fixture'] = {'origin': 'agent_review_draft', 'head_width': 44,
                               'preview_sha256': data['preview_sha256']}
    with pytest.raises(ValueError, match='personally checked'):
        store.validate(data, FixturePool())
    event = store.validate(data | {'draft_confirmed': True}, FixturePool())
    assert event['occupancy'] == 44 / 60
    assert event['head_width_basis'] == 'agent_draft_confirmed'
    assert event['agent_draft_personally_checked'] is True
    assert len(event['agent_draft_sha256']) == 64
    with pytest.raises(ValueError, match='original draft crop'):
        store.validate(data | {'draft_confirmed': True, 'head_width': 45}, FixturePool())


def test_fractional_geometry_is_not_silently_truncated():
    with pytest.raises(ValueError):
        parse_geometry(dict(x0=20.5, y0=20, size=60, angle=0))


def test_rejected_source_review_has_distinct_semantics(store):
    data = {'candidate_id': 'reject:rejected', 'reviewer': 'Automated fixture',
            'verdict': 'missed_head', 'reason': ''}
    with pytest.raises(ValueError):
        store.validate(data, FixturePool())
    event = store.validate(data | {'verdict': 'correct_rejection'}, FixturePool())
    assert 'label' not in event and 'x0' not in event


def test_hamming_tree_matches_brute_force():
    values = [0, 1, 3, 7, 255, 256, 65535, 0]
    tree = HammingTree()
    for i, value in enumerate(values):
        tree.add(value, i)
    for query in [0, 17, 256, 65535]:
        expected = sorted((i, (value ^ query).bit_count()) for i, value in enumerate(values)
                          if (value ^ query).bit_count() <= 3)
        assert sorted(tree.find(query, 3)) == expected


def test_holdout_is_stratified_and_group_disjoint():
    rows = [{'candidate_id': str(i), 'label': i % 2} for i in range(200)]
    groups = {str(i): f'family-{i // 4}-{i % 2}' for i in range(200)}
    train, validation = split_by_group(rows, groups, 0.05, 3024)
    assert {groups[r['candidate_id']] for r in train}.isdisjoint(
        {groups[r['candidate_id']] for r in validation})
    assert {r['label'] for r in validation} == {0, 1}
    assert split_by_group(rows, groups, 0.05, 3024) == (train, validation)

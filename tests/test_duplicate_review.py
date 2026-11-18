"""Synthetic colors test duplicate-review plumbing, never real student judgments."""
import csv
import hashlib
import json
import pytest
from PIL import Image
from src.curation.review import CRITERIA, parse_geometry, rendered_crop
from src.curation.duplicate_review import DuplicateReview


class SyntheticPool:
    def __init__(self):
        self.colors = {'a': (120, 80, 40), 'b': (120, 80, 40), 'c': (40, 80, 120)}

    def image(self, identifier):
        return Image.new('RGB', (120, 120), self.colors[identifier])


@pytest.fixture
def store(tmp_path):
    pool = SyntheticPool()
    decisions = []
    for identifier in 'abc':
        row = dict(candidate_id=identifier, pool_id=identifier, label=int(identifier == 'c'),
            x0=20, y0=20, size=60, angle=0, verdict='good', reviewer='Automated synthetic fixture only',
            timestamp='fixture time', origin='student_browser_review', criteria=dict.fromkeys(CRITERIA, True))
        row['preview_sha256'] = hashlib.sha256(rendered_crop(pool, identifier, parse_geometry(row))).hexdigest()
        decisions.append(row)
    path = tmp_path / 'test_only_crop_decisions.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in decisions))
    integrity = tmp_path / 'integrity.csv'
    integrity.write_text('pool_id,status,dhash\na,ok,0000000000000000\nb,ok,0000000000000000\nc,ok,0000000000000000\n')
    return DuplicateReview(pool, path, output=tmp_path / 'queue', integrity=integrity,
                           groups=tmp_path / 'groups.csv', log=tmp_path / 'test_only_duplicate_decisions.jsonl')


def choices(store):
    queue = store.queue()
    return {'reviewer': 'Automated synthetic fixture only', 'personally_compared': True,
        'snapshot_sha256': queue['snapshot_sha256'],
        'rows': [dict(candidate_id=identifier, group_id='cats' if identifier != 'c' else 'dog',
                      keep='no' if identifier == 'b' else 'yes', reason='synthetic duplicate fixture' if identifier == 'b' else '')
                 for identifier in 'abc']}


def test_similarity_does_not_create_verdicts(store):
    queue = store.queue()
    assert queue['components'] == [['a', 'b', 'c']]
    assert all(item['prior'] is None for item in queue['items'])
    assert not store.groups.exists() and not store.log.exists()


def test_large_comparison_groups_keep_all_members_with_at_most_eight_per_page():
    from src.curation.duplicate_review import review_pages
    members = [f'head-{i:02}' for i in range(19)]
    pages = review_pages([members, ['single-a'], ['single-b']])
    assert [len(page['ids']) for page in pages] == [8, 8, 3, 2]
    assert [key for page in pages for key in page['ids']] == members + ['single-a', 'single-b']
    assert [page['component_part'] for page in pages[:3]] == [1, 2, 3]
    assert all(page['component_parts'] == 3 for page in pages[:3])
    assert pages[-1]['singleton'] is True


def test_draft_inventory_never_promotes_unreviewed_or_edited_crops(store, monkeypatch, tmp_path):
    import src.curation.duplicate_review as module
    reviewed = [json.loads(line) for line in store.decisions.read_text().splitlines()]
    rows = [dict(row) for row in reviewed]
    rows[1]['size'] += 1
    rows.append(dict(rows[0], candidate_id='unreviewed-draft'))
    catalog = tmp_path / 'results/curation/expansion_duplicate_advisory_v050.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'batch_count': 1, 'batches': [{'batch': 1, 'items': rows}]}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    before = store.decisions.read_bytes()
    batch = store.batch_advisory()['batches'][0]
    assert [row['review_status'] for row in batch['items']] == ['good', 'changed', 'good', 'pending']
    assert store.decisions.read_bytes() == before
    assert not store.groups.exists() and not store.log.exists()
    assert 'unreviewed-draft' not in {row['candidate_id'] for row in store.queue()['items']}


def test_personal_choices_materialize_replayable_groups(store):
    assert store.save(choices(store))['saved'] == 3
    with store.groups.open(newline='') as stream:
        groups = list(csv.DictReader(stream))
    assert len(groups) == 3 and sum(row['keep'] == 'yes' for row in groups) == 2
    assert all(len(row['crop_review_sha256']) == 64 for row in groups)
    assert all(item['prior'] for item in store.queue()['items'])


def test_identical_pixels_cannot_be_kept_in_distinct_families(store):
    data = choices(store)
    for row in data['rows']:
        row.update(group_id=row['candidate_id'], keep='yes', reason='')
    with pytest.raises(ValueError, match='identical crop pixels'):
        store.save(data)
    assert not store.log.exists()


def test_crop_or_species_change_invalidates_family_review(store):
    store.save(choices(store))
    rows = [json.loads(line) for line in store.decisions.read_text().splitlines()]
    rows[0]['label'] = 1
    with store.decisions.open('a') as stream:
        stream.write(json.dumps(rows[0]) + '\n')
    queue = store.queue()
    assert queue['items'][0]['prior'] is None
    assert all(item['prior'] for item in queue['items'][1:])


def test_changed_pixels_fail_before_any_family_event(store):
    data = choices(store)
    store.pool.colors['a'] = (255, 0, 0)
    with pytest.raises(ValueError, match='pixels changed'):
        store.save(data)
    assert not store.log.exists()


def test_manual_csv_edit_is_preserved_and_not_partially_logged(store):
    data = choices(store)
    store.save(data)
    before = store.log.read_bytes()
    store.groups.write_text(store.groups.read_text() + '# synthetic manual edit\n')
    with pytest.raises(ValueError, match='edited manually'):
        store.save(data)
    assert store.log.read_bytes() == before
    assert '# synthetic manual edit' in store.groups.read_text()


def test_no_approval_or_partial_comparison_cannot_be_saved(store):
    data = choices(store)
    with pytest.raises(ValueError, match='confirm personal'):
        store.save(data | {'personally_compared': False})
    with pytest.raises(ValueError, match='every crop'):
        store.save(data | {'rows': data['rows'][:1]})


def test_keep_pending_singletons_preserves_existing_choices(store, monkeypatch):
    monkeypatch.setattr('src.curation.duplicate_review.suggested_components',
                        lambda items, features: ([['a'], ['b'], ['c']], []))
    data = choices(store)
    queue = store.queue()
    assert queue['pages'] == [{'ids': ['a', 'b', 'c'], 'singleton': True}]
    assert store.save(data | {'rows': [data['rows'][1]]})['saved'] == 1
    prior_exclusion = store.latest['b'].copy()
    rows = [dict(candidate_id=identifier, group_id=f'distinct:{identifier}', keep='yes', reason='')
            for identifier in ('a', 'c')]
    bulk = data | {'keep_pending_singletons': True, 'rows': rows}
    with pytest.raises(ValueError, match='pending crops changed'):
        store.save(bulk | {'rows': rows[:1]})
    assert store.save(bulk)['saved'] == 2
    assert store.latest['b'] == prior_exclusion
    with pytest.raises(ValueError, match='pending crops changed'):
        store.save(bulk)


def test_singleton_button_cannot_skip_a_suggested_comparison(store):
    data = choices(store)
    rows = [dict(candidate_id=identifier, group_id=f'distinct:{identifier}', keep='yes', reason='')
            for identifier in ('a', 'b', 'c')]
    with pytest.raises(ValueError, match='pending crops changed'):
        store.save(data | {'keep_pending_singletons': True, 'rows': rows})
    assert not store.log.exists()

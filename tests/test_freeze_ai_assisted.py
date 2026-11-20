"""Resolution rules for VLM curation decisions (plan section 5)."""
import json

from src.curation.freeze_ai_assisted import geometry_ok, resolve


def record(resolved, species='cat'):
    """A realistic v5 verdict whose recomputed resolution is `resolved`."""
    verdict = {'species': species, 'is_real_photo': True, 'heads_or_faces_visible': 1, 'view': 'head_closeup',
               'head_width_share': 'over_80', 'both_ears_fully_inside': True, 'head_clearly_visible': True,
               'upright': True, 'centered': True, 'edited_or_costume': False}
    if resolved == 'REJECT':
        verdict['is_real_photo'] = False
    if resolved == 'UNCERTAIN':
        verdict['species'] = 'uncertain'
    return {'resolved': 'stale-field-is-ignored', 'verdict': verdict}


def test_agreeing_or_single_pass_is_accepted_with_its_species():
    assert resolve(record('PASS', 'dog'), None) == ('ACCEPT', 'dog')
    assert resolve(record('PASS', 'cat'), record('PASS', 'cat')) == ('ACCEPT', 'cat')
    assert resolve(record('UNCERTAIN', 'uncertain'), record('PASS', 'dog')) == ('ACCEPT', 'dog')


def test_conflicts_and_disagreements_are_never_auto_accepted():
    assert resolve(record('PASS', 'cat'), record('PASS', 'dog'))[0] == 'UNRESOLVED_SPECIES'
    assert resolve(record('PASS'), record('REJECT'))[0] == 'UNRESOLVED_CONFLICT'
    assert resolve(record('REJECT'), record('PASS'))[0] == 'UNRESOLVED_CONFLICT'
    assert resolve(record('UNCERTAIN'), None)[0] == 'PENDING_ESCALATION'
    assert resolve(record('REJECT'), None)[0] == 'REJECT'
    assert resolve(None, None)[0] == 'NOT_SCREENED'


def test_geometry_requires_width_occupancy_and_bounds():
    row = {'x0': 10, 'y0': 10, 'size': 100, 'angle': 0, 'box': json.dumps([25, 20, 95, 95])}
    assert geometry_ok(row, 200, 200)
    assert not geometry_ok(row | {'box': json.dumps([25, 20, 50, 95])}, 200, 200)  # 25 px wide head
    assert not geometry_ok(row | {'box': json.dumps([30, 30, 70, 70])}, 200, 200)  # 40% occupancy
    assert not geometry_ok(row, 105, 200)  # square leaves the source

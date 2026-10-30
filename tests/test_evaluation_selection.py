import pytest
from src.evaluation.sweep import score_results


def result(accuracy=.93, fid=45, recall=.45, marks=15, memorization=.01, speed=30):
    return {'violations': [], 'marks': {'valid': True, 'total': marks},
        'metrics': {'accuracy': accuracy, 'fid': fid, 'recall': recall,
                    'kid': .01, 'kid_std': .001, 'precision': .8,
                    'memorization_rate': memorization, 'num_samples': 2000},
        'training_data': {'stats': {'valid_rows': 1000}},
        'resources': {'seconds_per_1000': speed, 'num_parameters': 10_664_371, 'submission_mb': 41}}


def test_ranking_uses_worst_seed_tier_then_balanced_margin():
    robust = score_results([result(), result(fid=46)])
    fragile = score_results([result(fid=40), result(accuracy=.89, marks=14)])
    assert robust > fragile
    assert robust[0] == 15
    assert robust[1] == pytest.approx(.4)
    assert score_results([result(memorization=.005)]) > score_results([result(memorization=.02)])


def test_missing_memorization_invalid_checks_and_small_screen_fail_selection():
    assert score_results([result(memorization=None)]) is None
    invalid = result()
    invalid['violations'] = ['bad manifest']
    assert score_results([invalid]) is None
    screen = result()
    screen['metrics']['num_samples'] = 500
    with pytest.raises(ValueError):
        score_results([screen])


@pytest.mark.parametrize('key,value', [('kid', float('nan')), ('precision', None),
    ('memorization_rate', .051), ('recall', 1.1), ('kid_std', -.1)])
def test_incomplete_or_invalid_quality_evidence_cannot_win(key, value):
    invalid = result()
    invalid['metrics'][key] = value
    assert score_results([invalid]) is None


def test_nonfinite_speed_or_empty_rebuild_cannot_win():
    assert score_results([result(speed=float('nan'))]) is None
    invalid = result()
    invalid['training_data']['stats']['valid_rows'] = 0
    assert score_results([invalid]) is None

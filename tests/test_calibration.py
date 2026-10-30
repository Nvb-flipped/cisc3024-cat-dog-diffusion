import pytest
from src.curation.calibration import original_is_good, wilson


def test_wilson_extremes_keep_uncertainty():
    low, high = wilson(200, 200)
    assert low == pytest.approx(.98115467, abs=1e-7) and high == 1
    low, high = wilson(0, 200)
    assert low == 0 and high > 0
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_correcting_species_or_geometry_is_not_raw_proposal_success():
    candidate = dict(label='1', x0='20', y0='20', size='60', angle='0')
    decision = dict(verdict='good', label=1, x0=20, y0=20, size=60, angle=0)
    assert original_is_good(candidate, decision)
    assert not original_is_good(candidate, decision | {'label': 0})
    assert not original_is_good(candidate, decision | {'size': 70})
    assert not original_is_good(candidate, decision | {'verdict': 'uncertain'})

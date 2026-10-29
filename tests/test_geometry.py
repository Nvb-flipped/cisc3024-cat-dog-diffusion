import numpy as np
import pytest
from PIL import Image
from catdog_eval.data import crop as official_crop, Pool
from catdog_eval.manifest import _square_corners
from src.curation.geometry import crop, inside, proposal


@pytest.mark.parametrize('angle', [0, -45, -17.5, 23, 45])
@pytest.mark.parametrize('mode', ['RGB', 'L', 'RGBA'])
def test_crop_parity_on_coordinate_fixture(angle, mode):
    y, x = np.mgrid[:180, :240]
    pixels = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype('uint8')
    image = Image.fromarray(pixels).convert(mode)
    assert np.array_equal(np.array(crop(image, 50, 40, 80, angle)),
                          np.array(official_crop(image, 50, 40, 80, angle)))


@pytest.mark.parametrize('geometry', [(0, 0, 40, 0), (10, 15, 40, 45),
                                     (190, 10, 60, 0), (20, 20, 40, -33)])
def test_boundary_convention_matches_official(geometry):
    x0, y0, size, angle = geometry
    corners = _square_corners(*geometry)
    expected = all(-0.5 <= x <= 240.5 and -0.5 <= y <= 180.5 for x, y in corners)
    assert inside(240, 180, x0, y0, size, angle) == expected


def test_proposals_reject_small_heads_and_never_pad():
    assert proposal((40, 40, 71, 100), 200, 200)[1] == 'head_width_below_32'
    assert proposal((0, 0, 80, 80), 200, 200)[1] == 'square_outside_source'
    valid, reason = proposal((60, 60, 130, 125), 200, 200)
    assert not reason and valid == dict(x0=45, y0=42, size=100, angle=0)


def test_actual_pool_crop_parity():
    from pathlib import Path
    if not Path('data/student/index.csv').exists():
        pytest.skip('Local course pool not installed')
    pool = Pool('data/student')
    for pool_id in ['s000000', 's003024', 's076249']:
        image = pool.image(pool_id)
        width, height = image.size
        side = min(width, height) // 2
        x0, y0 = (width - side) // 2, (height - side) // 2
        for angle in [0, -22, 31]:
            assert np.array_equal(np.array(crop(image, x0, y0, side, angle)),
                                  np.array(official_crop(image, x0, y0, side, angle)))

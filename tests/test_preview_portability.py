"""Cross-platform preview bindings: accept a PNG byte mismatch only with exact pixel evidence."""
import hashlib
import io

import numpy as np
from PIL import Image

from src.curation import freeze


def png(pixels, level):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG', compress_level=level)
    return stream.getvalue()


def setup(monkeypatch):
    pixels = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    reviewed, current = png(pixels, 9), png(pixels, 1)  # same pixels, different PNG bytes
    assert reviewed != current
    monkeypatch.setattr(freeze, 'rendered_crop', lambda pool, pool_id, geometry: current)
    item = dict(candidate_id='c1', pool_id='p1', label=1, x0=3, y0=4, size=60, angle=0.0,
                preview_sha256=hashlib.sha256(reviewed).hexdigest())
    record = {k: item[k] for k in ('candidate_id', 'pool_id', 'label', 'x0', 'y0', 'size', 'angle')} | {
        'recorded_preview_sha256': item['preview_sha256'], 'method': 'zlib_ng_reencoding',
        'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
    geometry = {k: item[k] for k in ('x0', 'y0', 'size', 'angle')}
    return item, record, geometry


def test_byte_mismatch_requires_receipt_and_identical_pixels(monkeypatch):
    item, record, geometry = setup(monkeypatch)
    assert freeze.preview_matches(None, item, geometry, {}) is None
    assert freeze.preview_matches(None, item, geometry, {'c1': record}) == 'zlib_ng_reencoding'
    assert freeze.preview_matches(None, item, geometry, {'c1': record | {'pixel_sha256': '0' * 64}}) is None


def test_receipt_cannot_cover_changed_geometry_or_species(monkeypatch):
    item, record, geometry = setup(monkeypatch)
    assert freeze.preview_matches(None, item | {'x0': 5}, geometry, {'c1': record}) is None
    assert freeze.preview_matches(None, item | {'label': 0}, geometry, {'c1': record}) is None
    assert freeze.preview_matches(None, item | {'preview_sha256': 'f' * 64}, geometry, {'c1': record}) is None

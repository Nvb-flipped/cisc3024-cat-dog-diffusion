"""Manifest crop geometry, independent of the official implementation."""
import math
from PIL import Image


def crop(image, x0, y0, size, angle=0.0, res=32):
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC,
                             center=(x0 + size / 2, y0 + size / 2))
    return image.crop((x0, y0, x0 + size, y0 + size)).resize(
        (res, res), Image.Resampling.LANCZOS)


def inside(width, height, x0, y0, size, angle=0):
    if size < 40 or not math.isfinite(angle) or abs(angle) > 45:
        return False
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    cx, cy = x0 + size / 2, y0 + size / 2
    for dx, dy in ((-size / 2, -size / 2), (size / 2, -size / 2),
                   (size / 2, size / 2), (-size / 2, size / 2)):
        x, y = cx + dx * c - dy * s, cy + dx * s + dy * c
        if x < -0.5 or y < -0.5 or x > width + 0.5 or y > height + 0.5:
            return False
    return True


def proposal(box, width, height, occupancy=0.70):
    """Preserve box center; do not clamp, pad, or silently shrink invalid crops."""
    if not 0.60 <= occupancy <= 0.80:
        raise ValueError('occupancy must be in [0.60, 0.80]')
    left, top, right, bottom = map(float, box)
    if not all(math.isfinite(v) for v in (left, top, right, bottom)):
        return None, 'nonfinite_box'
    if right - left < 32:
        return None, 'head_width_below_32'
    if bottom <= top:
        return None, 'empty_box'
    size = max(40, math.ceil(max(right - left, bottom - top) / occupancy))
    x0 = round((left + right - size) / 2)
    y0 = round((top + bottom - size) / 2)
    if not inside(width, height, x0, y0, size):
        return None, 'square_outside_source'
    return {'x0': x0, 'y0': y0, 'size': size, 'angle': 0}, ''

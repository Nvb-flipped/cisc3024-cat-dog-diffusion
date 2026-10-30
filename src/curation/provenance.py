"""Stable identity of the personally reviewed crop, independent of log ordering."""
import hashlib
import json


def crop_review_identity(row):
    identity = {'pool_id': row['pool_id'], 'label': int(row['label']),
        'x0': int(row['x0']), 'y0': int(row['y0']), 'size': int(row['size']),
        'angle': float(row['angle']), 'preview_sha256': row['preview_sha256']}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

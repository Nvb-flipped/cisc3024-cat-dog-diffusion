"""Content identity of compressed course pool files, independent of file timestamps."""
import hashlib
import json
from pathlib import Path
from src.common import sha256


def fingerprint(pool_path):
    pool_path = Path(pool_path)
    paths = sorted(pool_path.glob('*.zip')) + sorted((pool_path / 'images').glob('*.jpg'))
    hashes = {path.relative_to(pool_path).as_posix(): sha256(path) for path in paths}
    hashes['index.csv'] = sha256(pool_path / 'index.csv')
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes

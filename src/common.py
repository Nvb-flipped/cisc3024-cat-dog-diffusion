"""Small reproducibility helpers shared by command-line entry points."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    import torch
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'git_commit': commit, 'git_dirty': dirty}

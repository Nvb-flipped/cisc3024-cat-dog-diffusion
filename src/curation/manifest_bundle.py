"""Validate a complete immutable train/holdout/provenance bundle before publication."""
import csv
import shutil
import tempfile
from pathlib import Path
from src.common import sha256, write_json


def publish_bundle(out, training, validation, metadata, *, columns, check, validation_source=None):
    out = Path(out)
    targets = [out, out.with_suffix('.validation.csv'), out.with_suffix('.provenance.json')]
    if any(path.exists() for path in targets):
        raise ValueError('Manifest is immutable; choose a new output version')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='freeze_', dir=out.parent) as temporary:
        staged = [Path(temporary) / path.name for path in targets]
        written = []
        for path, target, rows in zip(staged[:2], targets[:2], [training, validation]):
            if target == targets[1] and validation_source is not None:
                shutil.copyfile(validation_source, path)
            else:
                with path.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, columns, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(rows)
            checked = check(path)
            if checked['errors']:
                raise ValueError(f'Official manifest check failed: {checked["errors"]}')
            written.append({'path': str(target), 'sha256': sha256(path), 'stats': checked['stats'],
                            'warnings': checked['warnings']})
        write_json(staged[2], metadata | {'manifests': written})
        created = []
        try:
            for path, target in zip(staged, targets):
                with target.open('xb') as destination:
                    created.append(target)
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, destination)
        except BaseException:
            for path in created:
                path.unlink()
            raise
    return written

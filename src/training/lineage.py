"""Baseline training requires a concrete frozen personal-review manifest lineage."""
import json
from pathlib import Path
from src.common import sha256


def reviewed_manifest_provenance(manifest):
    manifest = Path(manifest)
    path = manifest.with_suffix('.provenance.json')
    if not path.exists():
        raise ValueError('Baseline training requires the frozen personally reviewed manifest provenance')
    record = json.loads(path.read_text())
    if record.get('origin') not in ['student_reviewed_manifest', 'subset_of_student_reviewed_manifest'] or \
       record.get('diagnostic_only') or not record.get('pool_content_sha256') or \
       not record.get('manifests') or record['manifests'][0]['sha256'] != sha256(manifest):
        raise ValueError('Training manifest is diagnostic, changed, or lacks the frozen personal-review lineage')
    return sha256(path)

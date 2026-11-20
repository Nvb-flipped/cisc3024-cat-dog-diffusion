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
    if record.get('origin') == 'ai_assisted_reviewed_manifest':
        # Full-pool manifest: personal parent rows plus logged VLM curation decisions.
        parent = Path(record.get('parent_manifest', ''))
        if record.get('vlm_decisions_are_personal_verdicts') is not False or not parent.exists() or \
                sha256(parent) != record.get('parent_manifest_sha256') or \
                json.loads(parent.with_suffix('.provenance.json').read_text()).get('origin') != 'student_reviewed_manifest' or \
                not record.get('primary_ledgers') or not record.get('manifests') or \
                record['manifests'][0]['sha256'] != sha256(manifest) or not record.get('pool_content_sha256'):
            raise ValueError('AI-assisted manifest lacks a verified personal parent or VLM ledger lineage')
        return sha256(path)
    if record.get('origin') not in ['student_reviewed_manifest', 'subset_of_student_reviewed_manifest'] or \
       record.get('diagnostic_only') or not record.get('pool_content_sha256') or \
       not record.get('manifests') or record['manifests'][0]['sha256'] != sha256(manifest):
        raise ValueError('Training manifest is diagnostic, changed, or lacks the frozen personal-review lineage')
    return sha256(path)

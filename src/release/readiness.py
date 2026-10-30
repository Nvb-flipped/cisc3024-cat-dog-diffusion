"""Structural release checks. These cannot certify student authorship or hidden-test marks."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from src.common import ROOT, sha256, write_json
from src.release.check import check_files
from src.evaluation.audit import summarize as audit_summary

FINAL_SEEDS = [0, 101, 202, 303, 404]


def audit_check(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 200:
        raise ValueError('Official audit must contain exactly 200 rows')
    for index, row in enumerate(rows, 2):
        if row.get('verdict', '').strip().lower() not in ['good', 'bad']:
            raise ValueError(f'Unfinished audit verdict at line {index}')
        if row['verdict'].strip().lower() == 'bad' and not row.get('reason', '').strip():
            raise ValueError(f'Bad audit row lacks reason at line {index}')
    return {'rows': 200, 'good': sum(r['verdict'].strip().lower() == 'good' for r in rows),
            'sha256': sha256(path), 'scope': 'CSV completeness only; student must personally label the exact official sample'}


def readiness(root=ROOT):
    root = Path(root)
    issues, evidence, shortfalls = [], {}, []
    submission_hashes = None
    try:
        evidence['submission'] = check_files(root / 'submission')
        submission_hashes = evidence['submission']['file_sha256']
    except (ValueError, OSError, KeyError) as exc:
        issues.append(f'Final inference export: {exc}')
    try:
        evidence['audit'] = audit_check(root / 'audit/audit.csv')
        audit_provenance = json.loads((root / 'audit/provenance.json').read_text())
        evidence['audit_sample'] = audit_summary(root / 'audit/audit.csv', root / 'submission/manifest.csv',
                                               audit_provenance['student_id'])
    except (ValueError, OSError, KeyError) as exc:
        issues.append(f'Personal official audit: {exc}')
    for name in ['README.md', 'DATA_CARD.md', 'AGENTS.md', 'AGENT_LOG.md']:
        path = root / name
        if not path.exists():
            issues.append(f'Missing required document: {name}')
        else:
            text = path.read_text(encoding='utf-8')
            if name == 'DATA_CARD.md' and ('provisional' in text.lower() or 'audit precision' not in text.lower()):
                issues.append('Data card is provisional or lacks audit precision')
            if name == 'AGENT_LOG.md' and ('These are pending tasks' in text or len(re.findall(r'^##\s+', text, re.M)) < 3):
                issues.append('Three contemporaneous independent student checks are still pending')
    for seed in FINAL_SEEDS:
        path = root / f'results/final/seed_{seed}.json'
        try:
            result = json.loads(path.read_text())
            provenance = json.loads(path.with_suffix('.provenance.json').read_text())
            if provenance['result_sha256'] != sha256(path) or provenance['submission_file_sha256'] != submission_hashes:
                raise ValueError('result is stale or does not match final inference files')
            p, m = result['protocol'], result['metrics']
            if p['seed'] != seed or p['n_per_class'] != 1000 or p['batch_size'] != 500 or not p['grading']:
                raise ValueError('wrong official protocol')
            if result['violations'] or not result['marks']['valid'] or m['memorization_rate'] is None:
                raise ValueError('violations or missing memorization')
            if not result['training_data']['stats'].get('valid_rows'):
                raise ValueError('missing full training-set rebuild')
            if not all(isinstance(m.get(key), (int, float)) and math.isfinite(m[key])
                       for key in ['accuracy', 'fid', 'kid', 'kid_std', 'precision', 'recall', 'memorization_rate']):
                raise ValueError('missing or nonfinite quality metric')
            if m['memorization_rate'] > .05:
                raise ValueError('memorization exceeds official hard limit')
            if not (m['accuracy'] >= .90 and m['fid'] <= 50 and m['recall'] >= .40):
                shortfalls.append(f'Excellent-tier objective not met on generation seed {seed}')
            evidence[f'seed_{seed}'] = {'sha256': sha256(path), 'metrics': m}
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f'Final seed {seed}: {exc}')
    for device in ['cpu', 'cuda']:
        path = root / f'results/release/offline_{device}.json'
        try:
            record = json.loads(path.read_text())
            if not record['offline'] or record['device'] != device or \
               [r['count'] for r in record['records']] != [1, 2, 20, 50, 500]:
                raise ValueError('incomplete offline interface matrix')
            if record.get('file_sha256') != submission_hashes:
                raise ValueError('offline evidence is stale for final inference files')
            evidence[f'offline_{device}'] = sha256(path)
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f'Offline {device}: {exc}')
    for name in ['report/report.pdf', 'presentation/presentation.pptx',
                 'presentation/presentation.tex', 'presentation/presentation.pdf']:
        if not (root / name).is_file():
            issues.append(f'Missing final artifact: {name}')
    try:
        pages = json.loads((root / 'report/page_count.json').read_text())
        if pages['pdf_sha256'] != sha256(root / 'report/report.pdf') or pages['main_text_pages'] > 6:
            raise ValueError('report hash changed or main text exceeds six pages')
        evidence['report_pages'] = pages
    except (OSError, ValueError, KeyError) as exc:
        issues.append(f'Report page evidence: {exc}')
    return {'structural_checks_passed': not issues, 'issues': issues, 'objective_shortfalls': shortfalls,
        'excellent_tier_all_seeds': not issues and not shortfalls, 'evidence': evidence,
        'scope': 'File/protocol completeness and excellent-tier objective only. '
                 'Does not certify authentic student judgments, scientific interpretation, presentation ability, '
                 'qualitative marks, hidden-test results or submission approval.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=ROOT, type=Path)
    p.add_argument('--out', default='results/release/readiness.json', type=Path)
    args = p.parse_args()
    result = readiness(args.root)
    write_json(args.out, result)
    print('\n'.join(result['issues']) if result['issues'] else 'Structural checks passed; human final review still applies.')
    if result['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

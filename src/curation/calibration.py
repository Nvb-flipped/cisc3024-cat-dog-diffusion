"""Summarize personally saved pilot judgments without turning corrections into successes."""
import argparse
import csv
import json
import math
from pathlib import Path
from src.common import sha256, write_json
from src.curation.build import GRID
from src.curation.freeze import latest_decisions


def wilson(good, total, z=1.959963984540054):
    if total <= 0 or not 0 <= good <= total:
        raise ValueError('Wilson interval requires 0 <= good <= total and total > 0')
    p = good / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    lower = 0. if good == 0 else max(0., center-radius)
    upper = 1. if good == total else min(1., center+radius)
    return lower, upper


def original_is_good(candidate, decision):
    if decision['verdict'] != 'good' or int(candidate['label']) != decision['label']:
        return False
    return all(float(candidate[key]) == float(decision[key]) for key in ['x0', 'y0', 'size', 'angle'])


def summarize(candidates_path, selection_path, decisions_path):
    with Path(candidates_path).open(newline='', encoding='utf-8') as f:
        candidates = {r['candidate_id']: r for r in csv.DictReader(f)}
    selected = json.loads(Path(selection_path).read_text())
    decisions = latest_decisions(decisions_path)
    required = selected['proposed_candidate_ids'] + [f'reject:{pid}' for pid in selected['rejected_pool_ids']]
    missing = [key for key in required if key not in decisions]
    uncertain = [key for key in selected['proposed_candidate_ids']
                 if key in decisions and decisions[key]['verdict'] == 'uncertain']
    table = []
    for setting in GRID:
        setting = list(setting)
        eligible = [r for r in candidates.values() if r['size'] and setting in json.loads(r['eligible_settings'])]
        review = [r for r in eligible if r['candidate_id'] in selected['proposed_candidate_ids']
                  and r['candidate_id'] in decisions and decisions[r['candidate_id']]['verdict'] != 'uncertain']
        good = sum(original_is_good(r, decisions[r['candidate_id']]) for r in review)
        table.append({'box_threshold': setting[0], 'text_threshold': setting[1],
            'yield_proposals': len(eligible), 'yield_sources': len({r['pool_id'] for r in eligible}),
            'reviewed': len(review), 'original_good': good,
            'original_precision': good / len(review) if review else None,
            'wilson_95': wilson(good, len(review)) if review else None})
    eligible_settings = [row for row in table if row['original_precision'] is not None and row['original_precision'] >= .95]
    eligible_settings.sort(key=lambda r: (-r['yield_sources'], -r['yield_proposals'], -r['original_precision']))
    observed_settings = [row for row in table if row['original_precision'] is not None]
    fallback = max(observed_settings, key=lambda row: (row['original_precision'], row['yield_sources'])) if observed_settings else None
    rejected = [decisions.get(f'reject:{pid}') for pid in selected['rejected_pool_ids']]
    report = {'status': 'complete' if not missing and not uncertain else 'pending_student_review',
        'missing_ids': missing, 'uncertain_ids': uncertain, 'threshold_table': table,
        'recommended_setting': eligible_settings[0] if eligible_settings and not missing and not uncertain else None,
        'highest_precision_fallback': fallback if not eligible_settings and not missing and not uncertain else None,
        'rejected_sources_reviewed': sum(r is not None for r in rejected),
        'missed_head_diagnostics': sum(r is not None and r['verdict'] == 'missed_head' for r in rejected),
        'rejected_sample_inference': 'diagnostic only; not an unweighted whole-pool accuracy estimate',
        'correction_policy': 'Species or geometry corrections count as failures of the original proposal.',
        'acceptance_policy': 'Every final training crop still requires student specification/duplicate review.',
        'inputs': {str(path): sha256(path) for path in [candidates_path, selection_path, decisions_path]}}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', default='results/curation/pilot', type=Path)
    p.add_argument('--decisions', default='annotations/decisions.jsonl', type=Path)
    p.add_argument('--out', default='results/curation/calibration.json', type=Path)
    args = p.parse_args()
    if not args.decisions.exists():
        raise ValueError('No personally saved student judgments yet')
    report = summarize(args.pilot / 'candidates.csv', args.pilot / 'calibration_selection.json', args.decisions)
    write_json(args.out, report)
    print(report['status'], '; missing', len(report['missing_ids']), '; uncertain', len(report['uncertain_ids']))


if __name__ == '__main__':
    main()

"""Compare two pool-only ranking methods against existing personal crop verdicts."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from src.common import sha256, write_json

PROPOSALS = Path('results/curation/full/candidates.csv')
SCORES = Path('results/curation/photo_screen_full/scores.csv')
DECISIONS = Path('annotations/decisions.jsonl')
OUTPUT = Path('private_build/review_yield_v001')
GEOMETRY = ('x0', 'y0', 'size', 'angle')
MARGINS = ('source_species_margin', 'crop_species_margin', 'source_photo_margin', 'crop_photo_margin')


def features(row, proposal):
    side = float(proposal['size'])
    return [float(row[key]) for key in MARGINS] + [
        float(row['source_species_hint'] == row['crop_species_hint']),
        float(row['proposed_label'] == row['crop_species_hint']),
        float(proposal['score']), float(proposal['head_width']) / side,
        float(proposal['head_height']) / side, side / 400,
        *json.loads(row['source_cosine_scores']), *json.loads(row['crop_cosine_scores'])]


def fit_predict(train_x, train_y, test_x):
    centre = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x, test_x = (train_x - centre) / scale, (test_x - centre) / scale
    model = torch.nn.Linear(train_x.shape[1], 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=60, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        prediction = model(train_x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, train_y)
        loss = loss + .02 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return model(test_x).squeeze(1).sigmoid().tolist()


def at_budget(rows, scores, cap=25):
    hits, counts = Counter(), Counter()
    used = set()
    for index in sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]['candidate_id'])):
        row = rows[index]
        label = row['crop_species_hint']
        if counts[label] >= cap or row['pool_id'] in used:
            continue
        used.add(row['pool_id'])
        counts[label] += 1
        hits[label] += row['target']
    return {label: {'reviewed': counts[label], 'approved': hits[label]} for label in ('0', '1')}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(4)
    decisions = {row['candidate_id']: row for line in DECISIONS.read_text(encoding='utf-8').splitlines()
                 if (row := json.loads(line))}
    reviewed_sources = {row['pool_id'] for row in decisions.values()}
    with PROPOSALS.open(newline='', encoding='utf-8') as stream:
        proposals = {row['candidate_id']: row for row in csv.DictReader(stream) if row['size']}
    rows, matrix = [], []
    with SCORES.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            candidate = row['candidate_id']
            if candidate not in proposals:
                continue
            proposal = proposals[candidate]
            decision = decisions.get(candidate)
            if decision and decision['verdict'] in ('good', 'bad') and decision.get('origin') == 'student_browser_review':
                if all(float(decision[key]) == float(proposal[key]) for key in GEOMETRY):
                    row['target'] = int(decision['verdict'] == 'good'
                                        and str(decision['label']) == row['crop_species_hint'])
            if 'target' in row or row['pool_id'] not in reviewed_sources:
                matrix.append(features(row, proposal))
                rows.append(row)
    values = torch.tensor(matrix, dtype=torch.float32)
    labeled = [i for i, row in enumerate(rows) if 'target' in row]
    fold = {i: hashlib.sha256(rows[i]['pool_id'].encode()).digest()[0] % 5 for i in labeled}
    reports = []
    for part in range(5):
        train = [i for i in labeled if fold[i] != part]
        test = [i for i in labeled if fold[i] == part]
        y = torch.tensor([rows[i]['target'] for i in train], dtype=torch.float32)
        ranked = fit_predict(values[train], y, values[test])
        cohort = [rows[i] for i in test]
        reports.append(dict(fold=part, examples=len(test), positives=sum(r['target'] for r in cohort),
                            photo_rank=at_budget(cohort, [float(r['review_priority_score']) for r in cohort]),
                            learned_rank=at_budget(cohort, ranked)))
    candidates = [i for i, row in enumerate(rows) if 'target' not in row
                  and row['source_species_hint'] == row['crop_species_hint']
                  and min(float(row[key]) for key in MARGINS[-2:]) > 0]
    all_y = torch.tensor([rows[i]['target'] for i in labeled], dtype=torch.float32)
    candidate_scores = fit_predict(values[labeled], all_y, values[candidates])
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / 'unreviewed_rank.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['candidate_id', 'pool_id', 'crop_species_hint', 'review_priority_score', 'learned_score']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, score in sorted(zip(candidates, candidate_scores), key=lambda item: -item[1]):
            writer.writerow({key: rows[index][key] for key in fields[:-1]} | {'learned_score': score})
    report = dict(inputs={str(p): sha256(p) for p in (PROPOSALS, SCORES, DECISIONS, Path(__file__))},
                  labeled=len(labeled), positives=int(all_y.sum()), ranked_candidates=len(candidates),
                  folds=reports, scope='personal exact-crop verdict replay, not full-pool precision or training approval')
    write_json(OUTPUT / 'report.json', report)
    for name in ('photo_rank', 'learned_rank'):
        print(name, {label: sum(r[name][label]['approved'] for r in reports) for label in ('0', '1')})


if __name__ == '__main__':
    main()

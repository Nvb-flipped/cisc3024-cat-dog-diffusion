"""Rank unreviewed head candidates by a linear probe on local CLIP crop embeddings (curation only).

  embed   CLIP ViT-B/32 image embeddings of exact candidate squares (pool images only)
  rank    fit an L2-regularised logistic regression on spec judgements already recorded for pool
          crops (personal verdicts, agent visual reviews, audits) and score unreviewed candidates

The score only orders the agent's visual review queue; it never accepts an image by itself.
No dev image or dev feature is used, and the evaluator's networks are never involved.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import ROOT, write_json
from src.curation.rescue import verify_model

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, crop  # noqa: E402

R = Path('results/fullpool_20261119')


class Squares(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows, self.pool = rows, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.pool is None:
            self.pool = Pool('data/student')
        r = self.rows[i]
        return r['candidate_id'], crop(self.pool.image(r['pool_id']), int(r['x0']), int(r['y0']),
                                       int(r['size']), float(r['angle'] or 0), res=224)


def candidate_rows():
    rows = {}
    for q in ('queue_stage4_dino.csv', 'queue_stage4_rescue.csv'):
        for r in csv.DictReader(open(R / q, newline='', encoding='utf-8')):
            rows[r['candidate_id']] = r
    for r in csv.DictReader(open(R / 'calibration/personal_calibration.csv', newline='', encoding='utf-8')):
        rows.setdefault(r['candidate_id'], r)
    for r in csv.DictReader(open(R / 'calibration/personal_holdout.csv', newline='', encoding='utf-8')):
        rows.setdefault(r['candidate_id'], r)
    return rows


def embed(out):
    from transformers import CLIPModel, CLIPProcessor
    folder = Path('models/clip-photo-screen')
    verify_model(folder)
    rows = list(candidate_rows().values())
    processor = CLIPProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(folder, local_files_only=True).cuda().eval()
    loader = torch.utils.data.DataLoader(Squares(rows), batch_size=256, num_workers=6,
                                         collate_fn=lambda b: list(zip(*b)))
    ids, feats = [], []
    for batch_ids, images in loader:
        with torch.inference_mode():
            f = F.normalize(model.get_image_features(**processor(images=list(images), return_tensors='pt').to('cuda')), dim=-1)
        ids += list(batch_ids)
        feats.append(f.half().cpu().numpy())
        print(len(ids), flush=True)
    np.savez_compressed(out, ids=np.array(ids), feats=np.concatenate(feats))


def labels():
    """Spec judgements recorded for pool crops; later records override earlier ones."""
    y = {}
    for r in csv.DictReader(open(R / 'calibration/personal_calibration.csv')):
        y[r['candidate_id']] = r['personal_verdict'] == 'good'
    for r in csv.DictReader(open(R / 'calibration/personal_holdout.csv')):
        y[r['candidate_id']] = r['personal_verdict'] == 'good'
    for path in ('private_build/fullpool_qa/blind200/agent_labels.json', 'private_build/fullpool_qa/accepted150/agent_labels.json'):
        for k, v in json.load(open(path))['labels'].items():
            y[k] = v == 'good'
    for path in ('private_build/fullpool_qa/review_v011_ai/agent_review.jsonl', 'private_build/fullpool_qa/review_v6_rejected/agent_review.jsonl',
                 'private_build/fullpool_qa/rereview_lying/agent_review.jsonl', 'private_build/fullpool_qa/v8_yield100/agent_review.jsonl'):
        if Path(path).exists():
            for line in open(path):
                r = json.loads(line)
                y[r['candidate_id']] = r['verdict'] == 'good'
    return y


def rank(emb, out, reviewed_out):
    data = np.load(emb)
    index = {c: i for i, c in enumerate(data['ids'])}
    x_all = torch.tensor(data['feats'].astype(np.float32))
    y = {k: v for k, v in labels().items() if k in index}
    keys = sorted(y)
    x = x_all[[index[k] for k in keys]]
    t = torch.tensor([float(y[k]) for k in keys])
    # 5-fold cross-validated ranking quality, then a final fit on all labels.
    perm = torch.randperm(len(keys), generator=torch.Generator().manual_seed(0))
    folds, aucs = perm.chunk(5), []

    def fit(xs, ts):
        w = torch.zeros(xs.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        pos = ts.mean()

        def closure():
            opt.zero_grad()
            weight = torch.where(ts > 0, 0.5 / pos, 0.5 / (1 - pos))
            loss = (F.binary_cross_entropy_with_logits(xs @ w * 20 + b, ts, weight=weight) + 1e-3 * w.square().sum())
            loss.backward()
            return loss
        opt.step(closure)
        return w.detach(), b.detach()

    for k in range(5):
        test = folds[k]
        train = torch.cat([folds[j] for j in range(5) if j != k])
        w, b = fit(x[train], t[train])
        s = (x[test] @ w * 20 + b)
        pos, neg = s[t[test] > 0], s[t[test] == 0]
        aucs.append(float((pos[:, None] > neg[None, :]).float().mean()))
    w, b = fit(x, t)
    scores = (x_all @ w * 20 + b).sigmoid().numpy()
    unlabelled = [(float(scores[i]), c) for c, i in index.items() if c not in y]
    unlabelled.sort(reverse=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'score'])
        writer.writerows((c, round(s, 5)) for s, c in unlabelled)
    write_json(reviewed_out, {'labelled': len(keys), 'positives': int(t.sum()), 'cv_auc': [round(a, 4) for a in aucs],
                              'unlabelled_ranked': len(unlabelled),
                              'score_quantiles': {q: round(float(np.quantile([s for s, _ in unlabelled], q)), 4) for q in (0.5, 0.9, 0.95, 0.99)}})
    print({'labelled': len(keys), 'positives': int(t.sum()), 'cv_auc': aucs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('embed', 'rank'))
    args = p.parse_args()
    if args.phase == 'embed':
        embed(R / 'clip_crop_embeddings.npz')
    else:
        rank(R / 'clip_crop_embeddings.npz', R / 'recovery_ranked.csv', R / 'recovery_ranker.json')


if __name__ == '__main__':
    main()

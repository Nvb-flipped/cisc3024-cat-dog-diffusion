"""Exploratory CLIP photo/species scores on pool sources and crops; no auto-labels."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from src.common import ROOT, environment, read_config, sha256, write_json
from src.curation.geometry import crop
sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool


def main():
    from transformers import CLIPModel, CLIPProcessor
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/photo_screen.yaml')
    args = p.parse_args()
    torch.set_num_threads(8)
    config = read_config(args.config)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'scores.csv').exists():
        raise FileExistsError('Screening scores are immutable; choose a fresh configured output')
    folder = Path(config['model_path'])
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        if sha256(folder / name) != expected:
            raise ValueError(f'Curation model changed: {name}')
    processor = CLIPProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(folder, local_files_only=True).cuda().eval()
    pool = Pool(config['pool'])
    with Path(config['candidates']).open(newline='', encoding='utf-8') as f:
        candidates = [row for row in csv.DictReader(f) if row['size']]
    source_ids = sorted({row['pool_id'] for row in candidates})

    @torch.inference_mode()
    def text_features(prompts):
        inputs = processor(text=prompts, padding=True, return_tensors='pt').to('cuda')
        return F.normalize(model.get_text_features(**inputs), dim=-1)

    source_text = text_features(config['source_prompts'])
    crop_text = text_features(config['crop_prompts'])

    @torch.inference_mode()
    def similarities(images, text):
        inputs = processor(images=images, return_tensors='pt').to('cuda')
        features = F.normalize(model.get_image_features(**inputs), dim=-1)
        return (features @ text.T).cpu()

    begin = time.perf_counter()
    batch = config['batch_size']
    source_scores = {}
    for start in range(0, len(source_ids), batch):
        ids = source_ids[start:start + batch]
        scores = similarities([pool.image(pool_id) for pool_id in ids], source_text)
        source_scores.update(zip(ids, scores.tolist()))
    rows = []
    for start in range(0, len(candidates), batch):
        items = candidates[start:start + batch]
        images = [crop(pool.image(row['pool_id']), int(row['x0']), int(row['y0']),
                       int(row['size']), float(row['angle'])) for row in items]
        scores = similarities(images, crop_text)
        for item, crop_scores in zip(items, scores.tolist()):
            source = source_scores[item['pool_id']]
            source_margin = max(source[:2]) - max(source[2:])
            crop_margin = max(crop_scores[:2]) - max(crop_scores[2:])
            # Scores rank review candidates; they are not calibrated compliance probabilities.
            rows.append({'candidate_id': item['candidate_id'], 'pool_id': item['pool_id'],
                'proposed_label': item['label'], 'source_species_hint': int(source[1] > source[0]),
                'crop_species_hint': int(crop_scores[1] > crop_scores[0]),
                'source_species_margin': source[0] - source[1],
                'crop_species_margin': crop_scores[0] - crop_scores[1],
                'source_photo_margin': source_margin, 'crop_photo_margin': crop_margin,
                'review_priority_score': min(source_margin, crop_margin),
                'source_cosine_scores': json.dumps(source), 'crop_cosine_scores': json.dumps(crop_scores)})
        print(f'{min(start + batch, len(candidates))}/{len(candidates)} scored', flush=True)
    rows.sort(key=lambda row: (-row['review_priority_score'], row['candidate_id']))
    with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / 'experiment.json', {'purpose': 'exploratory pool-only review-priority aid',
        'status': 'No automatic accepts/rejects or final labels. Student calibration required.',
        'config': config, 'config_sha256': sha256(args.config), 'model_revision': provenance['revision'],
        'candidate_sha256': sha256(config['candidates']), 'candidates': len(rows), 'sources': len(source_ids),
        'seconds': time.perf_counter() - begin, 'environment': environment(),
        'rationale': 'Initial DINO pilot contains product prints, drawings, toys and conflicting species proposals. '
                     'Compare source/crop text similarities without using dev images/features or evaluator networks.'})


if __name__ == '__main__':
    main()

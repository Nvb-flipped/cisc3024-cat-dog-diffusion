"""Re-measure the head inside each exact candidate square with the local, pinned Grounding DINO.

The square (official crop geometry) is rendered at 448 px and searched for "a cat head. a dog head.".
`head_fraction` = larger side of the best head box / square side. Whole-body or small-head crops give
small fractions; a deterministic, model-independent framing check alongside the VLM. Curation only
(real pool images); never applied to generated images. Appends JSON lines, resumable.
"""
import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import torch

from src.common import ROOT
from src.curation.rescue import verify_model
from src.curation.vlm_screen import done

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, crop  # noqa: E402

PROMPT = 'a cat head. a dog head.'
SIDE = 448


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    folder = Path('models/grounding-dino-tiny')
    provenance = verify_model(folder)
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        folder, local_files_only=True, disable_custom_kernels=True).to(args.device).eval()
    post = processor.post_process_grounded_object_detection
    threshold_name = 'threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
    finished = done(args.out)
    rows = [r for r in csv.DictReader(open(args.queue, newline='', encoding='utf-8')) if r['candidate_id'] not in finished]
    pool = Pool('data/student')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a', encoding='utf-8') as stream:
        for row in rows:
            image = crop(pool.image(row['pool_id']), int(row['x0']), int(row['y0']), int(row['size']),
                         float(row['angle'] or 0), res=SIDE)
            inputs = processor(images=image, text=PROMPT, return_tensors='pt').to(args.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            detected = post(outputs, inputs.input_ids, **{threshold_name: 0.25}, text_threshold=0.25,
                            target_sizes=[(SIDE, SIDE)])[0]
            boxes, scores = detected['boxes'].tolist(), detected['scores'].tolist()
            best = max(range(len(scores)), key=scores.__getitem__) if scores else None
            fraction = None if best is None else max(boxes[best][2] - boxes[best][0], boxes[best][3] - boxes[best][1]) / SIDE
            stream.write(json.dumps({'candidate_id': row['candidate_id'], 'heads_detected': len(scores),
                                     'head_fraction': None if fraction is None else round(fraction, 4),
                                     'best_score': None if best is None else round(scores[best], 4),
                                     'model_revision': provenance['revision'], 'prompt': PROMPT}) + '\n')
            stream.flush()


if __name__ == '__main__':
    main()

"""Stage 2 full-pool rescue with local, pinned, curation-only models (resumable GPU pieces).

  clip   CLIP ViT-B/32 source screen of every pool image (real cat/dog photo vs other content).
  dino   targeted Grounding DINO re-detection on detector-negative sources the CLIP screen flags,
         with head/face prompts and lower thresholds; every box goes through the same official
         geometry rules (src.curation.fullpool.rescue_geometry, head width >= 32, no padding).

Each phase appends one JSON line per source and skips sources already recorded, so an
interrupted container (shared GPU) loses at most one in-flight batch. Outputs are proposals,
not verdicts. Dev images are never read.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.common import ROOT, sha256
from src.curation.build import candidate_key
from src.curation.fullpool import rescue_geometry
from src.curation.geometry import proposal

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool  # noqa: E402

OUT = Path('results/fullpool_20261119/rescue')
SOURCE_PROMPTS = ['a photograph of a real living cat', 'a photograph of a real living dog',
                  'a cartoon, drawing or painting of a cat or dog', 'a toy or stuffed animal',
                  'a cat or dog printed on a product or package', 'an animal sculpture or artificial animal',
                  'an object or scene without any animal']
DINO_PROMPT = 'a cat head. a dog head. a cat face. a dog face.'
DINO_BOX_THRESHOLD = 0.15
DINO_TEXT_THRESHOLD = 0.15
FLAG_PROBABILITY = 0.10  # P(real cat)+P(real dog); 99.75% recall on personally confirmed sources


def verify_model(folder):
    provenance = json.loads((Path(folder) / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        if sha256(Path(folder) / name) != expected:
            raise ValueError(f'Curation model changed: {folder}/{name}')
    return provenance


def done_ids(path):
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                ids.add(json.loads(line)['pool_id'])
            except (json.JSONDecodeError, KeyError):
                pass  # a torn final line from an interrupted write is simply redone
    return ids


class Sources(torch.utils.data.Dataset):
    def __init__(self, ids, pool_path):
        self.ids, self.pool_path, self.pool = ids, pool_path, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.pool is None:
            self.pool = Pool(self.pool_path)  # one zip handle set per worker process
        return self.ids[index], self.pool.image(self.ids[index])


def clip_screen(pool_path, batch_size=256, workers=6):
    from transformers import CLIPModel, CLIPProcessor
    folder = Path('models/clip-photo-screen')
    provenance = verify_model(folder)
    out = OUT / 'clip_sources.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open('results/curation/integrity.csv', newline='', encoding='utf-8') as stream:
        ids = [row['pool_id'] for row in csv.DictReader(stream) if row['status'] == 'ok']
    remaining = sorted(set(ids) - done_ids(out))
    print(f'clip: {len(remaining)} of {len(ids)} sources remaining', flush=True)
    processor = CLIPProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(folder, local_files_only=True).cuda().eval()
    with torch.inference_mode():
        text = processor(text=SOURCE_PROMPTS, padding=True, return_tensors='pt').to('cuda')
        text = F.normalize(model.get_text_features(**text), dim=-1)
    loader = torch.utils.data.DataLoader(Sources(remaining, pool_path), batch_size=batch_size,
                                         num_workers=workers, collate_fn=lambda batch: list(zip(*batch)))
    begin = time.perf_counter()
    with out.open('a', encoding='utf-8') as stream:
        for count, (batch_ids, images) in enumerate(loader):
            with torch.inference_mode():
                inputs = processor(images=list(images), return_tensors='pt').to('cuda')
                features = F.normalize(model.get_image_features(**inputs), dim=-1)
                cosine = features @ text.T
                probability = (cosine * model.logit_scale.exp()).softmax(dim=-1)
            for pool_id, cos, prob in zip(batch_ids, cosine.tolist(), probability.tolist()):
                stream.write(json.dumps({'pool_id': pool_id, 'cosine': [round(v, 5) for v in cos],
                                         'probability': [round(v, 5) for v in prob],
                                         'model_revision': provenance['revision']}) + '\n')
            stream.flush()
            if count % 20 == 0:
                print(f'clip {count * batch_size + len(batch_ids)}/{len(remaining)} '
                      f'{time.perf_counter() - begin:.0f}s', flush=True)


def dino_rescue(pool_path, device='cuda', shard=0, shards=1):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import inspect
    folder = Path('models/grounding-dino-tiny')
    provenance = verify_model(folder)
    clip = {}
    with (OUT / 'clip_sources.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            clip[row['pool_id']] = row['probability'][0] + row['probability'][1]
    with open('results/fullpool_20261119/source_registry.csv', newline='', encoding='utf-8') as stream:
        negative = [r['pool_id'] for r in csv.DictReader(stream)
                    if r['terminal_status'] == 'NO_VALID_DETECTION_PENDING_RESCUE']
    missing = [pool_id for pool_id in negative if pool_id not in clip]
    if missing:
        raise ValueError(f'CLIP screen incomplete for {len(missing)} detector-negative sources')
    flagged = sorted(pool_id for pool_id in negative if clip[pool_id] >= FLAG_PROBABILITY)
    finished = set().union(*(done_ids(path) for path in OUT.glob('dino_rescue*.jsonl')))
    out = OUT / (f'dino_rescue.shard{shard}of{shards}.jsonl' if shards > 1 else 'dino_rescue.jsonl')
    remaining = [pool_id for index, pool_id in enumerate(flagged)
                 if index % shards == shard and pool_id not in finished]
    print(f'dino: {len(flagged)} flagged of {len(negative)} negatives; {len(remaining)} remaining', flush=True)
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        folder, local_files_only=True, disable_custom_kernels=True).to(device).eval()
    post = processor.post_process_grounded_object_detection
    threshold_name = 'threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
    pool = Pool(pool_path)
    begin = time.perf_counter()
    with out.open('a', encoding='utf-8') as stream:
        for count, pool_id in enumerate(remaining):
            image = pool.image(pool_id)
            inputs = processor(images=image, text=DINO_PROMPT, return_tensors='pt').to(device)
            # On the shared GPU, bf16 autocast saves memory; CPU runs stay float32.
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                outputs = model(**inputs)
            outputs.logits, outputs.pred_boxes = outputs.logits.float(), outputs.pred_boxes.float()
            detected = post(outputs, inputs.input_ids, **{threshold_name: DINO_BOX_THRESHOLD},
                            text_threshold=DINO_TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]
            names = detected['text_labels'] if 'text_labels' in detected else detected['labels']
            proposals = []
            for box, score, name in zip(detected['boxes'].tolist(), detected['scores'].tolist(), names):
                name = str(name).lower()
                cat, dog = 'cat' in name, 'dog' in name
                if cat == dog:
                    continue
                label = 0 if cat else 1
                geometry, reason = proposal(box, *image.size, 0.70)
                origin = 'dino_rescue_valid'
                if geometry is None and reason == 'square_outside_source':
                    geometry = rescue_geometry(box, *image.size)
                    origin = 'dino_rescue_geometry_rescue'
                    reason = '' if geometry else reason
                proposals.append({'candidate_id': 'r' + candidate_key(pool_id, label, box)[1:],
                                  'label': label, 'phrase': name, 'score': round(score, 4),
                                  'box': [round(v, 2) for v in box], 'geometry': geometry,
                                  'geometry_rejection': reason, 'origin': origin if geometry else ''})
            stream.write(json.dumps({'pool_id': pool_id, 'clip_pet_probability': round(clip[pool_id], 5),
                                     'proposals': proposals, 'model_revision': provenance['revision'],
                                     'prompt': DINO_PROMPT, 'box_threshold': DINO_BOX_THRESHOLD,
                                     'text_threshold': DINO_TEXT_THRESHOLD,
                                     'precision': 'bf16-autocast' if device == 'cuda' else 'fp32-cpu'}) + '\n')
            stream.flush()
            if count % 200 == 0:
                print(f'dino {count}/{len(remaining)} {time.perf_counter() - begin:.0f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('clip', 'dino'))
    parser.add_argument('--pool', default='data/student')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.phase == 'clip':
        clip_screen(args.pool)
    else:
        dino_rescue(args.pool, args.device, args.shard, args.shards)


if __name__ == '__main__':
    main()

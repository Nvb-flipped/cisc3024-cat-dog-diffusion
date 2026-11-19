"""Zero-shot CLIP gate on exact candidate squares (local pinned CLIP, curation only).

Scores each square against two acceptable descriptions and generic specification-violation
descriptions; `gate_pass` means an acceptable description wins. The categories come from the
data specification (real photo, cat/dog only, one head, head close-up, no costume/edit), not
from particular images. Appends JSON lines; resumable by candidate id.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.common import ROOT
from src.curation.rescue import verify_model
from src.curation.vlm_screen import done as done_ids_by_candidate

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, crop  # noqa: E402

ACCEPT = ['a close-up photo of a cat\'s head and face', 'a close-up photo of a dog\'s head and face']
REJECT = ['a photo of a bear', 'a photo of a fox', 'a photo of a wolf',
          'a photo of a wild cat such as a cheetah, leopard, lion or tiger',
          'a photo of a rabbit, bird or other animal', 'a cat or dog wearing a hat, glasses or costume',
          'an advertisement or poster with text and graphics', 'a photo of a whole dog or cat body standing or lying',
          'two animals next to each other', 'a person holding or hugging a pet',
          'a drawing, cartoon or painting of an animal', 'a toy, plush or statue of an animal',
          'a blurry dark photo']
PROMPTS = ACCEPT + REJECT


class Squares(torch.utils.data.Dataset):
    def __init__(self, rows, pool_path):
        self.rows, self.pool_path, self.pool = rows, pool_path, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.pool is None:
            self.pool = Pool(self.pool_path)
        r = self.rows[i]
        image = crop(self.pool.image(r['pool_id']), int(r['x0']), int(r['y0']), int(r['size']),
                     float(r['angle'] or 0), res=224)
        return r['candidate_id'], image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    from transformers import CLIPModel, CLIPProcessor
    folder = Path('models/clip-photo-screen')
    provenance = verify_model(folder)
    rows = list(csv.DictReader(open(args.queue, newline='', encoding='utf-8')))
    finished = done_ids_by_candidate(args.out)
    rows = [r for r in rows if r['candidate_id'] not in finished]
    processor = CLIPProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
    model = CLIPModel.from_pretrained(folder, local_files_only=True).to(args.device).eval()
    with torch.inference_mode():
        text = F.normalize(model.get_text_features(**processor(text=PROMPTS, padding=True, return_tensors='pt').to(args.device)), dim=-1)
    loader = torch.utils.data.DataLoader(Squares(rows, 'data/student'), batch_size=128, num_workers=4,
                                         collate_fn=lambda b: list(zip(*b)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a', encoding='utf-8') as stream:
        for ids, images in loader:
            with torch.inference_mode():
                feats = F.normalize(model.get_image_features(**processor(images=list(images), return_tensors='pt').to(args.device)), dim=-1)
                prob = ((feats @ text.T) * model.logit_scale.exp()).softmax(-1).cpu()
            for cid, p in zip(ids, prob.tolist()):
                accept = sum(p[:len(ACCEPT)])
                stream.write(json.dumps({'candidate_id': cid, 'accept_probability': round(accept, 5),
                                         'top_reject': REJECT[max(range(len(REJECT)), key=lambda k: p[len(ACCEPT) + k])],
                                         'gate_pass': max(p[:len(ACCEPT)]) >= max(p[len(ACCEPT):]),
                                         'model_revision': provenance['revision']}) + '\n')
            stream.flush()


if __name__ == '__main__':
    main()

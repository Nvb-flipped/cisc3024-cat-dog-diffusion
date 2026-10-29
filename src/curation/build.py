"""Run a pinned, offline Grounding DINO proposal pilot on pool images only."""
import argparse
import csv
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path

import torch
from src.common import ROOT, environment, read_config, sha256, timestamp, write_json
from src.curation.geometry import proposal

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool

GRID = [(box, text) for box in (0.20, 0.30, 0.40) for text in (0.20, 0.25, 0.30)]


def candidate_key(pool_id, label, box):
    value = json.dumps([pool_id, label, [round(float(v), 3) for v in box]])
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def collect_candidates(processor, outputs, inputs, image, pool_id, occupancy):
    candidates = {}
    post = processor.post_process_grounded_object_detection
    threshold_name = 'threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
    for box_threshold, text_threshold in GRID:
        detected = post(outputs, inputs.input_ids, **{threshold_name: box_threshold},
                        text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
        names = detected['text_labels'] if 'text_labels' in detected else detected['labels']
        for box, score, name in zip(detected['boxes'].tolist(), detected['scores'].tolist(), names):
            name = str(name).lower()
            cat, dog = 'cat' in name, 'dog' in name
            if cat == dog:
                continue  # Ambiguous phrases are never silently assigned a class.
            label = 0 if cat else 1
            key = candidate_key(pool_id, label, box)
            geometry, reason = proposal(box, *image.size, occupancy)
            if key not in candidates:
                candidates[key] = {'candidate_id': key, 'pool_id': pool_id, 'label': label,
                    'detector_phrase': name, 'score': score, 'box': box,
                    'head_width': box[2] - box[0], 'head_height': box[3] - box[1],
                    'geometry': geometry, 'geometry_rejection': reason, 'eligible_settings': []}
            candidates[key]['eligible_settings'].append([box_threshold, text_threshold])
    return list(candidates.values())


def write_tables(out, records, seed):
    candidates = [c for record in records for c in record['candidates']]
    columns = ['candidate_id', 'pool_id', 'label', 'score', 'detector_phrase', 'head_width',
               'head_height', 'box', 'x0', 'y0', 'size', 'angle', 'geometry_rejection', 'eligible_settings']
    with (out / 'candidates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            row = {k: candidate[k] for k in columns if k in candidate}
            row.update(candidate['geometry'] or {})
            for field in ['box', 'eligible_settings']:
                row[field] = json.dumps(row[field])
            writer.writerow(row)
    rng = random.Random(seed + 1)
    proposed = sorted([c['candidate_id'] for c in candidates if c['geometry']])
    rejected = sorted([r['pool_id'] for r in records if not any(c['geometry'] for c in r['candidates'])
                       and not r.get('error')])
    write_json(out / 'calibration_selection.json', {
        'seed': seed + 1, 'proposed_candidate_ids': rng.sample(proposed, min(200, len(proposed))),
        'rejected_pool_ids': rng.sample(rejected, min(200, len(rejected))),
        'note': 'Fixed pool-only exploratory quality sample. Student review pending; not the official audit.'})
    summary = {'processed_images': len(records), 'detector_errors': sum(bool(r.get('error')) for r in records),
        'proposals': len(candidates), 'geometrically_valid_proposals': sum(bool(c['geometry']) for c in candidates),
        'images_with_valid_proposals': sum(any(c['geometry'] for c in r['candidates']) for r in records),
        'total_detector_seconds': sum(r['detector_seconds'] for r in records),
        'status': 'Proposals only; no student decisions, no frozen training manifest'}
    write_json(out / 'summary.json', summary)
    print(summary, flush=True)


def main():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/curation.yaml')
    p.add_argument('--limit', type=int)
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    config = read_config(args.config)
    folder = Path(config['model_path'])
    provenance = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    if config.get('revision') and provenance['revision'] != config['revision']:
        raise ValueError('Detector revision mismatch')
    for name, expected in provenance['files'].items():
        if sha256(folder / name) != expected:
            raise ValueError(f'Detector file changed: {name}')
    pool = Pool(config['pool'])
    ids = sorted(pool.index)
    rng = random.Random(config['seed'])
    rng.shuffle(ids)
    ids = ids[:args.limit if args.limit is not None else config['pilot_size']]
    out = args.out or Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    expected = {'config': config, 'pool_ids': ids, 'detector_revision': provenance['revision'],
                'index_sha256': sha256(Path(config['pool']) / 'index.csv'),
                'prompt': 'a cat head. a dog head.', 'threshold_grid': [list(x) for x in GRID]}
    header = out / 'pilot_spec.json'
    if header.exists():
        if not args.resume or json.loads(header.read_text()) != expected:
            raise ValueError('Output exists or resume specification differs; choose a fresh --out')
    else:
        write_json(header, expected)
        write_json(out / 'environment.json', environment() | {'started': timestamp()})
    records_path = out / 'detections.jsonl'
    records = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines()] \
        if records_path.exists() else []
    done = {r['pool_id'] for r in records}
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        folder, local_files_only=True, disable_custom_kernels=True).to(args.device).eval()
    with records_path.open('a', encoding='utf-8') as stream:
        for pool_id in ids:
            if pool_id in done:
                continue
            image = pool.image(pool_id)
            start = time.perf_counter()
            record = {'pool_id': pool_id, 'candidates': [], 'error': None}
            try:
                inputs = processor(images=image, text=expected['prompt'], return_tensors='pt').to(args.device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                record['candidates'] = collect_candidates(processor, outputs, inputs, image,
                                                            pool_id, config['occupancy'])
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['detector_seconds'] = time.perf_counter() - start
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            records.append(record)
            if record['error']:
                write_tables(out, records, config['seed'])
                raise RuntimeError(f'Pilot stopped on {pool_id}: {record["error"]}')
            if len(records) % 25 == 0:
                print(f'{len(records)}/{len(ids)}; {sum(len(r["candidates"]) for r in records)} proposals', flush=True)
    write_tables(out, records, config['seed'])


if __name__ == '__main__':
    main()

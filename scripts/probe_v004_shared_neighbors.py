"""Compare saved seed-zero exports against the same exact v002 pixel reference bank."""

import json
from pathlib import Path

import torch

from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.neighbors import pixel_neighbors, reference_bank
from src.evaluation.select import validated_result
from src.release.check import check_files
from starter_kit import evaluate as official


CONTROL = Path('results/experiments/capacity_32_seed17_v002/comparison.json')
TRIAL = Path('results/experiments/v004_coverage/screening.json')
OUT = Path('results/experiments/v004_coverage/shared_pixel_neighbors.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def training_pixels(export, pool):
    manifest = Path(export) / 'manifest.csv'
    checked = official.check_manifest(manifest, pool)
    if checked['errors'] or checked['stats']['valid_rows'] != checked['stats']['rows']:
        raise ValueError(f'Invalid diagnostic manifest: {manifest}')
    return official.rebuild(checked['rows'], pool)[0], sha256(manifest)


def summary(images, labels, reference):
    distances, _ = pixel_neighbors(images, reference)
    ratios = distances[:, 0] / distances[:, 1].clamp_min(1e-8)
    flags = ratios < official.MEMORIZATION_RATIO
    return dict(rate=flags.float().mean().item(), count=int(flags.sum()),
                cat_count=int(flags[labels == 0].sum()), dog_count=int(flags[labels == 1].sum()),
                median_ratio=ratios.median().item(), reference_rows=len(reference))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    verify_starter()
    control = read(CONTROL)
    trial = next(item for item in read(TRIAL)['screened'] if item['update'] == 30000)
    pool = official.Pool(official.pool_dir())
    shared_pixels, shared_manifest = training_pixels(control['exported'], pool)
    expanded_pixels, expanded_manifest = training_pixels(trial['exported'], pool)
    shared_bank, _ = reference_bank(shared_pixels)
    expanded_bank, _ = reference_bank(expanded_pixels)
    findings = {}
    for name, run in [('v002', control), ('v004', trial)]:
        path = Path(next(row['path'] for row in run['evidence'] if row['seed'] == 0))
        exported = Path(run['exported'])
        hashes = check_files(exported)['file_sha256']
        result = validated_result(path, 0, exported)
        device = result['protocol']['device']
        module = official.load_submission(exported)
        model = module.load_model(torch.device(device))
        images, labels, _ = official.sample_submission(module, model, 1000, 0, 500, device)
        shared = summary(images, labels, shared_bank)
        expanded = summary(images, labels, expanded_bank)
        own = shared if name == 'v002' else expanded
        if abs(own['rate'] - result['metrics']['memorization_rate']) > 1e-6 or \
                own['reference_rows'] != result['metrics']['training_images_deduplicated'] or \
                hashes != check_files(exported)['file_sha256']:
            raise ValueError(f'The official {name} memorization replay disagrees with its receipt')
        findings[name] = dict(shared_v002=shared, expanded_v004=expanded, official_receipt_sha256=sha256(path))
        del model
    write_json(OUT, dict(created=timestamp(), scope='Local generated-to-pool diagnostic. No dev inputs.',
                         seed=0, n_per_class=1000, shared_manifest_sha256=shared_manifest,
                         expanded_manifest_sha256=expanded_manifest, results=findings))
    print(json.dumps(findings))


if __name__ == '__main__':
    main()

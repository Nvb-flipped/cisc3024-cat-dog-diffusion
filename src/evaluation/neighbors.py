"""Replay the official pixel-neighbor test and retain local diagnostic examples."""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from scripts.setup_inputs import verify_starter
from src.common import ROOT, sha256, write_json
from src.evaluation.select import validated_result
from src.release.check import check_files
from starter_kit import evaluate as official


def reference_bank(training):
    """Match each official representative back to its manifest row and orientation."""
    mirrored = torch.cat([training, training.flip(3)])
    representatives = official.dedup(mirrored)
    if len(representatives) < 2:
        raise ValueError('At least two official reference representatives are required')
    origins = {}
    for index, tile in enumerate(mirrored):
        origins.setdefault(tile.numpy().tobytes(), (index % len(training), index >= len(training)))
    return representatives, [origins[tile.numpy().tobytes()] for tile in representatives]


def pixel_neighbors(images, reference):
    distances, indices = [], []
    flat_reference = reference.flatten(1).float()
    for batch in images.split(500):
        nearest = torch.cdist(batch.flatten(1).float(), flat_reference).topk(2, dim=1, largest=False)
        distances.append(nearest.values)
        indices.append(nearest.indices)
    return torch.cat(distances), torch.cat(indices)


def diagnostic(exported, result_path, output, *, seed=0, data_root=ROOT / 'data', device='cuda'):
    exported, result_path, output = map(Path, (exported, result_path, output))
    if output.exists():
        raise FileExistsError(f'Diagnostic already exists: {output}')
    verify_starter()
    hashes = check_files(exported)['file_sha256']
    result = validated_result(result_path, seed, exported)
    if result['protocol']['device'] != device:
        raise ValueError('Diagnostic device must match the original evaluation device')

    pool = official.Pool(official.pool_dir(data_root))
    manifest = official.check_manifest(exported / 'manifest.csv', pool)
    if manifest['errors'] or manifest['stats']['valid_rows'] != manifest['stats']['rows']:
        raise ValueError('Diagnostic requires a complete official manifest rebuild')
    manifest_rows = manifest['rows']
    training, _ = official.rebuild(manifest_rows, pool)
    reference, origins = reference_bank(training)

    module = official.load_submission(exported)
    model = module.load_model(torch.device(device))
    images, labels, _ = official.sample_submission(module, model, 1000, seed, 500, device)
    distances, indices = pixel_neighbors(images, reference)
    ratios = distances[:, 0] / distances[:, 1].clamp_min(1e-8)
    flagged = ratios < official.MEMORIZATION_RATIO
    replay = {'memorization_rate': flagged.float().mean().item(),
              'nn_distance_median': distances[:, 0].median().item(),
              'nn_ratio_median': ratios.median().item(),
              'training_images_deduplicated': len(reference)}
    if any(abs(value - result['metrics'][name]) > 1e-6 for name, value in replay.items()):
        raise ValueError(f'Pixel-neighbor replay differs from the official receipt: {replay}')
    if check_files(exported)['file_sha256'] != hashes:
        raise ValueError('Export changed during the diagnostic')

    records = []
    for index, label in enumerate(labels.tolist()):
        first, second = indices[index].tolist()
        row, flipped = origins[first]
        next_row, next_flipped = origins[second]
        records.append({'generated_index': index, 'label': label,
                        'nearest_manifest_row': row + 1, 'nearest_source': manifest_rows[row]['pool_id'],
                        'nearest_flipped': flipped, 'second_manifest_row': next_row + 1,
                        'second_source': manifest_rows[next_row]['pool_id'], 'second_flipped': next_flipped,
                        'd1': distances[index, 0].item(), 'd2': distances[index, 1].item(),
                        'ratio': ratios[index].item(), 'flagged': bool(flagged[index])})

    output.mkdir(parents=True)
    with (output / 'neighbors.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    examples = []
    for label in range(2):
        for marked in (True, False):
            matching = [r for r in records if r['label'] == label and r['flagged'] == marked][:4]
            for record in matching:
                index = record['generated_index']
                first, second = indices[index].tolist()
                for suffix, tile in [('generated', images[index]), ('nearest', reference[first]),
                                     ('second', reference[second])]:
                    pixels = (tile.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
                    Image.fromarray(pixels).save(output / f'{index:04d}_{suffix}.png')
            examples.extend(record['generated_index'] for record in matching)
    write_json(output / 'replay.json', {'seed': seed, 'official_result': str(result_path),
        'official_result_sha256': sha256(result_path), 'export_sha256': hashes,
        'diagnostic_code_sha256': sha256(__file__), 'metrics': replay, 'examples': examples,
        'selection': 'First four generated indices in each requested-class and flagged-status group',
        'scope': 'Local pixel-neighbor replay only. No dev images, classifier or Inception features were used.'})
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(diagnostic(args.submission, args.result, args.out, seed=args.seed, device=args.device))


if __name__ == '__main__':
    main()

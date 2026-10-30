import copy
import numpy as np
from PIL import Image
import pytest
import torch
from src.training.data import load_manifest_data
from src.training.step import optimizer_step
from src.training.train import atomic_checkpoint
from src.training.lineage import reviewed_manifest_provenance
from src.curation.pool_fingerprint import fingerprint
from src.common import sha256, write_json
from submission.diffusion import cosine_alpha_bar
from submission.model import ConditionalUNet


def pool_fixture(tmp_path):
    folder = tmp_path / 'pool'
    (folder / 'images').mkdir(parents=True)
    y, x = np.mgrid[:120, :120]
    pixels = np.stack([x, y, (x + y) % 255], axis=-1).astype('uint8')
    Image.fromarray(pixels).save(folder / 'images/source.jpg')
    (folder / 'index.csv').write_text('pool_id,width,height\nsource,120,120\n')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('pool_id,x0,y0,size,angle,label\nsource,20,20,60,20,0\n')
    return folder, manifest


def test_cache_rebuild_is_exact_and_content_tampering_fails(tmp_path):
    folder, manifest = pool_fixture(tmp_path)
    cache_dir = tmp_path / 'cache'
    first, labels, digest, stats = load_manifest_data(manifest, folder, cache_dir)
    second, repeated_labels, _, _ = load_manifest_data(manifest, folder, cache_dir)
    assert torch.equal(first, second) and torch.equal(labels, repeated_labels)
    assert first.shape == (1, 3, 32, 32) and stats['rotated'] == 1
    cache = cache_dir / f'v2_{digest}.npz'
    with cache.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='content hash mismatch'):
        load_manifest_data(manifest, folder, cache_dir)


def test_changed_source_cannot_reuse_a_cache(tmp_path):
    folder, manifest = pool_fixture(tmp_path)
    load_manifest_data(manifest, folder, tmp_path / 'cache')
    Image.new('RGB', (120, 120), 'red').save(folder / 'images/source.jpg')
    with pytest.raises(ValueError, match='provenance mismatch'):
        load_manifest_data(manifest, folder, tmp_path / 'cache')


def test_frozen_review_lineage_rejects_changed_pool_even_without_an_old_cache(tmp_path):
    folder, manifest = pool_fixture(tmp_path)
    write_json(manifest.with_suffix('.provenance.json'), {'origin': 'student_reviewed_manifest',
        'scope': 'SYNTHETIC UNIT FIXTURE', 'pool_content_sha256': fingerprint(folder)[0],
        'manifests': [{'sha256': sha256(manifest)}]})
    assert reviewed_manifest_provenance(manifest)
    load_manifest_data(manifest, folder, tmp_path / 'cache')
    Image.new('RGB', (120, 120), 'red').save(folder / 'images/source.jpg')
    with pytest.raises(ValueError, match='Source pool differs'):
        load_manifest_data(manifest, folder, tmp_path / 'new_cache')


def test_baseline_requires_frozen_reviewed_provenance(tmp_path):
    folder, manifest = pool_fixture(tmp_path)
    with pytest.raises(ValueError, match='requires the frozen'):
        reviewed_manifest_provenance(manifest)
    write_json(manifest.with_suffix('.provenance.json'), {'diagnostic_only': True})
    with pytest.raises(ValueError, match='diagnostic, changed'):
        reviewed_manifest_provenance(manifest)


@pytest.mark.parametrize('row', [
    'missing,0,0,40,0,0', 'source,0,0,39,0,0', 'source,0,0,40,46,0',
    'source,0,0,40,45,0', 'source,20,20,40,0,7'])
def test_invalid_provenance_never_reaches_training(row, tmp_path):
    folder, manifest = pool_fixture(tmp_path)
    manifest.write_text('pool_id,x0,y0,size,angle,label\n' + row + '\n')
    with pytest.raises(ValueError, match='Invalid manifest'):
        load_manifest_data(manifest, folder, tmp_path / 'cache')


def test_optimizer_dropout_and_rng_resume_exactly(tmp_path):
    torch.manual_seed(17)
    model = ConditionalUNet(channels=[8, 16, 24, 32], blocks_per_level=1,
                            conditioning_dim=32, dropout=.10).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    generator = torch.Generator().manual_seed(29)
    images = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    schedule = cosine_alpha_bar()
    settings = dict(effective_batch=4, microbatch=2, learning_rate=2e-4,
                    warmup=4, grad_clip=1., precision='fp32', null_probability=.1)
    for update in [1, 2]:
        optimizer_step(model, optimizer, images, labels, schedule, generator, settings, update)
    path = tmp_path / 'state.pt'
    atomic_checkpoint(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'private_rng': generator.get_state(), 'global_rng': torch.get_rng_state()})
    for update in [3, 4]:
        optimizer_step(model, optimizer, images, labels, schedule, generator, settings, update)
    expected = copy.deepcopy(model.state_dict())
    restored = ConditionalUNet(channels=[8, 16, 24, 32], blocks_per_level=1,
                               conditioning_dim=32, dropout=.10).train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-4)
    stored = torch.load(path, weights_only=True)
    restored.load_state_dict(stored['model'])
    restored_optimizer.load_state_dict(stored['optimizer'])
    generator.set_state(stored['private_rng'])
    torch.set_rng_state(stored['global_rng'])
    for update in [3, 4]:
        optimizer_step(restored, restored_optimizer, images, labels, schedule, generator, settings, update)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in expected.items())

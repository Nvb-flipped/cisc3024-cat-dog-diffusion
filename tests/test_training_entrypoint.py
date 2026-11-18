"""Exercise CLI bookkeeping on synthetic tensors; not a data or model-quality experiment."""
import importlib
import json
import sys
import pytest
import torch
import yaml
from src.common import sha256

training = importlib.import_module('src.training.train')


def test_training_checkpoint_holdout_and_new_branch_resume(tmp_path, monkeypatch):
    config = {'model': dict(channels=[8, 16, 24, 32], blocks_per_level=1,
                conditioning_dim=32, attention_heads=4, dropout=.1),
        'diffusion': dict(timesteps=1000, cosine_offset=.008),
        'sampling': dict(steps=2, guidance=1, microbatch=2, execution_batch=2),
        'training': dict(updates=2, effective_batch=2, microbatch=2, learning_rate=.0002,
            betas=[.9,.999], weight_decay=.0001, warmup=1, grad_clip=1, ema_decay=.9,
            null_probability=.1, checkpoint_every=1, log_every=1, precision='fp32', threads=2)}
    configuration = tmp_path / 'synthetic_config.yaml'
    configuration.write_text(yaml.safe_dump(config))
    manifest = tmp_path / 'synthetic.csv'
    manifest.write_text('pool_id\nfixture_training\n')
    held = manifest.with_suffix('.validation.csv')
    held.write_text('pool_id\nfixture_heldout\n')
    def synthetic_loader(path, pool):
        is_heldout = str(path).endswith('.validation.csv')
        return (torch.zeros(2, 3, 32, 32), torch.tensor([0, 1]), sha256(path),
                {'cat': 1 if is_heldout else 500, 'dog': 1 if is_heldout else 500,
                 'pool_content_sha256': 'synthetic_loader_stub_not_a_real_pool'})
    monkeypatch.setattr(training, 'load_manifest_data', synthetic_loader)
    monkeypatch.setattr(training, 'reviewed_manifest_provenance', lambda _: 'synthetic_stub_no_student_claim')
    monkeypatch.setattr(training, 'environment', lambda: {'scope': 'synthetic test fixture'})
    def run(out, *extra):
        monkeypatch.setattr(sys, 'argv', ['training', '--config', str(configuration), '--manifest', str(manifest),
            '--pool', 'synthetic_stub', '--out', str(out), '--device', 'cpu', *map(str, extra)])
        training.main()
    first = tmp_path / 'first_run'
    run(first)
    state = torch.load(first / 'update_000002.pt', weights_only=True)
    assert state['validation_manifest_sha256'] == sha256(held)
    assert (first / 'validation_000002.json').exists()
    assert len(list(first.glob('loss_*.csv'))) == 1
    with pytest.raises(ValueError, match='fresh --out'):
        run(first, '--resume', first / 'update_000001.pt', '--updates', 3)
    branch = tmp_path / 'new_branch'
    run(branch, '--resume', first / 'update_000002.pt', '--updates', 3)
    segment = json.loads(next(branch.glob('segment_*.json')).read_text())
    assert segment['parent_checkpoint_sha256'] == sha256(first / 'update_000002.pt')
    assert segment['initial_update'] == 2
    assert (branch / 'update_000003.pt').exists()
    assert torch.load(first / 'update_000002.pt', weights_only=True)['update'] == 2
    lower_rate = tmp_path / 'lower_rate'
    run(lower_rate, '--resume', first / 'update_000002.pt', '--updates', 3, '--resume-learning-rate', .0001)
    lower_state = torch.load(lower_rate / 'update_000003.pt', weights_only=True)
    assert lower_state['config']['training']['learning_rate'] == .0001
    assert lower_state['optimizer']['param_groups'][0]['lr'] == .0001
    lower_segment = json.loads(next(lower_rate.glob('segment_*.json')).read_text())
    assert lower_segment['learning_rate_transition'] == dict(previous=.0002, current=.0001, update=2)
    with pytest.raises(ValueError, match='fresh --out'):
        run(lower_rate, '--resume', first / 'update_000002.pt', '--updates', 4, '--resume-learning-rate', .0001)
    with pytest.raises(ValueError, match='requires --resume'):
        run(tmp_path / 'invalid', '--resume-learning-rate', .0001)

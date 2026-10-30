import pytest
import torch
from src.training.validation import validation_loss, require_disjoint_sources
from submission.diffusion import cosine_alpha_bar


class ZeroPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.9)

    def forward(self, x, t, y):
        return self.dropout(torch.zeros_like(x))


def test_validation_preserves_training_mode_and_global_rng():
    model = ZeroPredictor().train()
    images = torch.zeros(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    schedule = cosine_alpha_bar()
    before = torch.get_rng_state().clone()
    first = validation_loss(model, images, labels, schedule, timesteps=[50, 750])
    assert torch.equal(before, torch.get_rng_state()) and model.training
    assert first == validation_loss(model, images, labels, schedule, timesteps=[50, 750])
    assert .9 < first['mean_mse'] < 1.1
    assert first != validation_loss(model, images, labels, schedule, seed=999, timesteps=[50, 750])


def test_validation_keeps_original_eval_mode():
    model = ZeroPredictor().eval()
    validation_loss(model, torch.zeros(1, 3, 32, 32), torch.tensor([0]), cosine_alpha_bar())
    assert not model.training


def test_same_source_with_different_crop_is_still_split_leakage(tmp_path):
    train, held = tmp_path / 'train.csv', tmp_path / 'held.csv'
    train.write_text('pool_id,x0\nfixture,0\n')
    held.write_text('pool_id,x0\nfixture,40\n')
    with pytest.raises(ValueError, match='share pool sources'):
        require_disjoint_sources(train, held)
    held.write_text('pool_id,x0\nother,0\n')
    require_disjoint_sources(train, held)

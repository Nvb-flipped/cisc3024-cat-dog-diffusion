"""Optional left-right mirroring of exact manifest crops in the training step."""
import pytest
import torch

from src.training import step


class Recorder(torch.nn.Module):
    """Tiny noise predictor that records the clean images implied by its inputs."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, x, t, y, hint=None):
        self.inputs.append((x.detach().clone(), t.detach().clone()))
        return x * self.weight


def run_step(images, labels, flip, seed=5):
    model = Recorder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    settings = dict(effective_batch=8, microbatch=8, learning_rate=1e-3, warmup=1, grad_clip=1.0,
                    precision='fp32', null_probability=0.0)
    if flip is not None:
        settings['horizontal_flip'] = flip
    step.validate_loss_weighting(settings)
    alpha_bar = torch.linspace(.999, .001, 1000)
    generator = torch.Generator().manual_seed(seed)
    step.optimizer_step(model, optimizer, images, labels, alpha_bar, generator, settings, 1)
    return model.inputs[0], generator.get_state()


def test_disabled_flip_preserves_the_original_random_stream():
    images = torch.rand(6, 3, 32, 32) * 2 - 1
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    (x_default, t_default), state_default = run_step(images, labels, None)
    (x_zero, t_zero), state_zero = run_step(images, labels, 0)
    assert torch.equal(x_default, x_zero) and torch.equal(t_default, t_zero)
    assert torch.equal(state_default, state_zero)


def test_enabled_flip_only_mirrors_exact_crops():
    torch.manual_seed(0)
    images = torch.rand(4, 3, 32, 32) * 2 - 1
    labels = torch.tensor([0, 1, 0, 1])
    mirrored_seen = original_seen = 0
    for seed in range(12):
        settings = dict(effective_batch=64, microbatch=64, learning_rate=1e-3, warmup=1, grad_clip=1.0,
                        precision='fp32', null_probability=0.0, horizontal_flip=0.5)
        generator = torch.Generator().manual_seed(seed)
        # Replay the step's own draws to recover each clean target exactly.
        replay = torch.Generator().manual_seed(seed)
        indices = torch.randint(len(labels), (64,), generator=replay)
        mirrored = torch.rand(64, generator=replay) < .5
        t = torch.randint(1000, (64,), generator=replay)
        noise = torch.randn((64, 3, 32, 32), generator=replay)
        model = Recorder()
        step.optimizer_step(model, torch.optim.SGD(model.parameters(), lr=0.0), images, labels,
                            torch.linspace(.999, .001, 1000), generator, settings, 1)
        x, recorded_t = model.inputs[0]
        assert torch.equal(recorded_t, t)
        alpha = torch.linspace(.999, .001, 1000)[t][:, None, None, None]
        clean = (x - (1 - alpha).sqrt() * noise) / alpha.sqrt()
        expected = torch.where(mirrored[:, None, None, None], images[indices].flip(-1), images[indices])
        torch.testing.assert_close(clean, expected, atol=2e-4, rtol=0)
        mirrored_seen += int(mirrored.sum())
        original_seen += int((~mirrored).sum())
    assert mirrored_seen > 0 and original_seen > 0


@pytest.mark.parametrize('value', [-0.1, 1.5, 'yes', float('nan')])
def test_flip_probability_is_validated(value):
    with pytest.raises(ValueError):
        step.validate_loss_weighting({'horizontal_flip': value})

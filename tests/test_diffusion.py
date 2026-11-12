import torch
import pytest
from torch import nn
from submission.diffusion import (add_noise, cosine_alpha_bar, ddim_indices,
                                  reconstruct_clean, reconstruct_clean_from_velocity,
                                  sample, velocity_target)
from submission.model import ConditionalUNet


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, y):
        return x * 0.1 + y[:, None, None, None] * 0.02 + self.anchor


def test_forward_inversion_at_endpoints_and_middle():
    schedule = cosine_alpha_bar()
    assert schedule.shape == (1000,)
    assert torch.isfinite(schedule).all() and (schedule > 0).all() and (schedule < 1).all()
    assert torch.all(schedule[1:] < schedule[:-1])
    generator = torch.Generator().manual_seed(3024)
    clean = torch.randn(4, 3, 32, 32, generator=generator, dtype=torch.float64)
    noise = torch.randn(4, 3, 32, 32, generator=generator, dtype=torch.float64)
    times = torch.tensor([0, 1, 500, 999])
    noisy = add_noise(clean, noise, times, schedule.double())
    restored = reconstruct_clean(noisy, noise, schedule.double()[times, None, None, None])
    torch.testing.assert_close(restored, clean, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('steps', [1, 2, 50, 100, 200, 1000])
def test_ddim_indices(steps):
    indices = ddim_indices(1000, steps)
    assert len(indices) == len(set(indices)) == steps
    assert indices[0] == 999
    assert all(a > b for a, b in zip(indices, indices[1:]))
    if steps > 1:
        assert indices[-1] == 0


def test_sampler_oracle_reaches_clean_endpoint():
    class Oracle(ProbeModel):
        def forward(self, x, t, y):
            a = cosine_alpha_bar()[t, None, None, None]
            return (x - a.sqrt() * 0.25) / (1 - a).sqrt()
    images = sample(Oracle().eval(), torch.tensor([0, 1]), 42, steps=10, guidance=1)
    torch.testing.assert_close(images, torch.full_like(images, 0.625), rtol=0, atol=3e-4)


def test_velocity_inverse_includes_pure_noise_endpoint():
    rng = torch.Generator().manual_seed(17)
    clean = torch.randn(3, 3, 32, 32, generator=rng)
    noise = torch.randn(clean.shape, generator=rng)
    alpha = torch.tensor([1., .5, 0.])
    a = alpha[:, None, None, None]
    noisy = add_noise(clean, noise, torch.arange(3), alpha)
    velocity = velocity_target(clean, noise, a)
    torch.testing.assert_close(reconstruct_clean_from_velocity(noisy, velocity, a), clean)


def test_velocity_oracle_reaches_clean_endpoint():
    class Oracle(ProbeModel):
        def forward(self, x, t, y):
            a = cosine_alpha_bar()[t, None, None, None]
            clean = torch.full_like(x, .25)
            noise = (x - a.sqrt() * clean) / (1 - a).sqrt()
            return velocity_target(clean, noise, a)
    images = sample(Oracle().eval(), torch.tensor([0, 1]), 42, steps=10, guidance=1,
                    prediction_type='velocity')
    torch.testing.assert_close(images, torch.full_like(images, .625))


def test_guidance_interval_uses_conditional_prediction_outside_bounds():
    class GuidanceProbe(ProbeModel):
        def __init__(self):
            super().__init__()
            self.unconditional_times = []

        def forward(self, x, t, y):
            if y.eq(2).all():
                self.unconditional_times.append(t[0].item())
            return super().forward(x, t, y)
    model = GuidanceProbe().eval()
    sample(model, torch.tensor([0, 1]), 42, timesteps=9, steps=5, start_timestep=8,
           guidance=2.5, guidance_min_timestep=2, guidance_max_timestep=6)
    assert model.unconditional_times == [6, 4, 2]


@pytest.mark.parametrize('count', [0, 1, 2, 20, 50, 500])
def test_public_shapes_and_private_rng(count):
    model = ProbeModel().eval()
    labels = torch.arange(count) % 2
    before = torch.get_rng_state().clone()
    first = sample(model, labels, 17, steps=2, microbatch=7)
    assert torch.equal(before, torch.get_rng_state())
    assert first.shape == (count, 3, 32, 32) and first.dtype == torch.float32
    assert torch.isfinite(first).all() and (first >= 0).all() and (first <= 1).all()
    assert torch.equal(first, sample(model, labels, 17, steps=2, microbatch=7))
    if count:
        assert not torch.equal(first, sample(model, labels, 29, steps=2, microbatch=7))


def test_seed_stream_label_isolation_and_microbatch_boundaries():
    model = ProbeModel().eval()
    labels = torch.arange(20) % 2
    first = sample(model, labels, 5, steps=4, microbatch=7)
    changed = labels.clone()
    changed[1:] = 1 - changed[1:]
    other = sample(model, changed, 5, steps=4, microbatch=7)
    assert torch.equal(first[0], other[0])
    assert not torch.equal(first[1:], other[1:])
    single = sample(model, labels[:1], 5, steps=4, microbatch=7)
    assert torch.equal(first[:1], single)
    regrouped = sample(model, labels, 5, steps=4, microbatch=3)
    assert torch.equal(first, regrouped)


def test_actual_unet_shape_gradients_and_label_effect():
    torch.manual_seed(17)
    model = ConditionalUNet(channels=[16, 32, 48, 64], blocks_per_level=1,
                            conditioning_dim=64, dropout=0)
    x = torch.randn(2, 3, 32, 32)
    t, labels = torch.tensor([0, 999]), torch.tensor([0, 1])
    output = model(x, t, labels)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    first = sample(model, labels, 2, steps=2)
    changed = sample(model, torch.tensor([0, 0]), 2, steps=2)
    assert torch.equal(first[0], changed[0])
    assert not torch.equal(first[1], changed[1])
    single = sample(model, labels[:1], 2, steps=2)
    torch.testing.assert_close(first[:1], single, rtol=0, atol=2e-5)


def test_microbatch_regrouping_keeps_each_physical_slot():
    class SlotProbe(ProbeModel):
        def forward(self, x, t, y):
            a = cosine_alpha_bar()[t, None, None, None]
            clean = torch.arange(len(x), dtype=x.dtype)[:, None, None, None] * 0.125
            return (x - a.sqrt() * clean) / (1 - a).sqrt()

    model = SlotProbe().eval()
    labels = torch.arange(11) % 2
    default = sample(model, labels, 5, steps=3, guidance=1, microbatch=4, execution_batch=4)
    regrouped = sample(model, labels, 5, steps=3, guidance=1, microbatch=3, execution_batch=4)
    assert torch.equal(default, regrouped)


def test_invalid_sampler_inputs_fail():
    model = ProbeModel().eval()
    for bad in [torch.tensor([2]), torch.tensor([0.0]), torch.tensor([[0]])]:
        with pytest.raises(ValueError):
            sample(model, bad, 0, steps=1)
    with pytest.raises(ValueError):
        sample(model.train(), torch.tensor([0]), 0, steps=1)

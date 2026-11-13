import torch
import pytest
from torch import nn
from submission.diffusion import (add_noise, cosine_alpha_bar, ddim_indices, dpmpp_2m_coefficients,
                                  ddim_step_variance,
                                  reconstruct_clean, reconstruct_clean_from_velocity,
                                  rescale_guidance, sample, velocity_target)
from submission.model import ConditionalUNet
from submission.edm import precondition


@pytest.mark.parametrize('patch_size', [4, 8])
@pytest.mark.parametrize('patch_detail', [False, True])
def test_patch_denoiser_geometry_and_seeded_sampling(patch_size, patch_detail):
    x = torch.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).float()
    packed = torch.nn.functional.pixel_unshuffle(x, patch_size)
    assert packed.shape == (2, 3 * patch_size ** 2, 32 // patch_size, 32 // patch_size)
    assert torch.equal(torch.nn.functional.pixel_shuffle(packed, patch_size), x)

    model = ConditionalUNet(channels=(8, 16, 24, 32), blocks_per_level=1,
                            conditioning_dim=32, patch_size=patch_size, patch_detail=patch_detail).eval()
    labels = torch.tensor([0, 1, 0])
    state = torch.get_rng_state().clone()
    options = dict(steps=3, execution_batch=2, guidance=2.5)
    first = sample(model, labels, 17, **options)
    assert first.shape == (3, 3, 32, 32)
    assert torch.isfinite(first).all() and (first >= 0).all() and (first <= 1).all()
    assert torch.equal(first[:1], sample(model, labels[:1], 17, **options))
    assert torch.equal(first, sample(model, labels, 17, microbatch=1, **options))
    changed_labels = labels.clone()
    changed_labels[1:] = 1 - changed_labels[1:]
    assert torch.equal(first[0], sample(model, changed_labels, 17, **options)[0])
    assert torch.equal(state, torch.get_rng_state())


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, y):
        return x * 0.1 + y[:, None, None, None] * 0.02 + self.anchor


def test_guidance_rescale_matches_each_conditional_scale_independently():
    conditional = torch.tensor([[0., 1., 2.], [1., 3., 5.]]).reshape(2, 1, 1, 3)
    guided = conditional * torch.tensor([2., 5.]).reshape(2, 1, 1, 1)
    torch.testing.assert_close(rescale_guidance(guided, conditional, 1.), conditional)
    torch.testing.assert_close(rescale_guidance(guided, conditional, .5), (guided + conditional) / 2)
    assert torch.equal(rescale_guidance(guided, conditional, 0.), guided)
    assert torch.isfinite(rescale_guidance(torch.ones_like(guided), conditional, 1.)).all()


def test_dpmpp_first_order_matches_deterministic_ddim_and_uses_positive_step_ratios():
    schedule = cosine_alpha_bar()
    indices = [999, 980, 500, 0]
    coefficients = dpmpp_2m_coefficients(schedule, indices)
    alpha = schedule[indices].double().sqrt()
    sigma = (1 - schedule[indices].double()).sqrt()
    h = (alpha / sigma).log().diff()
    torch.testing.assert_close(coefficients[:, 0], (sigma[1:] / sigma[:-1]).float())
    torch.testing.assert_close(coefficients[:, 1], (alpha[1:] - sigma[1:] / sigma[:-1] * alpha[:-1]).float())
    torch.testing.assert_close(coefficients[:, 2], torch.cat((h.new_zeros(1), h[1:] / (2 * h[:-1]))).float())
    assert torch.isfinite(coefficients).all()
    labels = torch.tensor([0, 1])
    model = ProbeModel().eval()
    torch.testing.assert_close(sample(model, labels, 17, steps=2, solver='dpmpp_2m'),
                               sample(model, labels, 17, steps=2, solver='ddim'))


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


def test_edm_oracle_reaches_clean_endpoint():
    class Oracle(ProbeModel):
        def forward(self, x, t, y):
            assert torch.isfinite(t).all()
            sigma = (t * 4).exp()[:, None, None, None]
            scale_in, skip, scale_out = precondition(sigma)
            return (0.25 - skip * (x / scale_in)) / scale_out

    images = sample(Oracle().eval(), torch.tensor([0, 1]), 42, steps=10, guidance=1,
                    prediction_type='edm')
    torch.testing.assert_close(images, torch.full_like(images, .625), atol=1e-6, rtol=0)


def test_edm_uses_stable_slots_and_private_rng():
    model = ProbeModel().eval()
    labels = torch.tensor([0, 1, 1, 0, 1])
    options = dict(steps=6, guidance=2.5, execution_batch=4, prediction_type='edm')
    state = torch.get_rng_state().clone()
    images = sample(model, labels, 42, **options)
    assert torch.equal(images, sample(model, labels, 42, microbatch=3, **options))
    assert torch.equal(images[:1], sample(model, labels[:1], 42, **options))
    assert torch.equal(images[:1], sample(model, torch.tensor([0, 0, 0, 1, 0]), 42, **options)[:1])
    assert torch.equal(state, torch.get_rng_state())


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


def test_seeded_ddim_noise_variance_and_output_isolation():
    alpha, previous = torch.tensor(.2), torch.tensor(.6)
    torch.testing.assert_close(ddim_step_variance(alpha, previous, .5), torch.tensor(1/12))
    model = ProbeModel().eval()
    labels = torch.arange(11) % 2
    options = dict(steps=4, eta=.7, execution_batch=4)
    before = torch.get_rng_state().clone()
    baseline = sample(model, labels, 17, microbatch=4, **options)
    assert torch.equal(baseline, sample(model, labels, 17, microbatch=3, **options))
    assert torch.equal(baseline[:1], sample(model, labels[:1], 17, **options))
    altered = labels.clone()
    altered[1:] = 1 - altered[1:]
    assert torch.equal(baseline[:1], sample(model, altered, 17, **options)[:1])
    assert not torch.equal(baseline, sample(model, labels, 29, **options))
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError):
        sample(model, labels, 17, steps=4, eta=1.1)


def test_invalid_sampler_inputs_fail():
    model = ProbeModel().eval()
    for bad in [torch.tensor([2]), torch.tensor([0.0]), torch.tensor([[0]])]:
        with pytest.raises(ValueError):
            sample(model, bad, 0, steps=1)
    with pytest.raises(ValueError):
        sample(model.train(), torch.tensor([0]), 0, steps=1)

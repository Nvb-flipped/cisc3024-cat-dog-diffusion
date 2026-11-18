import torch
from torch import nn

from submission.diffusion import sample
from submission.flow import flow_pair, heun_flow, reconstruct_flow


def test_linear_flow_training_pair_recovers_both_endpoints():
    images = torch.arange(24).view(2, 3, 2, 2).double() / 16
    noise = -.5 * images
    time = torch.tensor([0, 4])
    noisy, velocity = flow_pair(images, noise, time, 5)
    torch.testing.assert_close(noisy[0], images[0])
    torch.testing.assert_close(noisy[1], noise[1])
    torch.testing.assert_close(reconstruct_flow(noisy, velocity, time, 5), images)


def test_heun_integrates_a_constant_class_conditional_velocity_exactly():
    class Oracle:
        def __call__(self, image, time, labels):
            return (labels.to(image.dtype) / 4)[:, None, None, None].expand_as(image)

    initial = torch.zeros(2, 3, 2, 2)
    labels = torch.tensor([0, 1])
    result = heun_flow(Oracle(), initial, labels, [4, 2, 1, 0], 5, guidance=2.)
    torch.testing.assert_close(result[0], torch.full_like(result[0], .5))
    torch.testing.assert_close(result[1], torch.zeros_like(result[1]))


def test_flow_sampler_preserves_labels_prefix_and_caller_rng():
    class Oracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('unused', nn.Parameter(torch.zeros(())))

        def forward(self, image, time, labels):
            return (labels.to(image.dtype) / 2)[:, None, None, None].expand_as(image)

    model = Oracle().eval()
    labels = torch.tensor([0, 1, 1, 0, 1])
    options = dict(steps=4, prediction_type='flow', solver='flow_heun', execution_batch=4)
    rng = torch.get_rng_state().clone()
    baseline = sample(model, labels, 77, microbatch=4, **options)
    assert torch.equal(baseline, sample(model, labels, 77, microbatch=3, **options))
    assert torch.equal(baseline[:1], sample(model, labels[:1], 77, **options))
    altered = labels.clone()
    altered[1:] = 1 - altered[1:]
    assert torch.equal(baseline[:1], sample(model, altered, 77, **options)[:1])
    assert torch.equal(rng, torch.get_rng_state())

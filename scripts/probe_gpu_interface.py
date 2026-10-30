"""Stress actual selected architecture on CUDA, without training or retaining weights."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.common import environment, read_config, write_json
from submission.diffusion import sample
from submission.model import ConditionalUNet


def main():
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = read_config('configs/ddpm.yaml')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = ConditionalUNet(**config['model'])
    model = model.cuda().eval()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    labels = (torch.arange(20, device='cuda') % 2).long()
    begin = time.perf_counter()
    first = sample(model, labels, 3024, steps=10)
    repeated = sample(model, labels, 3024, steps=10)
    changed_labels = labels.clone()
    changed_labels[1:] = 1 - changed_labels[1:]
    changed = sample(model, changed_labels, 3024, steps=10)
    changed_seed = sample(model, labels, 3025, steps=10)
    single = sample(model, labels[:1], 3024, steps=10)
    regrouped = sample(model, labels, 3024, steps=10, microbatch=7)
    large = sample(model, (torch.arange(500, device='cuda') % 2).long(), 3024, steps=2)
    result = {'scope': 'Untrained actual selected architecture; interface smoke only, not official quality evidence',
        'parameters': sum(p.numel() for p in model.parameters()), 'environment': environment(),
        'repeat_exact': torch.equal(first, repeated), 'label_isolation_exact': torch.equal(first[0], changed[0]),
        'prefix_exact': torch.equal(first[:1], single), 'seed_sensitive': not torch.equal(first, changed_seed),
        'cpu_rng_preserved': torch.equal(cpu_rng, torch.get_rng_state()),
        'cuda_rng_preserved': torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        'microbatch_max_absolute_difference': (first - regrouped).abs().max().item(),
        'count_500_shape': list(large.shape), 'finite': torch.isfinite(large).all().item(),
        'range': [large.min().item(), large.max().item()], 'seconds': time.perf_counter() - begin,
        'steps': {'isolation': 10, 'capacity_500': 2}, 'weights_retained': False}
    print(result)
    result['microbatch_invariant'] = result['microbatch_max_absolute_difference'] <= 1e-4
    write_json('results/diagnostics/gpu_interface.json', result)
    required = ['repeat_exact', 'label_isolation_exact', 'prefix_exact', 'seed_sensitive', 'microbatch_invariant',
                'cpu_rng_preserved', 'cuda_rng_preserved', 'finite']
    if not all(result[k] for k in required):
        raise AssertionError('CUDA interface smoke failed')


if __name__ == '__main__':
    main()

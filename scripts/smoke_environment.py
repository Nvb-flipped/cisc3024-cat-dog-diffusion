"""Actually execute Blackwell convolution, attention, BF16 and backward passes."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.common import environment, write_json
from submission.model import ConditionalUNet


def main():
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    record = environment() | {'device_capability': torch.cuda.get_device_capability(),
              'architectures': torch.cuda.get_arch_list(), 'bf16_supported': torch.cuda.is_bf16_supported()}
    torch.manual_seed(3024)
    for precision in ['fp32', 'bf16']:
        model = ConditionalUNet(channels=[16, 32, 48, 64], blocks_per_level=1,
                                conditioning_dim=64).cuda()
        x = torch.randn(4, 3, 32, 32, device='cuda')
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            output = model(x, torch.tensor([0, 1, 500, 999], device='cuda'),
                           torch.tensor([0, 1, 2, 0], device='cuda'))
            loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                             for p in model.parameters()):
            raise RuntimeError(f'{precision} produced nonfinite outputs/gradients')
        record[precision] = {'loss': loss.item(), 'seconds': time.perf_counter() - start,
                             'output_dtype': str(output.dtype)}
        del model
    proposed = ConditionalUNet(channels=[64, 128, 192, 256])
    record['original_proposal_parameters'] = sum(p.numel() for p in proposed.parameters())
    main_model = ConditionalUNet()
    record['selected_channels'] = [48, 96, 144, 192]
    record['selected_model_parameters'] = sum(p.numel() for p in main_model.parameters())
    record['engineering_cap_pass'] = record['selected_model_parameters'] <= 18_000_000
    write_json('results/setup/environment_smoke.json', record)
    print(record)


if __name__ == '__main__':
    main()

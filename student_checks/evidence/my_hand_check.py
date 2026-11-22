"""Check 3, part B: compare my hand calculation with the real functions in submission/diffusion.py.

Run from the project root:
    .venvs/training/bin/python student_checks/evidence/my_hand_check.py 2>&1 | tee student_checks/evidence/my_hand_check.log
(The comparison snippet was suggested by the agent; the hand values below are from my own calculation.)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from submission.diffusion import add_noise, reconstruct_clean  # noqa: E402

HAND_NOISY = 0.2728220211291865     # my hand value of x_t = 0.9*0.4 + sqrt(0.19)*(-0.2)
HAND_CLEAN = 0.4                    # my hand value of the recovered clean value

clean = torch.tensor([[[[0.4]]]], dtype=torch.float64)
noise = torch.tensor([[[[-0.2]]]], dtype=torch.float64)
alpha_bar = torch.tensor([0.81], dtype=torch.float64)      # one cumulative alpha, at index 0
t = torch.tensor([0])

noisy = add_noise(clean, noise, t, alpha_bar).item()
a = alpha_bar[t].reshape(-1, 1, 1, 1)
recovered = reconstruct_clean(torch.tensor([[[[noisy]]]], dtype=torch.float64), noise, a).item()

print(f'add_noise           : code {noisy!r}   hand {HAND_NOISY!r}   |diff| {abs(noisy - HAND_NOISY):.2e}')
print(f'reconstruct_clean   : code {recovered!r}   hand {HAND_CLEAN!r}   |diff| {abs(recovered - HAND_CLEAN):.2e}')
print('MATCH' if abs(noisy - HAND_NOISY) < 1e-12 and abs(recovered - HAND_CLEAN) < 1e-12 else 'MISMATCH - investigate')

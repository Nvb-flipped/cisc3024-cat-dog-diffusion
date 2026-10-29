import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'starter_kit'))


@pytest.fixture(autouse=True, scope='session')
def bounded_cpu_threads():
    torch.set_num_threads(4)

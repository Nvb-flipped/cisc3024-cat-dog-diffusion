import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'starter_kit'))


def pytest_addoption(parser):
    parser.addoption('--run-training', action='store_true', help='Run the fixed 2000-update GPU integration test')


@pytest.fixture(autouse=True, scope='session')
def bounded_cpu_threads():
    torch.set_num_threads(4)

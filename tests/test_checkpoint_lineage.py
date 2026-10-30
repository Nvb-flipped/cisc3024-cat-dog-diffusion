import pytest
import torch
from src.training.train import atomic_checkpoint, validate_resume_destination


def test_checkpoint_is_not_silently_replaced(tmp_path):
    path = tmp_path / 'update_000010.pt'
    atomic_checkpoint(path, {'update': 10})
    with pytest.raises(FileExistsError, match='original training lineage'):
        atomic_checkpoint(path, {'update': 20})
    assert torch.load(path, weights_only=True)['update'] == 10


def test_resuming_old_checkpoint_requires_a_separate_branch(tmp_path):
    (tmp_path / 'update_000020.pt').write_bytes(b'fixture only')
    with pytest.raises(ValueError, match='fresh --out'):
        validate_resume_destination(tmp_path, 10, 30)
    validate_resume_destination(tmp_path, 20, 30)
    with pytest.raises(ValueError, match='must exceed'):
        validate_resume_destination(tmp_path, 20, 20)

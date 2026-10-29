import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from src.common import sha256
from src.release.export import export_checkpoint
from src.release.check import check_files
from submission.model import ConditionalUNet


@pytest.fixture
def training_fixture(tmp_path):
    model_config = dict(channels=[16, 32, 48, 64], blocks_per_level=1,
                        conditioning_dim=64, dropout=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = ConditionalUNet(**model_config)
    manifest = tmp_path / 'fixture_manifest.csv'
    manifest.write_text('pool_id,x0,y0,size,angle,label\ns000000,0,0,40,0,0\n')
    state = {'model': model.state_dict(), 'ema': model.state_dict(),
        'manifest_sha256': sha256(manifest), 'update': 0, 'seed': 17,
        'config': {'model': model_config, 'sampling': {'steps': 2, 'guidance': 1.5, 'microbatch': 2},
                   'diffusion': {'timesteps': 1000, 'cosine_offset': 0.008}}}
    checkpoint = tmp_path / 'untrained_test_fixture.pt'
    torch.save(state, checkpoint)
    return checkpoint, manifest


def test_export_refuses_manifest_mismatch(training_fixture, tmp_path):
    checkpoint, manifest = training_fixture
    manifest.write_text(manifest.read_text() + 's000001,0,0,40,0,1\n')
    with pytest.raises(ValueError, match='manifest mismatch'):
        export_checkpoint(checkpoint, manifest, tmp_path / 'export')


def test_export_is_immutable_and_detects_tampered_code(training_fixture, tmp_path):
    checkpoint, manifest = training_fixture
    out = tmp_path / 'export'
    export_checkpoint(checkpoint, manifest, out)
    assert check_files(out)['submission_bytes'] < 90_000_000
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, manifest, out)
    (out / 'model.py').write_text((out / 'model.py').read_text() + '\n# test tamper\n')
    with pytest.raises(ValueError, match='code changed'):
        check_files(out)


def test_offline_fresh_process_uses_only_export(training_fixture, tmp_path):
    checkpoint, manifest = training_fixture
    out = tmp_path / 'export'
    export_checkpoint(checkpoint, manifest, out)
    result = tmp_path / 'offline.json'
    subprocess.run([sys.executable, '-m', 'src.release.check', '--submission', str(out),
                    '--counts', '1,2', '--out', str(result)], check=True, capture_output=True, text=True)
    evidence = json.loads(result.read_text())
    assert evidence['offline'] is True and evidence['parameters'] > 0
    assert [r['count'] for r in evidence['records']] == [1, 2]

import json
import hashlib
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import zipfile
from src.common import ROOT, sha256
from src.release.export import export_checkpoint
from src.release.check import check_files
from src.release.verify_archive import verify_archive
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


@pytest.mark.parametrize('sampling', [{'guidance': float('nan')}, {'execution_batch': 0}, {'steps': 2.5}])
def test_invalid_sampling_fails_before_writing_export(training_fixture, tmp_path, sampling):
    checkpoint, manifest = training_fixture
    out = tmp_path / 'export'
    with pytest.raises(ValueError, match='Invalid sampler'):
        export_checkpoint(checkpoint, manifest, out, **sampling)
    assert not out.exists()


def test_diagnostic_checkpoint_cannot_be_exported(training_fixture, tmp_path):
    checkpoint, manifest = training_fixture
    state = torch.load(checkpoint, weights_only=True)
    state['diagnostic_only'] = True
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='Diagnostic weights'):
        export_checkpoint(checkpoint, manifest, tmp_path / 'export')


def test_nonfinite_checkpoint_fails_before_export(training_fixture, tmp_path):
    checkpoint, manifest = training_fixture
    state = torch.load(checkpoint, weights_only=True)
    next(iter(state['ema'].values())).view(-1)[0] = float('nan')
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='nonfinite weights'):
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


def test_actual_archive_extracts_and_runs_its_own_inference_code(training_fixture, tmp_path):
    checkpoint, manifest = training_fixture
    out = tmp_path / 'submission'
    export_checkpoint(checkpoint, manifest, out)
    members = {f'submission/{path.name}': path.read_bytes() for path in out.iterdir() if path.is_file()}
    for name in ['src/__init__.py', 'src/common.py', 'src/release/__init__.py', 'src/release/check.py']:
        members[name] = (ROOT / name).read_bytes()
    archive_path = tmp_path / 'synthetic_archive.zip'
    with zipfile.ZipFile(archive_path, 'w') as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
        bundle.writestr('GIT_HISTORY.txt', 'Synthetic software fixture; not a final course release.\n')
        bundle.writestr('ARTIFACT_HASHES.json', json.dumps({name: hashlib.sha256(content).hexdigest()
                                                         for name, content in members.items()}))
    evidence = verify_archive(archive_path, tmp_path / 'extracted', devices=['cpu'], counts='1,2',
                              expected_sha256=sha256(archive_path))
    assert evidence['archive_sha256'] == sha256(archive_path)
    assert list(evidence['devices']) == ['cpu']
    assert evidence['official_evaluation'] is None

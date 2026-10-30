import pytest


@pytest.mark.integration
def test_overfits_one_manifest_batch(request, tmp_path):
    if not request.config.getoption('--run-training'):
        pytest.skip('Run explicitly with --run-training; fixed 2000-update GPU diagnostic')
    from src.training.diagnostics import overfit
    record = overfit(out='results/diagnostics/overfit')
    assert record['updates'] == 2000
    assert record['reduction'] >= .80
    assert record['reconstruction_mse_after'] < record['reconstruction_mse_before']

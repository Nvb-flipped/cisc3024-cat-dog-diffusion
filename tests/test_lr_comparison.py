from src.common import read_config
from src.training.lr_comparison import configurations, materialize, run_plan
import pytest
import importlib
from pathlib import Path
from src.common import sha256, write_json


def test_declared_lr_recipes_change_only_lr_at_equal_budget():
    base = read_config('configs/ddpm.yaml')
    recipes = configurations(base)
    assert [recipe['learning_rate'] for recipe in recipes] == [2e-4, 1e-4, 3e-4]
    assert sum(recipe['is_reused_baseline'] for recipe in recipes) == 1
    for recipe in recipes:
        assert recipe['training_seed'] == 17 and recipe['updates'] == 50000
        expected = dict(base['training']) | {'updates': 50000, 'learning_rate': recipe['learning_rate']}
        assert recipe['config']['training'] == expected
        assert recipe['config']['model'] == base['model']
    assert base['training']['learning_rate'] == 2e-4


def test_missing_reviewed_data_produces_no_runnable_or_fake_checkpoint_plan(tmp_path):
    plan = materialize(tmp_path / 'missing.csv', output=tmp_path / 'plans')
    assert plan['status'] == 'pending_personally_reviewed_manifest'
    assert plan['commands'] == [] and not (tmp_path / 'plans').exists()
    with pytest.raises(ValueError, match='gated'):
        run_plan(plan)


def test_ready_plan_reuses_baseline_and_preflights_all_jobs(tmp_path, monkeypatch):
    module = importlib.import_module('src.training.lr_comparison')
    manifest = tmp_path / 'synthetic_manifest.csv'
    manifest.write_text('pool_id,label\n' + ''.join(f'fixture_{label}_{i},{label}\n' for label in [0, 1] for i in range(500)))
    write_json(manifest.with_suffix('.provenance.json'), {'origin': 'student_reviewed_manifest',
        'scope': 'SYNTHETIC SOFTWARE FIXTURE; no actual approvals', 'manifests': [{'sha256': sha256(manifest)}]})
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    plan = materialize(manifest, output=tmp_path / 'plans')
    repeated = materialize(manifest, output=tmp_path / 'plans')
    assert plan == repeated
    assert Path(plan['commands'][0]['output']).name == f'ddpm_seed17_{sha256(manifest)[:10]}'
    partial = Path(plan['commands'][2]['output'])
    partial.mkdir(parents=True)
    (partial / 'run.json').write_text('synthetic interrupted run')
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match='Interrupted/partial'):
        run_plan(plan)
    assert calls == []

import json
from pathlib import Path

import torch

from src.training import optimize
from src.training.density_sampling import appearance_sampling_weights
from src.training.step import timestep_sampling_probabilities


def test_appearance_weights_increase_rare_exposure_within_species():
    images = torch.zeros(39, 3, 32, 32)
    images[18, 0] = .85
    labels = torch.tensor([0] * 19 + [1] * 20)
    weights = appearance_sampling_weights(images, labels)
    torch.testing.assert_close(weights[:19].sum(), torch.tensor(19.))
    torch.testing.assert_close(weights[19:], torch.ones(20))
    assert weights[18] > weights[0] > 0
    assert torch.equal(weights, appearance_sampling_weights(images, labels))


def test_timestep_focus_shifts_mass_without_dropping_endpoints():
    count = 1000
    timesteps = torch.linspace(0, 1, count)
    probabilities = timestep_sampling_probabilities(count, .65, 'cpu')
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert torch.all(probabilities >= .25 / count)
    assert .58 < torch.dot(timesteps, probabilities) < .63


def test_extension_requires_a_valid_quality_gain():
    previous = {'score': (10, -1.2, -.02, -30)}
    assert not optimize.quality_improved(previous, {'score': None})
    assert not optimize.quality_improved(previous, {'score': (10, -1.18, -.01, -20)})
    assert optimize.quality_improved(previous, {'score': (10, -1.1, -.02, -30)})
    assert optimize.quality_improved(previous, {'score': (11, -1.3, -.02, -30)})


def test_pipeline_stops_extension_and_confirms_frozen_sampler(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('synthetic orchestration fixture\n')
    plan_path = tmp_path / 'lr.json'
    plan_path.write_text('{}')
    baseline_results = tmp_path / 'baseline_results'
    baseline_results.mkdir()
    (baseline_results / 'seed_0.json').write_text('{}')
    recipes = [{'learning_rate': rate, 'is_reused_baseline': rate == .0002,
                'output': str(tmp_path / f'run_{rate:g}')}
               for rate in (.0002, .0001, .0003)]
    plan = {'manifest_sha256': optimize.sha256(manifest), 'plan_path': str(plan_path), 'commands': recipes}
    monkeypatch.setattr(optimize, 'materialize', lambda path: plan)
    monkeypatch.setattr(optimize, 'run_plan', lambda plan: None)

    extensions = []
    monkeypatch.setattr(optimize, 'extend', lambda recipe, manifest, target: extensions.append(target))
    out = tmp_path / 'results'
    config = tmp_path / 'evaluation.yaml'
    config.write_text('weights: [ema, model]\nsteps: [100]\nguidance: [1.0, 1.5]\n'
                      'screening_seeds: [0]\nscreening_n_per_class: 250\n'
                      'selection_seeds: [0, 101, 202]\nconfirmation_seeds: [303, 404]\n')

    def evaluate(checkpoint, manifest, exported, output, *, seeds=(0,), n_per_class=1000,
                 weights='ema', steps=100, guidance=1.0):
        update = int(Path(checkpoint).stem.split('_')[-1])
        if seeds == [303, 404]:
            assert (out / 'winner.json').exists()
        # The lower LR wins at 50k and its 100k continuation regresses.
        tier = 13 if update == 50000 and '0.0001' in str(checkpoint) else 12
        metrics = {'accuracy': .91, 'fid': 55, 'recall': .43, 'memorization_rate': .01}
        return {'checkpoint': str(checkpoint), 'update': update, 'exported': str(exported),
                'score': (tier, -1.0, -.01, -30), 'weights': weights, 'steps': steps,
                'guidance': guidance, 'evidence': [{'metrics': metrics} for seed in seeds]}

    monkeypatch.setattr(optimize, 'evaluate', evaluate)
    optimize.optimize(manifest, tmp_path / 'baseline', baseline_results, out, tmp_path / 'exports', config)
    report = json.loads((out / 'status.json').read_text())
    assert extensions == [100000]
    assert report['chosen_training']['update'] == 50000
    assert report['chosen_training']['learning_rate'] == .0001
    assert len(report['full_comparisons']) == 3
    assert report['full_comparisons'][0]['guidance'] == 1.0
    assert report['stage'] == 'complete'

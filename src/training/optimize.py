"""Run the planned LR, training-length and sampler comparisons on the reviewed pool."""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

from src.common import ROOT, read_config, sha256, timestamp, write_json
from src.evaluation.run import run_official
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.check import check_files
from src.release.export import export_checkpoint
from src.training.lr_comparison import materialize, run_plan

MINIMUM_MARGIN_GAIN = .05


def evaluate(checkpoint, manifest, exported, output, *, seeds=(0,), n_per_class=1000,
             weights='ema', steps=100, guidance=1.0):
    checkpoint, exported, output = map(Path, (checkpoint, exported, output))
    if not exported.exists():
        export_checkpoint(checkpoint, manifest, exported, weights=weights, steps=steps, guidance=guidance)
    files = check_files(exported)['file_sha256']
    lineage = json.loads((exported / 'provenance.json').read_text())
    sampling = json.loads((exported / 'inference.json').read_text())['sampling']
    if (lineage['checkpoint_sha256'], lineage['manifest_sha256'], lineage['weights_variant'],
            sampling['steps'], sampling['guidance'], sampling['execution_batch']) != (
            sha256(checkpoint), sha256(manifest), weights, steps, guidance, 32):
        raise ValueError(f'Existing export does not match the declared experiment: {exported}')

    results, evidence = [], []
    for seed in seeds:
        path = output / f'seed_{seed}.json'
        if not path.exists():
            print(f'Evaluating {exported.name}: {n_per_class} per class, seed {seed}', flush=True)
            run_official(exported, path, seed=seed, n_per_class=n_per_class)
        result = validated_result(path, seed, exported, n_per_class=n_per_class)
        results.append(result)
        evidence.append({'seed': seed, 'path': str(path), 'sha256': sha256(path),
                         'metrics': result['metrics'], 'marks': result['marks'],
                         'violations': result['violations']})
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': lineage['checkpoint_sha256'],
            'update': lineage['training_update'], 'exported': str(exported), 'export_hashes': files,
            'weights': weights, 'steps': steps, 'guidance': guidance, 'evidence': evidence,
            'score': score_results(results, n_per_class=n_per_class)}


def meets_target(candidate):
    if candidate['score'] is None:
        return False
    return all(row['metrics']['accuracy'] >= .93 and row['metrics']['fid'] <= 45 and
               row['metrics']['recall'] >= .45 and row['metrics']['memorization_rate'] <= .02
               for row in candidate['evidence'])


def quality_improved(earlier, later):
    """Allow more updates for quality gains, without using speed as evidence of a gain."""
    old, new = earlier['score'], later['score']
    return new is not None and (old is None or new[0] > old[0] or
                                (new[0] == old[0] and new[1] - old[1] >= MINIMUM_MARGIN_GAIN))


def extend(recipe, manifest, target):
    """Resume at the latest saved update of this same training recipe."""
    run = Path(recipe['output'])
    target_path = run / f'update_{target:06d}.pt'
    if not target_path.exists():
        previous = max(path for path in run.glob('update_*.pt') if path < target_path)
        command = recipe['command'] + ['--resume', str(previous), '--updates', str(target)]
        print(f'Extending {recipe["name"]} from {previous.stem} to {target:,}', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    receipt = json.loads(target_path.with_suffix('.json').read_text())
    if receipt['sha256'] != sha256(target_path) or receipt['manifest_sha256'] != sha256(manifest) or \
            receipt['update'] != target:
        raise ValueError(f'Extended checkpoint does not match its receipt: {target_path}')
    return target_path


def optimize(manifest, baseline_export, baseline_results, output, export_root, evaluation_config):
    manifest, baseline_export, baseline_results, output, export_root = map(
        Path, (manifest, baseline_export, baseline_results, output, export_root))
    config = read_config(evaluation_config)
    plan = materialize(manifest)
    status_path = output / 'status.json'
    report = {'manifest_sha256': plan['manifest_sha256'], 'plan': plan['plan_path'],
              'plan_sha256': sha256(plan['plan_path']), 'evaluation_config_sha256': sha256(evaluation_config),
              'minimum_margin_gain_for_extension': MINIMUM_MARGIN_GAIN, 'started': timestamp()}

    def status(stage, **details):
        report.update(stage=stage, updated=timestamp(), **details)
        temporary = status_path.with_suffix('.tmp')
        write_json(temporary, report)
        for attempt in range(6):
            try:
                temporary.replace(status_path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.1)

    def training_evaluation(recipe, update):
        path = Path(recipe['output']) / f'update_{update:06d}.pt'
        if recipe['is_reused_baseline'] and update == 50000:
            exported, results = baseline_export, baseline_results
        else:
            name = f'lr_{recipe["learning_rate"]:g}_u{update:06d}'
            exported, results = export_root / name, output / 'training' / name
        return evaluate(path, manifest, exported, results) | {'learning_rate': recipe['learning_rate']}

    try:
        status('baseline_check')
        baseline = plan['commands'][0]
        if not baseline['is_reused_baseline'] or not (baseline_results / 'seed_0.json').is_file():
            raise ValueError('The completed baseline evaluation must exist before comparisons start')
        training_evaluation(baseline, 50000)
        status('learning_rate_training')
        run_plan(plan)

        comparisons = []
        for update in (50000, 40000, 30000):
            status('learning_rate_evaluation', comparison_update=update, training_comparisons=comparisons)
            for recipe in plan['commands']:
                comparisons.append(training_evaluation(recipe, update))
                status('learning_rate_evaluation', training_comparisons=comparisons)
            valid = [row for row in comparisons if row['score'] is not None]
            if valid:
                break
        if not valid:
            status('no_valid_training_candidate')
            return

        candidate = max(valid, key=lambda row: row['score'])
        recipe = next(row for row in plan['commands'] if row['learning_rate'] == candidate['learning_rate'])
        decisions = []
        if candidate['update'] == 50000 and not meets_target(candidate):
            status('training_length_evaluation')
            earlier = training_evaluation(recipe, 40000)
            comparisons.append(earlier)
            if quality_improved(earlier, candidate):
                for target in (100000, 200000):
                    decisions.append({'extend_to': target, 'reason': 'Higher official tier total or balanced quality margin'})
                    status('extending_training', training_comparisons=comparisons, length_decisions=decisions)
                    extend(recipe, manifest, target)
                    longer = training_evaluation(recipe, target)
                    comparisons.append(longer)
                    if not quality_improved(candidate, longer):
                        decisions.append({'stop_at': target, 'reason': 'Official quality did not improve'})
                        break
                    candidate = longer
                    if meets_target(candidate):
                        decisions.append({'stop_at': target, 'reason': 'Seed-0 development quality targets met'})
                        break
                else:
                    decisions.append({'stop_at': 200000, 'reason': 'Training budget ceiling reached'})
            else:
                if earlier['score'] is not None and earlier['score'] > candidate['score']:
                    candidate = earlier
                decisions.append({'stop_at': 50000, 'reason': 'No official quality gain from 40k to 50k'})
        else:
            decisions.append({'stop_at': candidate['update'], 'reason': 'Earlier valid checkpoint or seed-0 quality targets met'})

        screen = []
        status('sampler_screening', training_comparisons=comparisons, length_decisions=decisions,
               chosen_training=candidate)
        for weights, steps, guidance in itertools.product(config['weights'], config['steps'], config['guidance']):
            name = f'{weights}_s{steps}_g{guidance:g}'
            row = evaluate(candidate['checkpoint'], manifest, export_root / 'sampling' / name,
                           output / 'screening' / name, seeds=config['screening_seeds'],
                           n_per_class=config['screening_n_per_class'], weights=weights, steps=steps, guidance=guidance)
            screen.append(row)
            status('sampler_screening', sampler_comparisons=screen)
        incumbent = next(row for row in screen if (row['weights'], row['steps'], row['guidance']) == ('ema', 100, 1.0))
        alternatives = sorted((row for row in screen if row is not incumbent and row['score'] is not None),
                              key=lambda row: row['score'], reverse=True)

        full = []
        for row in [incumbent, *alternatives[:2]]:
            status('sampler_selection', full_comparisons=full)
            full.append(evaluate(row['checkpoint'], manifest, row['exported'],
                        output / 'selection' / Path(row['exported']).name, seeds=config['selection_seeds'],
                        weights=row['weights'], steps=row['steps'], guidance=row['guidance']))
        eligible = [row for row in full if row['score'] is not None]
        if not eligible:
            status('no_valid_sampler_at_full_protocol', full_comparisons=full)
            return

        winner = max(eligible, key=lambda row: row['score'])
        frozen = {'winner': winner, 'selection_seeds': config['selection_seeds'],
                  'confirmation_seeds': config['confirmation_seeds'], 'manifest_sha256': sha256(manifest)}
        winner_path = output / 'winner.json'
        if winner_path.exists():
            if json.loads(winner_path.read_text()) != json.loads(json.dumps(frozen)):
                raise ValueError('Winner differs from the candidate frozen before confirmation')
        else:
            write_json(winner_path, frozen)
        status('confirmation', full_comparisons=full, winner=winner, winner_sha256=sha256(winner_path))
        confirmation = evaluate(winner['checkpoint'], manifest, winner['exported'], output / 'confirmation',
                                seeds=config['confirmation_seeds'], weights=winner['weights'],
                                steps=winner['steps'], guidance=winner['guidance'])
        status('complete' if confirmation['score'] is not None else 'confirmation_failed',
               confirmation=confirmation, all_five_meet_development_targets=meets_target(winner) and meets_target(confirmation))
    except Exception as error:
        status('failed', error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-export', type=Path, required=True)
    parser.add_argument('--baseline-results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--export-root', type=Path, required=True)
    parser.add_argument('--evaluation-config', type=Path, default='configs/evaluation.yaml')
    args = parser.parse_args()
    optimize(args.manifest, args.baseline_export, args.baseline_results, args.out, args.export_root,
             args.evaluation_config)


if __name__ == '__main__':
    main()

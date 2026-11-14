"""Refresh an explicitly incomplete report outline from recorded implementation evidence."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results


def development_claims(paths):
    manifest = ROOT / 'manifests/v001.csv'
    holdout = manifest.with_suffix('.validation.csv')
    provenance = manifest.with_suffix('.provenance.json')
    baseline = ROOT / 'results/experiments/baseline_v001_seed17/seed_0.json'
    refinement = ROOT / 'results/experiments/refinement_v001_seed17'
    selection_path = refinement / 'selection.json'
    confirmation_path = refinement / 'confirmation.json'
    if not confirmation_path.exists():
        return []

    manifest_info = json.loads(provenance.read_text())
    manifest_hash, holdout_hash = sha256(manifest), sha256(holdout)
    if manifest_info['origin'] != 'student_reviewed_manifest' or \
            [manifest_hash, holdout_hash] != [row['sha256'] for row in manifest_info['manifests']]:
        raise ValueError('Reviewed training or holdout manifest no longer matches provenance')

    baseline_result = validated_result(baseline, 0, ROOT / 'private_build/baseline_v001_seed17')
    selection = json.loads(selection_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    if selection['manifest_sha256'] != manifest_hash or \
            confirmation['selection_sha256'] != sha256(selection_path):
        raise ValueError('Development confirmation does not bind this selection and manifest')
    chosen = selection['selected']
    export = ROOT / chosen['exported']
    evidence = chosen['trials'] + confirmation['confirmation']['evidence']
    seeds = [0, 101, 202, 303, 404]
    if [row['seed'] for row in evidence] != seeds:
        raise ValueError('Development evidence does not cover the five declared seeds')
    files = [manifest, holdout, provenance, baseline, baseline.with_suffix('.provenance.json'),
             selection_path, confirmation_path]
    results = []
    for seed, row in zip(seeds, evidence):
        path = ROOT / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError(f'Development evaluation changed: {path}')
        results.append(validated_result(path, seed, export))
        files.append(path)
    paths.update({f'E{index:03d}': str(path.relative_to(ROOT).as_posix())
                  for index, path in enumerate(files, 9)})

    def metric_range(series, metric, fmt):
        values = [result['metrics'][metric] for result in series]
        return f'{min(values):{fmt}}–{max(values):{fmt}}'

    train, validation = (manifest_info[name] for name in ('train_by_class', 'validation_by_class'))
    metrics = baseline_result['metrics']
    valid_count = sum(result['marks']['valid'] for result in results)
    claims = [
        ('C007', 'Data', 'numeric', f"The personally reviewed v001 split contains {train['0']} cat and {train['1']} dog training crops and a disjoint holdout of {validation['0']} cat and {validation['1']} dog crops.", ['E009', 'E010', 'E011']),
        ('C008', 'Development results', 'result', f"The full-width seed-17 baseline at 50,000 updates reached {metrics['accuracy']:.2%} accuracy and {metrics['fid']:.2f} FID on evaluator seed 0. Its {metrics['memorization_rate']:.2%} memorization exceeded the 5% limit.", ['E012', 'E013']),
        ('C009', 'Development results', 'result', f"The 21,000-update EMA development candidate using 50 DDIM steps and CFG {chosen['guidance']:g} passed the hard conditions on {valid_count} of {len(seeds)} seeds. Accuracy was {metric_range(results, 'accuracy', '.2%')}, FID {metric_range(results, 'fid', '.2f')}, recall {metric_range(results, 'recall', '.2%')}, and memorization {metric_range(results, 'memorization_rate', '.2%')}.", [f'E{index:03d}' for index in range(14, 21)]),
    ]
    capacity = ROOT / 'results/experiments/capacity_32_seed17'
    capacity_confirmation_path = capacity / 'confirmation.json'
    if capacity_confirmation_path.exists():
        capacity_selection_path = capacity / 'selection.json'
        capacity_selection = json.loads(capacity_selection_path.read_text())
        capacity_confirmation = json.loads(capacity_confirmation_path.read_text())
        if capacity_selection['manifest_sha256'] != manifest_hash or \
                capacity_confirmation['selection_sha256'] != sha256(capacity_selection_path):
            raise ValueError('Capacity confirmation does not bind this selection and manifest')
        small = capacity_selection['selected']
        small_evidence = small['evidence'] + capacity_confirmation['confirmation']['evidence']
        if [row['seed'] for row in small_evidence] != seeds:
            raise ValueError('Capacity evidence does not cover the five declared seeds')
        small_files = [capacity_selection_path, capacity_confirmation_path]
        small_results = []
        for seed, row in zip(seeds, small_evidence):
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError(f'Capacity evaluation changed: {path}')
            small_results.append(validated_result(path, seed, ROOT / small['exported']))
            small_files.append(path)
        paths.update({f'E{index:03d}': path.relative_to(ROOT).as_posix()
                      for index, path in enumerate(small_files, 21)})
        small_valid = sum(result['marks']['valid'] for result in small_results)
        claims.append(('C010', 'Development results', 'result',
            f"The {small['update']:,}-update smaller EMA candidate passed on {small_valid} of {len(seeds)} seeds. Accuracy was {metric_range(small_results, 'accuracy', '.2%')}, FID {metric_range(small_results, 'fid', '.2f')}, recall {metric_range(small_results, 'recall', '.2%')}, and memorization {metric_range(small_results, 'memorization_rate', '.2%')}.",
            [f'E{index:03d}' for index in range(21, 28)]))
    quantity = ROOT / 'results/experiments/quantity_v001_seed17'
    comparison_path = quantity / 'comparison.json'
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        candidate = comparison['candidate']
        result_path = quantity / 'seed_0.json'
        if comparison['baseline']['result_sha256'] != sha256(baseline) or \
                candidate['evidence'][0]['sha256'] != sha256(result_path):
            raise ValueError('Matched quantity evidence changed after evaluation')
        measured = validated_result(result_path, 0, ROOT / candidate['exported'])['metrics']
        files = [comparison_path, result_path, result_path.with_suffix('.provenance.json')]
        paths.update({f'E{index:03d}': path.relative_to(ROOT).as_posix()
                      for index, path in enumerate(files, 28)})
        claims.append(('C011', 'Matched data quantity comparison', 'result',
            f"At 50,000 updates, the balanced 1,000-crop subset reached {measured['accuracy']:.2%} accuracy, FID {measured['fid']:.2f}, and {measured['recall']:.2%} recall on seed 0. Memorization was {measured['memorization_rate']:.2%} for the subset and {metrics['memorization_rate']:.2%} for the 1,197-crop parent. Both exceeded the 5% limit.",
            ['E012', 'E028', 'E029', 'E030']))
    frontier = ROOT / 'results/experiments/capacity_frontier_v001'
    frontier_selection = frontier / 'selection.json'
    if frontier_selection.exists():
        comparison = json.loads(frontier_selection.read_text())
        longer = next(row for row in comparison['checks'] if row['name'] == 'u032000_ema_s50_g2')
        files = [frontier_selection]
        outcomes = []
        for row in longer['evidence']:
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError(f'Frontier evaluation changed: {path}')
            files.append(path)
            outcomes.append(validated_result(path, row['seed'], ROOT / longer['exported']))
        initial, last = outcomes[0], outcomes[-1]
        ids = {f'E{index:03d}': path.relative_to(ROOT).as_posix()
               for index, path in enumerate(files, 31)}
        paths.update(ids)
        rate = last['metrics']['memorization_rate']
        flags = round(rate * last['metrics']['num_samples'])
        claims.append(('C012', 'Training-length and sampler comparison', 'result',
            f"The 32,000-update small candidate with 50 DDIM steps and CFG 2.0 reached {initial['metrics']['recall']:.2%} recall on seed 0. Seed {last['protocol']['seed']} had {flags} memorization flags among {last['metrics']['num_samples']:,} generated crops. The official evaluator stored the rate as {rate:.11f} and rejected the candidate.", list(ids)))
    dropout = ROOT / 'results/experiments/dropout25_wide_seed17'
    dropout_plan = ROOT / 'results/plans/dropout25_wide_seed17_86c3c843eb.json'
    dropout_selection = dropout / 'selection.json'
    if dropout_selection.exists():
        plan = json.loads(dropout_plan.read_text())
        decision = json.loads(dropout_selection.read_text())
        matched = ROOT / plan['matched_22k_reference']
        if (decision['plan_sha256'], decision['manifest_sha256'], sha256(matched)) != (
                sha256(dropout_plan), manifest_hash, plan['matched_22k_reference_sha256']):
            raise ValueError('Dropout comparison no longer matches its plan or reference')
        variant = 'u022000_ema_s50_g2.5'
        result_path = dropout / 'full' / variant / 'seed_0.json'
        reference = validated_result(matched, 0, ROOT / 'private_build/refinement_v001_seed17' / variant)
        measured = validated_result(result_path, 0, ROOT / 'private_build/dropout25_wide_seed17' / variant)
        files = [dropout_plan, dropout_selection, matched, result_path, result_path.with_suffix('.provenance.json')]
        ids = {f'E{index:03d}': path.relative_to(ROOT).as_posix()
               for index, path in enumerate(files, 34)}
        paths.update(ids)
        low, high = reference['metrics'], measured['metrics']
        claims.append(('C013', 'Matched residual dropout comparison', 'result',
            f"At 22,000 updates on seed 0, increasing dropout from 0.10 to 0.25 changed memorization from {low['memorization_rate']:.2%} to {high['memorization_rate']:.2%}, FID from {low['fid']:.2f} to {high['fid']:.2f}, and recall from {low['recall']:.2%} to {high['recall']:.2%}. The larger dropout reduced memorization and worsened sample quality at the matched budget.", list(ids)))
    sampler_source = ROOT / 'figures/F5_sampling_tradeoffs/source_summary.json'
    if sampler_source.exists():
        summary = json.loads(sampler_source.read_text())
        rows = summary['rows']
        if summary['manifest_sha256'] != manifest_hash or len(rows) != 24 or any(
                sha256(ROOT / row['source']) != row['source_sha256'] or row['num_samples'] != 500 for row in rows):
            raise ValueError('Sampler figure sources changed after verification')
        paths['E039'] = sampler_source.relative_to(ROOT).as_posix()
        mem = {weights: [r['memorization_rate'] for r in rows if r['weights'] == weights]
               for weights in ('model', 'ema')}
        claims.append(('C014', 'Raw and EMA sampler screening', 'result',
            f"On the fixed {summary['update']:,}-update checkpoint, all {len(mem['model'])} raw-weight sampler screens exceeded the 5% memorization limit. Their rates were {min(mem['model']):.1%} to {max(mem['model']):.1%}, compared with {min(mem['ema']):.1%} to {max(mem['ema']):.1%} for EMA. Each setting generated 500 images on seed 0.", ['E039']))
    geometry = ROOT / 'results/curation/v001_review_geometry.json'
    if geometry.exists():
        receipt = json.loads(geometry.read_text())
        table = geometry.with_suffix('.csv')
        source_hashes = {name.replace('\\', '/'): digest for name, digest in receipt['source_hashes'].items()}
        if source_hashes[manifest.relative_to(ROOT).as_posix()] != manifest_hash or sha256(table) != receipt['table_sha256']:
            raise ValueError('Frozen reviewed geometry receipt no longer matches the manifest or table')
        paths.update(E040=geometry.relative_to(ROOT).as_posix(), E041=table.relative_to(ROOT).as_posix())
        medians = [receipt['training'][name]['head_width_px']['median'] for name in ('cat', 'dog')]
        count = len(receipt['strict_range_rechecks'])
        claims.append(('C015', 'Recorded review geometry', 'result',
            f"The frozen development split recorded median source head widths of {medians[0]:.1f} pixels for cats and {medians[1]:.1f} pixels for dogs. Recomputed width divided by side flagged {count} of {receipt['kept']:,} kept crops for a personal boundary recheck.", ['E040', 'E041']))
    repeatability = ROOT / 'results/experiments/capacity_repeatability_v001/selection.json'
    if repeatability.exists():
        repeat_plan = ROOT / 'results/plans/capacity_repeatability_86c3c843eb.json'
        decision = json.loads(repeatability.read_text())
        if sha256(repeat_plan) != decision['plan_sha256']:
            raise ValueError('Training-seed comparison plan changed')
        files = [repeat_plan, repeatability]
        summaries = []
        for run in [decision['reference'], *decision['alternatives']]:
            outcomes = []
            for row in run['evidence']:
                path = ROOT / row['path']
                if sha256(path) != row['sha256']:
                    raise ValueError(f'Training-seed comparison result changed: {path}')
                files.append(path)
                outcomes.append(validated_result(path, row['seed'], ROOT / run['exported']))
            summaries.append(f"Training seed {run['training_seed']} produced FID {metric_range(outcomes, 'fid', '.2f')}, recall {metric_range(outcomes, 'recall', '.2%')}, and memorization {metric_range(outcomes, 'memorization_rate', '.2%')}.")
        ids = {f'E{index:03d}': path.relative_to(ROOT).as_posix()
               for index, path in enumerate(files, 42)}
        paths.update(ids)
        claims.append(('C016', 'Fresh training-seed comparison', 'result',
            ' '.join(summaries) + ' Each run used the same 30,000-update recipe and three 2,000-image development evaluations.', list(ids)))
    corrected = ROOT / 'results/experiments/capacity_32_seed17_v002/comparison.json'
    if corrected.exists():
        corrected_manifest = ROOT / 'manifests/v002.csv'
        plan_path = ROOT / 'results/plans/capacity_32_seed17_63bc3fce35.json'
        plan = json.loads(plan_path.read_text())
        comparison = json.loads(corrected.read_text())
        export = ROOT / comparison['exported']
        if sha256(plan_path) != comparison['plan_sha256'] or \
                sha256(corrected_manifest) != sha256(export / 'manifest.csv') or \
                [row['seed'] for row in comparison['evidence']] != plan['generation_seeds']:
            raise ValueError('Corrected comparison manifest or protocol changed')
        files, outcomes = [plan_path, corrected_manifest, corrected], []
        for row in comparison['evidence']:
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError(f'Corrected comparison result changed: {path}')
            files.append(path)
            outcomes.append(validated_result(path, row['seed'], export))
        ids = {f'E{index:03d}': path.relative_to(ROOT).as_posix()
               for index, path in enumerate(files, 53)}
        paths.update(ids)
        valid = sum(outcome['marks']['valid'] for outcome in outcomes)
        marks = ', '.join(str(outcome['marks']['total']) for outcome in outcomes)
        claims.append(('C017', 'Corrected-manifest reference', 'result',
            f"The fresh 30,000-update v002 reference passed all hard conditions on {valid} of {len(outcomes)} "
            f"development seeds, with scores {marks} of 15. Accuracy was {metric_range(outcomes, 'accuracy', '.2%')}, "
            f"FID {metric_range(outcomes, 'fid', '.2f')}, recall {metric_range(outcomes, 'recall', '.2%')}, "
            f"and memorization {metric_range(outcomes, 'memorization_rate', '.2%')}.", list(ids)))
    late_comparisons = [
        ('C018', 'Matched velocity target', 'velocity_32_seed17_v002',
         'velocity_32_seed17_63bc3fce35.json', 59),
        ('C019', 'Matched sampler endpoint', 'v002_sampler_comparison',
         'v002_sampler_comparison.json', 64),
    ]
    for claim, heading, directory, plan_name, first in late_comparisons:
        path = ROOT / 'results/experiments' / directory / 'comparison.json'
        if not path.exists():
            continue
        plan = ROOT / 'results/plans' / plan_name
        comparison = json.loads(path.read_text())
        if comparison['plan_sha256'] != sha256(plan):
            raise ValueError(f'Late comparison plan changed: {path}')
        candidate = comparison if 'evidence' in comparison else comparison.get('candidate')
        if candidate is None:
            continue
        export = ROOT / candidate['exported']
        if [row['seed'] for row in candidate['evidence']] != [0, 101, 202] or \
                sha256(export / 'manifest.csv') != sha256(ROOT / 'manifests/v002.csv'):
            raise ValueError(f'Late comparison has different seeds or training data: {path}')
        files, outcomes = [plan, path], []
        for row in candidate['evidence']:
            result_path = ROOT / row['path']
            if sha256(result_path) != row['sha256']:
                raise ValueError(f'Late comparison result changed: {result_path}')
            files.append(result_path)
            outcomes.append(validated_result(result_path, row['seed'], export))
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate(files, first)}
        paths.update(ids)
        marks = ', '.join(str(result['marks']['total']) for result in outcomes)
        claims.append((claim, heading, 'result',
            f"The matched v002 comparison scored {marks} out of 15 across the three official development seeds, "
            f"with FID {metric_range(outcomes, 'fid', '.2f')}, recall {metric_range(outcomes, 'recall', '.2%')}, "
            f"and memorization {metric_range(outcomes, 'memorization_rate', '.2%')}.", list(ids)))
    minsnr = ROOT / 'results/experiments/minsnr5_32_seed17_v002/screening.json'
    if minsnr.exists():
        plan = ROOT / 'results/plans/minsnr5_32_seed17_63bc3fce35.json'
        screen = json.loads(minsnr.read_text())
        if screen['plan_sha256'] != sha256(plan) or \
                [row['update'] for row in screen['screened']] != json.loads(plan.read_text())['updates']:
            raise ValueError('Min-SNR checkpoint screen differs from the declared plan')
        files, summaries = [plan, minsnr], []
        for trial in screen['screened']:
            export = ROOT / trial['exported']
            row, = trial['evidence']
            path = ROOT / row['path']
            if sha256(export / 'manifest.csv') != sha256(ROOT / 'manifests/v002.csv') or sha256(path) != row['sha256']:
                raise ValueError(f'Min-SNR checkpoint evidence changed: {path}')
            result = validated_result(path, 0, export)
            files.append(path)
            metrics = result['metrics']
            summaries.append(f"At {trial['update']:,} updates, FID was {metrics['fid']:.2f}, memorization was "
                             f"{metrics['memorization_rate']:.2%}, and the score was {result['marks']['total']}/15.")
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate(files, 69)}
        paths.update(ids)
        claims.append(('C020', 'Matched Min-SNR weighting', 'result', ' '.join(summaries), list(ids)))
    ddim = ROOT / 'results/experiments/v002_ddim_steps/comparison.json'
    if ddim.exists():
        plan = ROOT / 'results/plans/v002_ddim_steps.json'
        comparison = json.loads(ddim.read_text())
        trial = comparison['seed_zero']
        row, = trial['evidence']
        path = ROOT / row['path']
        export = ROOT / trial['exported']
        if comparison['plan_sha256'] != sha256(plan) or sha256(path) != row['sha256'] or \
                sha256(export / 'manifest.csv') != sha256(ROOT / 'manifests/v002.csv'):
            raise ValueError('DDIM step comparison has changed')
        result = validated_result(path, 0, export)
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate((plan, ddim, path), 75)}
        paths.update(ids)
        metrics = result['metrics']
        claims.append(('C021', 'Matched DDIM step count', 'result',
            f"At 100 steps, seed 0 scored {result['marks']['total']}/15 with FID {metrics['fid']:.2f}, "
            f"recall {metrics['recall']:.2%}, and memorization {metrics['memorization_rate']:.2%}.", list(ids)))
    eta = ROOT / 'results/experiments/v002_ddim_eta/comparison.json'
    if eta.exists():
        comparison = json.loads(eta.read_text())
        files = [eta]
        summaries = []
        for trial in comparison['trials']:
            row, = trial['evidence']
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('Stochastic DDIM evaluation changed')
            result = validated_result(path, row['seed'], ROOT / trial['exported'])
            metrics = result['metrics']
            files.append(path)
            summaries.append(f"At eta {trial['eta']:.2f}, accuracy was {metrics['accuracy']:.2%}, "
                             f"FID {metrics['fid']:.2f}, recall {metrics['recall']:.2%}, and "
                             f"memorization {metrics['memorization_rate']:.2%}.")
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate(files, 78)}
        paths.update(ids)
        claims.append(('C022', 'Stochastic DDIM comparison', 'result', ' '.join(summaries), list(ids)))
    expansion = ROOT / 'results/curation/v003_cache_parity.json'
    if expansion.exists():
        record = json.loads(expansion.read_text())
        train, holdout = record['checks']
        manifest = ROOT / 'manifests/v003.csv'
        validation = manifest.with_suffix('.validation.csv')
        if (train['manifest_sha256'], holdout['manifest_sha256']) != (sha256(manifest), sha256(validation)) or \
                sha256(validation) != sha256(ROOT / 'manifests/v002.validation.csv') or \
                not all(row['labels_equal'] and row['pixels_equal'] for row in record['checks']):
            raise ValueError('Expansion manifest or official reconstruction changed')
        paths.update(E081=manifest.relative_to(ROOT).as_posix(),
                     E082=validation.relative_to(ROOT).as_posix(), E083=expansion.relative_to(ROOT).as_posix())
        claims.append(('C023', 'Reviewed data expansion', 'result',
            f"The v003 freeze contains {train['rows']:,} training crops ({train['cat']} cats and "
            f"{train['dog']} dogs) and preserves the same {holdout['rows']} holdout crops. "
            "All cached labels and crop pixels matched the official reconstruction.", ['E081', 'E082', 'E083']))
    edm = ROOT / 'results/diagnostics/edm_v002_shared/result.json'
    if edm.exists():
        record = json.loads(edm.read_text())
        if (record['manifest_sha256'], record['updates'], record['passed']) != (
                sha256(ROOT / 'manifests/v002.csv'), 2000, True):
            raise ValueError('EDM diagnostic does not match the declared manifest and budget')
        paths['E084'] = edm.relative_to(ROOT).as_posix()
        claims.append(('C024', 'EDM integration diagnostic', 'result',
            f"The two-crop EDM diagnostic reduced fixed-probe MSE from {record['initial']['probe_mse']:.5f} "
            f"to {record['final']['probe_mse']:.5f} in {record['updates']:,} updates "
            f"({record['reduction']:.2%} reduction).", ['E084', 'E110']))
    edm_screen = ROOT / 'results/experiments/edm_32_seed17_v002/screening.json'
    if edm_screen.exists():
        record = json.loads(edm_screen.read_text())
        plan = ROOT / 'results/plans/edm_32_seed17_63bc3fce35.json'
        if record['plan_sha256'] != sha256(plan):
            raise ValueError('EDM comparison plan changed')
        files = [plan, edm_screen]
        summaries = []
        for trial in record['screened']:
            row, = trial['evidence']
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('EDM evaluation changed')
            result = validated_result(path, row['seed'], ROOT / trial['exported'])
            metrics = result['metrics']
            files.append(path)
            summaries.append(f"At {trial['update']:,} updates, accuracy was {metrics['accuracy']:.2%}, "
                             f"FID {metrics['fid']:.2f}, recall {metrics['recall']:.2%}, "
                             f"memorization {metrics['memorization_rate']:.2%}, and the score "
                             f"{result['marks']['total']}/15.")
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate(files, 85)}
        paths.update(ids)
        claims.append(('C025', 'Matched EDM comparison', 'result', ' '.join(summaries), list(ids)))
    patch_screen = ROOT / 'results/experiments/patch_grid_v002/screening.json'
    patch_decision = patch_screen.with_name('comparison.json')
    if patch_decision.exists():
        plan = ROOT / 'results/plans/patch_grid_v002.json'
        plan_hash = sha256(plan)
        record = json.loads(patch_screen.read_text())
        decision = json.loads(patch_decision.read_text())
        trials = record['screened']
        if (record['plan_sha256'], decision['plan_sha256']) != (plan_hash, plan_hash) or \
                [(row['patch_size'], row['update']) for row in trials] != [
                    (4, 24000), (4, 30000), (8, 24000), (8, 30000)]:
            raise ValueError('Patch comparison is incomplete or differs from its plan')
        files = [plan, patch_screen, patch_decision]
        summaries = []
        for trial in trials:
            row, = trial['evidence']
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('Patch evaluation changed')
            result = validated_result(path, 0, ROOT / trial['exported'])
            metrics = result['metrics']
            size = trial['patch_size']
            files.append(path)
            summaries.append(f"At {trial['update']:,} updates with {size}×{size} patches, accuracy was "
                             f"{metrics['accuracy']:.2%}, FID {metrics['fid']:.2f}, recall "
                             f"{metrics['recall']:.2%}, memorization {metrics['memorization_rate']:.2%}, "
                             f"and the score {result['marks']['total']}/15.")
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate(files, 90)}
        paths.update(ids)
        claims.append(('C026', 'Matched patch-grid comparison', 'result', ' '.join(summaries), list(ids)))
    coverage_screen = ROOT / 'results/experiments/v004_coverage/screening.json'
    solver_comparison = ROOT / 'results/experiments/v002_dpmpp/comparison.json'
    comparison_specs = (
        (coverage_screen, 'v004_coverage', 'C027', 'Matched pool coverage comparison', 120),
        (solver_comparison, 'v002_dpmpp', 'C028', 'Matched DPM-Solver++ comparison', 125),
        (ROOT / 'results/experiments/v006_coverage/screening.json',
         'v006_coverage', 'C029', 'Larger matched pool coverage comparison', 129),
        (ROOT / 'results/experiments/v006_coverage/continuation_032000/comparison.json',
         'v006_continuation', 'C030', 'Bounded v006 continuation', 134),
        (ROOT / 'results/experiments/v006_self_condition/screening.json',
         'v006_self_condition', 'C032', 'Matched self-conditioning comparison', 143),
        (ROOT / 'results/experiments/high_noise_probe/comparison.json',
         'high_noise_probe', 'C033', 'High-noise dynamic thresholding', 148),
        (ROOT / 'results/experiments/ema_trajectory_probe/comparison.json',
         'ema_trajectory_probe', 'C034', 'Same-trajectory EMA snapshot average', 153),
        (ROOT / 'results/experiments/v006_dropout15/screening.json',
         'v006_dropout15', 'C035', 'Matched residual dropout comparison', 158),
        (ROOT / 'results/experiments/rescale_interval/comparison.json',
         'rescale_interval', 'C036', 'Restricted CFG with rescale 0.25', 162),
        (ROOT / 'results/experiments/v002_coarse_epsilon/screening.json',
         'v002_coarse_epsilon', 'C037', 'Matched 4×4 block-average epsilon loss', 166),
        (ROOT / 'results/experiments/v002_reflected_denoiser/comparison.json',
         'v002_reflected_denoiser', 'C038', 'Horizontal reflection denoiser', 170),
        (ROOT / 'results/experiments/v002_temporal_score/comparison.json',
         'v002_temporal_score', 'C039', 'Temporal score rescaling', 173),
        (ROOT / 'results/experiments/v002_pag/comparison.json',
         'v002_pag', 'C040', 'Perturbed-attention guidance', 177),
        (ROOT / 'results/experiments/v007_coverage/screening.json',
         'v007_coverage', 'C041', 'Matched Batch 19 expansion', 180),
        (ROOT / 'results/experiments/v007_pag_half/comparison.json',
         'v007_pag_half', 'C042', 'Half-strength PAG on v007', 184),
        (ROOT / 'results/experiments/v007_pag_cfg3/comparison.json',
         'v007_pag_cfg3', 'C043', 'CFG 3.0 with half-strength PAG on v007', 187),
        (ROOT / 'results/experiments/v002_weight_decay003/screening.json',
         'v002_weight_decay003', 'C044', 'Matched AdamW weight decay', 190),
        (ROOT / 'results/experiments/v007_pag_cfg325/comparison.json',
         'v007_pag_cfg325', 'C045', 'CFG 3.25 with half-strength PAG on v007', 194),
        (ROOT / 'results/experiments/v007_continuation_032000/comparison.json',
         'v007_continuation_032000', 'C046', 'Bounded v007 continuation', 197),
        (ROOT / 'results/experiments/v007_32k_pag030/comparison.json',
         'v007_32k_pag030', 'C047', 'v007 with PAG 0.30', 200),
        (ROOT / 'results/experiments/v007_input_perturbation/comparison.json',
         'v007_input_perturbation', 'C049', 'Matched v007 input noise perturbation', 208),
    )
    for record_path, plan_name, claim_id, title, first_evidence_id in comparison_specs:
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        plan = ROOT / f'results/plans/{plan_name}.json'
        if record['plan_sha256'] != sha256(plan):
            raise ValueError(f'{title} plan changed')
        trials = record['screened']
        files, summaries = [plan, record_path], []
        if plan_name in ('rescale_interval', 'v002_temporal_score'):
            trials = [trial | {'setting': name} for name, trial in trials.items()]
        for trial in trials if isinstance(trials, list) else [trials]:
            row, = trial['evidence']
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError(f'{title} evaluation changed')
            result = validated_result(path, 0, ROOT / trial['exported'])
            metrics = result['metrics']
            files.append(path)
            accuracy = f"{metrics['accuracy']:.2%}"
            if metrics['accuracy'] < .90 and accuracy == '90.00%':
                accuracy += ' after rounding (below the official 90% threshold)'
            setting = f" with {trial['setting']}" if 'setting' in trial else ''
            summaries.append(f"At {trial['update']:,} training updates{setting}, generation seed 0 scored "
                             f"{result['marks']['total']}/15 with accuracy {accuracy}, "
                             f"FID {metrics['fid']:.2f}, {metrics['recall']:.2%} recall, and "
                             f"{metrics['memorization_rate']:.2%} memorization.")
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix()
               for index, file in enumerate(files, first_evidence_id)}
        paths.update(ids)
        sources = list(ids) + (['E111'] if plan_name == 'v002_dpmpp' else [])
        claims.append((claim_id, title, 'result', ' '.join(summaries), sources))
    rescale_path = ROOT / 'results/experiments/rescale_followup/comparison.json'
    if rescale_path.exists():
        plan = ROOT / 'results/plans/rescale_followup.json'
        result = json.loads(rescale_path.read_text())
        candidate = result['candidate']
        if result['plan_sha256'] != sha256(plan) or candidate is None:
            raise ValueError('The recorded rescale selection is incomplete')
        files = [plan, rescale_path]
        evidence = []
        for row in candidate['evidence']:
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('Rescale evaluation changed')
            evidence.append(validated_result(path, row['seed'], ROOT / candidate['exported']))
            files.append(path)
        if [r['protocol']['seed'] for r in evidence] != [0, 101, 202]:
            raise ValueError('The rescale candidate lacks a declared selection seed')
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix() for index, file in enumerate(files, 137)}
        paths.update(ids)
        marks = '/'.join(str(row['marks']['total']) for row in evidence)
        claims.append(('C031', 'Three-seed CFG rescale follow-up', 'result',
                       f"With the exact v002 EMA weights and 0.25 rescale, the official marks were {marks}. "
                       f"FID was {metric_range(evidence, 'fid', '.2f')}, recall {metric_range(evidence, 'recall', '.2%')}, "
                       f"and memorization {metric_range(evidence, 'memorization_rate', '.2%')}.", list(ids) + ['E112']))
    v007_path = ROOT / 'results/experiments/v007_32k_pag015/comparison.json'
    if v007_path.exists():
        plan_path = ROOT / 'results/plans/v007_32k_pag015.json'
        comparison = json.loads(v007_path.read_text())
        candidate = comparison['candidate']
        if comparison['plan_sha256'] != sha256(plan_path) or not comparison['improves_reference']:
            raise ValueError('The v007 selection or its frozen plan differs from the recorded winner')
        files, evidence = [plan_path, v007_path], []
        for row in candidate['evidence']:
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('The v007 selection receipt changed')
            evidence.append(validated_result(path, row['seed'], ROOT / candidate['exported']))
            files.append(path)
        if [row['protocol']['seed'] for row in evidence] != [0, 101, 202] or score_results(evidence) is None:
            raise ValueError('The v007 winner needs three valid selection seeds')
        ids = {f'E{index:03d}': file.relative_to(ROOT).as_posix() for index, file in enumerate(files, 203)}
        paths.update(ids)
        marks = '/'.join(str(row['marks']['total']) for row in evidence)
        claims.append(('C048', 'Three-seed v007 selection', 'result',
                       f"The v007 32k EMA with CFG 3.25 and PAG 0.15 scored {marks}. "
                       f"Accuracy was {metric_range(evidence, 'accuracy', '.2%')}, FID "
                       f"{metric_range(evidence, 'fid', '.2f')}, recall {metric_range(evidence, 'recall', '.2%')}, "
                       f"and memorization {metric_range(evidence, 'memorization_rate', '.2%')}.", list(ids)))
    return claims


def main():
    report = ROOT / 'report'
    report.mkdir(exist_ok=True)
    snapshot_path = report / 'evidence_snapshot.json'
    if snapshot_path.exists():
        old_snapshot = json.loads(snapshot_path.read_text())
        for name, expected in old_snapshot.get('generated_file_sha256', {}).items():
            if not (report / name).exists() or sha256(report / name) != expected:
                raise ValueError(f'Generated report evidence was edited; reconcile manually: {name}')
    paths = {'E001': 'results/curation/integrity.json', 'E002': 'results/curation/full/summary.json',
             'E003': 'results/diagnostics/overfit/result.json',
             'E004': 'results/diagnostics/benchmark/result.json',
             'E005': 'results/diagnostics/gpu_interface.json',
             'E006': 'configs/ddpm.yaml', 'E007': 'submission/model.py',
             'E008': 'results/setup/all_fast_tests.xml'}
    records = {key: json.loads((ROOT / path).read_text()) for key, path in paths.items() if path.endswith('.json')}
    integrity, full, overfit, benchmark, interface = [records[f'E{i:03d}'] for i in range(1, 6)]
    experiments = development_claims(paths)
    sources = []
    for identifier, path in paths.items():
        sources.append({'evidence_id': identifier, 'source_type': 'software' if path.endswith(('.py', '.yaml')) else 'report',
            'title': f'Local project evidence: {path}', 'authors': [], 'year': 2026,
            'identifiers': {'doi': '', 'pmid': '', 'pmcid': '', 'isbn': '', 'url': (ROOT / path).as_uri()},
            'locator': path, 'confidentiality': 'restricted',
            'verification': {'status': 'unverified', 'source_opened': False, 'verified_by': '', 'verified_on': ''}})
    literature = [
        ('E101', 'Denoising Diffusion Probabilistic Models', ['Jonathan Ho', 'Ajay Jain', 'Pieter Abbeel'],
         2020, 'https://arxiv.org/abs/2006.11239', 'Noise prediction objective and reverse diffusion', 'preprint'),
        ('E102', 'Denoising Diffusion Implicit Models', ['Jiaming Song', 'Chenlin Meng', 'Stefano Ermon'],
         2020, 'https://arxiv.org/abs/2010.02502', 'Deterministic sampling, eta=0', 'preprint'),
        ('E103', 'Improved Denoising Diffusion Probabilistic Models', ['Alexander Quinn Nichol', 'Prafulla Dhariwal'],
         2021, 'https://proceedings.mlr.press/v139/nichol21a.html', 'Section 3.2 cosine schedule', 'conference_paper'),
        ('E104', 'Classifier-Free Diffusion Guidance', ['Jonathan Ho', 'Tim Salimans'],
         2022, 'https://arxiv.org/abs/2207.12598', 'Section 3.2, guidance equation; our s=1+w', 'preprint'),
        ('E105', 'Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection',
         ['Shilong Liu', 'Zhaoyang Zeng', 'Tianhe Ren', 'Feng Li', 'Hao Zhang', 'Jie Yang', 'Qing Jiang',
          'Chunyuan Li', 'Jianwei Yang', 'Hang Su', 'Jun Zhu', 'Lei Zhang'],
         2023, 'https://arxiv.org/abs/2303.05499', 'Pool curation detector; author metadata checked against arXiv v5 record', 'preprint'),
        ('E106', 'Learning Transferable Visual Models From Natural Language Supervision',
         ['Alec Radford', 'Jong Wook Kim', 'Chris Hallacy', 'Aditya Ramesh', 'Gabriel Goh', 'Sandhini Agarwal',
          'Girish Sastry', 'Amanda Askell', 'Pamela Mishkin', 'Jack Clark', 'Gretchen Krueger', 'Ilya Sutskever'],
         2021, 'https://proceedings.mlr.press/v139/radford21a.html', 'CLIP review ranking; bibliographic metadata checked against PMLR', 'conference_paper'),
        ('E107', 'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents',
         ['Timothy Kassis', 'Vinayak Agarwal', 'Yuhuan He', 'Darshil Patel', 'Aubrey M. Brueckner'],
         2026, 'https://arxiv.org/abs/2609.00065', 'Materially used scientific writing/visualization procedures', 'preprint'),
        ('E108', 'Progressive Distillation for Fast Sampling of Diffusion Models', ['Tim Salimans', 'Jonathan Ho'],
         2022, 'https://arxiv.org/abs/2202.00512', 'Velocity parameterization used in the matched comparison', 'preprint'),
        ('E109', 'Efficient Diffusion Training via Min-SNR Weighting Strategy',
         ['Tiankai Hang', 'Shuyang Gu', 'Chen Li', 'Jianmin Bao', 'Dong Chen', 'Han Hu', 'Xin Geng', 'Baining Guo'],
         2023, 'https://arxiv.org/abs/2303.09556', 'Timestep weighting used in the matched epsilon comparison', 'preprint'),
        ('E110', 'Elucidating the Design Space of Diffusion-Based Generative Models',
         ['Tero Karras', 'Miika Aittala', 'Timo Aila', 'Samuli Laine'],
         2022, 'https://arxiv.org/abs/2206.00364', 'EDM preconditioning, weighted denoising loss and Heun sampler',
         'preprint'),
        ('E111', 'DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models',
         ['Cheng Lu', 'Yuhao Zhou', 'Fan Bao', 'Jianfei Chen', 'Chongxuan Li', 'Jun Zhu'],
         2022, 'https://arxiv.org/abs/2211.01095', 'Algorithm 2 and midpoint two-step update', 'preprint'),
        ('E112', 'Common Diffusion Noise Schedules and Sample Steps Are Flawed',
         ['Shanchuan Lin', 'Bingchen Liu', 'Jiashi Li', 'Xiao Yang'], 2024,
         'https://arxiv.org/abs/2305.08891', 'Section 3.4 per-image classifier-free guidance rescale', 'conference_paper')]
    for identifier, title, authors, year, url, locator, kind in literature:
        sources.append({'evidence_id': identifier, 'source_type': kind, 'title': title, 'authors': authors,
            'year': year, 'identifiers': {'url': url, 'doi': '10.48550/arXiv.2609.00065' if identifier == 'E107' else '',
                                        'pmid': '', 'pmcid': '', 'isbn': ''},
            'locator': locator, 'confidentiality': 'public',
            'verification': {'status': 'unverified', 'source_opened': False, 'verified_by': '', 'verified_on': ''}})
    # A refresh must never discard a named human's existing verification record.
    manifest_path = report / 'source_manifest.json'
    if manifest_path.exists():
        prior = {item['evidence_id']: item for item in json.loads(manifest_path.read_text())['sources']}
        for source in sources:
            old = prior.get(source['evidence_id'])
            if old and old['verification']['status'] != 'unverified':
                raise ValueError('Source verification exists; reconcile evidence manually before refreshing')
    write_json(manifest_path, {'schema_version': '1.0', 'sources': sources})
    write_json(report / 'evidence_snapshot.json', {'status': 'development_evidence_only' if experiments else 'implementation_evidence_only_not_final_results',
        'files': {identifier: {'path': path, 'sha256': sha256(ROOT / path)} for identifier, path in paths.items()},
        'verification': 'Agent inspected local files; named human source/claim checks remain pending.'})
    claims = [
        ('C001', 'Data', 'numeric', f"All {integrity['decoded_ok']:,} supplied pool images passed the recorded integrity scan.", ['E001']),
        ('C002', 'Data', 'numeric', f"The full detector run produced {full['proposals']:,} proposals and {full['geometrically_valid_proposals']:,} valid square proposals from {full['images_with_valid_proposals']:,} sources; these are not student acceptances.", ['E002']),
        ('C003', 'Method', 'numeric', f"The implemented channel fallback has {interface['parameters']:,} parameters.", ['E005','E006','E007']),
        ('C004', 'Verification', 'result', f"The separate two-image diagnostic reduced fixed-probe noise MSE from {overfit['probe_mse_before']:.6f} to {overfit['probe_mse_after']:.6f} in {overfit['updates']:,} updates; it is not evidence of generalization.", ['E003']),
        ('C005', 'Verification', 'result', f"The actual-architecture diagnostic benchmark measured {benchmark['steady_seconds_per_update']:.5f} steady seconds per update; its 50,000-update projection excludes evaluation and full-dataset overhead.", ['E004']),
        ('C006', 'Verification', 'result', f"The recorded CUDA interface stress test measured a maximum output difference of {interface['microbatch_max_absolute_difference']} when logical microbatch sizes changed while physical execution size stayed fixed.", ['E005'])] + experiments
    headings = ['# Evidence outline — incomplete course report', '',
        'This outline records engineering and development evidence. The final personal audit, independent checks and final submission results remain pending. The student must check sources and conclusions before submission.', '',
        'Planned main-text allocation: summary/problem 0.5 page; data/audit 1.25; methods 1.25; results/ablations 1.75; limitations 0.5; agent verification/conclusion 0.75. References and appendix are separate.', '']
    rows = []
    for cid, section, kind, sentence, ids in claims:
        headings.extend([f'## {section}: {cid}', '', f'{sentence} [claim:{cid}] [evidence:{",".join(ids)}]', ''])
        rows.append({'claim_id': cid, 'section': section, 'claim_kind': kind,
            'claim_text_sha256': hashlib.sha256(sentence.encode()).hexdigest(), 'evidence_ids': ';'.join(ids),
            'verification_status': 'unverified', 'uncertainty': 'not_estimated', 'analysis_intent': 'descriptive'})
    headings.extend(['## Evidence still required', '',
        '- Planned equal-budget ablations with exact checkpoint, manifest and code lineage.',
        '- Full official five-seed metric table: accuracy, FID, KID and variability, precision, recall, memorization; parameter count, hardware, measured training/sampling time.',
        '- Personal official 200-row audit with Wilson intervals, concrete errors, and class denominators.',
        '- Predeclared sample grid and nearest-neighbor diagnostics, with no selected best images.',
        '- Three authentic student independent checks, actual failures/agreements and resulting rules.',
        '- Actual student name/ID, source verification, final page count, figure selection, and rehearsal.', '',
        'The completed placeholder evaluator smoke only establishes environment operation. Its metrics must never enter the project result table.', '',
        'Scientific Agent Skills materially informed evidence handling and visualization. Its verified bibliographic record is retained as E107 and in references.bib; human confirmation remains pending.', ''])
    (report / 'evidence_outline.md').write_text('\n'.join(headings), encoding='utf-8')
    with (report / 'claims.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ids = ['acl23-0074_fig1', 'iclr26-0034_fig1', 'aaai25-0110_fig1', 'icml23-0003_fig1',
           'aaai25-0124_fig1', 'iclr23-0049_fig1', 'iclr25-0058_g1', 'cvpr25-0063_fig1',
           'aaai25-0026_fig1', 'neurips23-0070_fig1']
    exemplar_path = ROOT / 'figures/review/exemplars.json'
    if not exemplar_path.exists():
        cards = {item['id']: item for item in map(json.loads, (ROOT / '.agents/skills/figgenie-paper-diagram/assets/exemplars/ai/cards.jsonl').read_text(encoding='utf-8').splitlines())}
        write_json(exemplar_path, {'status': 'pending_student_choice',
            'copyright': 'Authors/publishers own reference figures. Study only; excluded from course archive.',
            'candidates': [dict(number=i+1, id=identifier, title=cards[identifier]['title'],
                                paper_url=cards[identifier]['paper_url'], layout=cards[identifier]['layout'])
                           for i, identifier in enumerate(ids)]})
    snapshot = json.loads(snapshot_path.read_text())
    snapshot['generated_file_sha256'] = {name: sha256(report / name)
        for name in ['source_manifest.json', 'evidence_outline.md', 'claims.csv']}
    write_json(snapshot_path, snapshot)
    print('Updated incomplete outline, source/claim registries and evidence hashes.')


if __name__ == '__main__':
    main()

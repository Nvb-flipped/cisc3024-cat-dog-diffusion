"""Write EXPERIMENTS.md: every experiment folder with its best official result (numbers read from results/experiments_index.csv,
built by scripts.build_experiments_index) plus a hand-written purpose and verdict.

    python -m scripts.build_experiments_index && python -m scripts.build_experiments_md

The hand-written text only restates what the numbers show (verdicts name the metric that decides them).
"""
import csv
from pathlib import Path

DATA = {r['experiment']: r for r in csv.DictReader(open('results/experiments_index.csv', newline='', encoding='utf-8'))}

# group -> [(folder, purpose, verdict)]
GROUPS = [
 ('A. First baselines and search on the 1,197-image manifest (v001/v002)', [
  ('baseline_v001_seed17', 'First conditional baseline on v001, guidance 1.0', 'Copied its training crops (memorization far above the 5% limit): 0 marks.'),
  ('quantity_v001_seed17', 'Quantity ablation on a balanced subset of the reviewed rows', 'A smaller training set copied even more: 0 marks.'),
  ('optimization_v001_seed17', 'Planned learning-rate, training-length and sampler comparisons (46 evaluations)', 'Best result is below the v002 reference (11/15, FID 71.6).'),
  ('refinement_v001_seed17', 'Refinement screens around the v001 run (EMA weights, 50 steps, guidance 2.5)', 'Below the v002 reference.'),
  ('capacity_32_seed17_v002', 'v002 reference: pixel U-Net, noise (epsilon) target', 'Reference for every matched comparison below: 11/15, memorization just under the limit.'),
  ('capacity_32_seed17', 'Width-32 U-Net screens at several checkpoints and samplers', '12/15 at best (FID 72.0, same as the reference; the extra mark comes from accuracy crossing 0.90).'),
  ('capacity_frontier_v001', 'Capacity-frontier screens (14 evaluations)', '12/15 at best, FID worse than the reference.'),
  ('capacity_repeatability_v001', 'Repeatability of the width-32 fit across training seeds 17/29/43', '12/15 at best (FID 73.65 on the best seed).'),
 ]),
 ('B. Loss, objective and architecture alternatives, matched to the reference (about 1.2-1.4k images)', [
  ('minsnr5_32_seed17_v002', 'Min-SNR (gamma 5) epsilon loss', 'Much worse than the reference (FID 139.8 vs 71.6).'),
  ('edm_32_seed17_v002', 'EDM-style training and sampling challenger', 'Much worse (FID 147.7).'),
  ('velocity_32_seed17_v002', 'Velocity (v) prediction target', 'Memorization 39%: 0 marks.'),
  ('patch_grid_v002', 'Pixel-patch U-Nets (4x4 and 8x8 patches)', 'Collapsed (FID 220, recall about 0).'),
  ('v002_coarse_epsilon', 'Coarse-resolution loss variant', 'Worse than the reference (FID 87.6).'),
  ('v002_weight_decay003', 'Higher weight decay', 'No gain (11/15, FID 71.65 vs 71.6).'),
  ('v006_width40', 'Width 40 vs width 32 at 26k updates (the result shown is the width-32 comparator, the folder\'s best)', 'The wider model did not beat width 32.'),
  ('dropout25_wide_seed17', 'Wide model with dropout 0.25', 'Worse (FID 91.2).'),
  ('v006_dropout15', 'Dropout 0.15', 'No gain: FID 72.3 and recall 0.237 are not better.'),
  ('v006_self_condition', 'Epsilon self-conditioning', 'Failed to train usefully (FID 268, recall 0).'),
  ('v007_input_perturbation', 'Noise-input perturbation during training', 'Small gain over the v007 expansion run (12/15, FID 70.9 vs 11/15, 74.9); later superseded by attention.'),
  ('v007_timestep_focus065', 'More training exposure to low-SNR timesteps', 'Memorization 51%: 0 marks.'),
  ('v007_timestep_focus065_20k', 'Same, 20k-update checkpoint', '8/15, FID 121.6.'),
  ('v007_timestep_focus065_26k', 'Same, 26k-update checkpoint', 'Memorization 16.5%: 0 marks.'),
  ('v007_score_average', 'Average noise prediction of two full fits', 'No gain (11/15, FID 75.9).'),
  ('ema_trajectory_probe', 'Average of same-run EMA snapshots', 'No gain (10/15).'),
  ('v007_v008_distribution_mixture', 'Fixed sample-space mixture of the v007 and v008 fits', 'Similar to v008 alone (12/15, FID 66.55 vs 66.8).'),
  ('v009_flow32k', 'Linear flow-matching recipe from scratch', 'Memorization 34.7%: 0 marks.'),
  ('v009_haar32k', 'Reversible 2x2 Haar U-Net', 'Worse (7/15, FID 121.4).'),
  ('v009_appearance32k', 'Changed training-crop sampling probabilities only', 'No gain over the uniform control (11/15, FID 71.4 vs 71.0).'),
  ('v009_attention16_32k', '16x16 encoder attention block', 'Better than the matched uniform control (12/15, FID 67.9 vs 11/15, 71.0): kept.'),
 ]),
 ('C. Adding personally reviewed batches (matched 32k comparisons)', [
  ('v004_coverage', 'First reviewed expansion (v004, 1,221 rows)', '11/15, FID 74.4: no gain.'),
  ('v006_coverage', 'v006 (1,296 rows)', '10/15, FID 70.85.'),
  ('v007_coverage', 'v007 (1,309 rows)', '11/15, FID 74.9.'),
  ('v008_coverage32k', 'v008 (1,369 rows)', '12/15, FID 66.8: first clear step.'),
  ('v008_extension34k', 'v008 continued to 34k', '12/15, FID 65.2, accuracy 0.95.'),
  ('v009_coverage32k', 'v009 (1,399 rows), uniform control', '11/15, FID 71.0.'),
 ]),
 ('D. Sampler-only experiments on frozen weights (no retraining)', [
  ('sampler_endpoint_screen_v001', 'Two DDIM endpoint settings kept from the v001 comparison', '11/15 but FID 92: not competitive.'),
  ('v002_sampler_comparison', 'DDIM start/endpoint comparison', '11/15, FID 71.2: no gain.'),
  ('v002_ddim_steps', 'More DDIM steps', 'No gain (10/15).'),
  ('v002_ddim_eta', 'Stochastic DDIM (eta 0.35)', 'Memorization just over the limit (5.15%): 0 marks.'),
  ('v002_dpmpp', 'DPM-Solver++', 'Worse (10/15, FID 79.5).'),
  ('v007_quadratic_ddim50', 'Quadratic DDIM spacing', '12/15, FID 67.6.'),
  ('v002_guidance_rescale', 'CFG rescale', '12/15 (accuracy crosses 0.90); FID 71.35 is barely different from 71.6.'),
  ('rescale_followup', 'Reduced CFG rescale on v002 and v006', '12/15, FID 71.3.'),
  ('rescale_interval', 'Restricted-interval guidance on the rescale export', 'Worse (11/15, FID 77.7).'),
  ('v002_pag', 'Perturbed-attention guidance (full strength)', 'Memorization 11.2%: 0 marks.'),
  ('v007_pag_half', 'PAG at half strength', '11/15, FID 70.2.'),
  ('v007_pag_cfg3', 'PAG half strength, guidance 3.0', '11/15, FID 67.7.'),
  ('v007_pag_cfg325', 'PAG half strength, guidance 3.25', '11/15, FID 66.7.'),
  ('v007_32k_pag015', 'PAG 0.15 on the 32k checkpoint', '12/15, FID 64.5: kept in the final sampler.'),
  ('v007_32k_pag030', 'PAG 0.30', '12/15, FID 66.3.'),
  ('v007_perturbation_pag050', 'Stronger PAG on the perturbation checkpoint', '11/15, FID 68.5.'),
  ('v007_perturbation_34k', '34k vs 32k perturbation checkpoint', '11/15, FID 69.0.'),
  ('v002_temporal_score', 'Temporal score rescaling', '12/15, FID 70.1.'),
  ('v007_32k_temporal103', 'Temporal score 1.03', '12/15, FID 63.7: kept in the final sampler.'),
  ('v007_32k_temporal105', 'Temporal score 1.05', '12/15, FID 66.0.'),
  ('v007_freeu_mild', 'FreeU decoder feature profile (mild)', '12/15, FID 61.6: FreeU was adopted (the final sampler uses strength 2.0).'),
  ('v007_freeu_cfg310', 'FreeU with guidance 3.10', '12/15, FID 62.2, memorization 4.95% (almost at the limit).'),
  ('v007_freeu_strength150', 'Stronger FreeU (1.5)', '12/15, FID 61.3.'),
  ('v007_freeu_ddim100', '100 DDIM steps with FreeU', 'Memorization 5.35%: 0 marks.'),
  ('v009_34k_freeu180', 'FreeU 1.8 on the v009 34k weights', '12/15, FID 60.9.'),
  ('v009_34k_freeu220', 'FreeU 2.2', '12/15, FID 60.5.'),
  ('v009_34k_cfg310', 'Guidance 3.10 on v009 34k', '12/15, FID 61.6.'),
  ('v009_34k_pag010', 'PAG 0.10 on v009 34k', '12/15, FID 60.9.'),
  ('v009_attention16_freeu', 'FreeU on the attention checkpoint', '12/15, FID 64.25.'),
  ('v009_attention16_freeu200', 'FreeU 2.0 on the attention checkpoint', '12/15, FID 62.5.'),
  ('v007_class_annealing010', 'Class-condition noise annealing 0.10', '12/15, FID 64.5: kept (class noise 0.1 is in the final sampler).'),
  ('v007_class_annealing010_rescale025', 'Annealing 0.10 with CFG rescale 0.25', '12/15, FID 64.65: no gain.'),
  ('v007_class_annealing025', 'Annealing 0.25', '11/15, FID 69.0: worse.'),
  ('v007_autoguidance', 'Same-label autoguidance', 'Memorization 8.4%: 0 marks.'),
  ('v002_reflected_denoiser', 'Horizontal reflection averaging of the denoiser', 'Worse (7/15, FID 100.6).'),
  ('high_noise_probe', 'High-noise DDIM change', 'No gain (12/15, FID 74.2).'),
 ]),
 ('E. Continuations and learning-rate changes (v006-v009)', [
  ('v007_continuation_032000', 'Resume v007 for one interval', 'Memorization 5.65%: 0 marks.'),
  ('v009_extension36k', 'v009 extended by 4k updates (36k)', 'Memorization 5.05%: 0 marks; with guidance 3.0 it recovers.'),
  ('v009_extension36k_cfg3', 'Same, guidance 3.0', '12/15, FID 68.1: worse than 34k.'),
  ('v009_attention16_extension34k', 'Attention model extended to 34k', '12/15, FID 60.7, recall 0.345: the selected v009 development model.'),
  ('v009_attention16_extension35k', 'Extended to 35k', '12/15, FID 60.5.'),
  ('v009_attention16_lr100', 'Lower learning rate continuation', '12/15, FID 60.4.'),
  ('v009_attention16_lr100_35k', 'One more 1k at the lower rate', 'Memorization 5.05%: 0 marks.'),
  ('v009_lr100_35k_freeu220', 'FreeU 2.2 on the lower-rate 35k checkpoint', '12/15, FID 60.0.'),
  ('v009_linux_baseline', 'v009 34k reproduced on this Linux machine (CPU)', 'Reproduces 12/15; this is the 12/15 starting point of the final push.'),
 ]),
 ('F. Linux generation: mirroring, more data, EMA (v010-v014)', [
  ('v010_attn16', 'All 50 personally reviewed batches (1,929 rows), no mirroring', '13/15 at best (FID 58.0, recall 0.319).'),
  ('v010_attn16_flip', 'Same plus left-right mirroring of exact crops', 'Largest single gain: 14/15, FID 51.0, recall 0.42, memorization 0%. Only 13-14 across seeds.'),
  ('v010_flip_ext', 'Mirroring run continued to 160k updates', 'No gain: FID rises with length (50.9 to 53.1).'),
  ('v010_wide_flip', 'Wider 10.7M-parameter U-Net with mirroring', 'Memorization climbs to 5.5% at 80k: 0 marks there; recall below the narrow model.'),
  ('v011_flip', 'v011: 5,536 rows (1,929 personal + 3,607 AI-assisted)', '15/15 on two of three seeds: the first run to reach FID below 50.'),
  ('v011_ema9995_flip', 'v011 with EMA 0.9995', '15/14/15 on three seeds; the recall margin is thin.'),
  ('v011_wide_flip', 'v011 wide model', '14/15: sharper (precision 0.86) but recall 0.385.'),
  ('v012_flip', 'v012: 4,610 rows (agent-approved AI rows only)', '14/14/15: cleaner but smaller data, recall 0.347-0.402.'),
  ('v013_flip', 'v013: 4,922 rows (v012 + recovered rejects)', '14/15; recall 0.33-0.365.'),
  ('v013_ema9995_flip', 'v013 with EMA 0.9995', '14/15; recall 0.365-0.388.'),
  ('v014_flip', 'v014: 8,029 rows, EMA 0.9999', '15/15/14 at 80k; 15/14/15 at 140k.'),
  ('v014_ema9995_flip', 'v014, EMA 0.9995: FINAL model (80k updates)', 'SUBMITTED checkpoint (80k): 15 on 60 of 61 official seeds (accuracy 0.971, FID 46.7, recall 0.433 +/- 0.019, minimum 0.395, memorization 0%); seed 31415 scores 14. The metrics column shows the folder\'s best single evaluation (a 120k checkpoint).'),
 ]),
]

DATA_ROWS = [
 ('v001, v002', '1,197 (583 cat / 614 dog)', 'student-reviewed', 'First reviewed sets; v002 is the reference recipe'),
 ('v003-v009', '1,205 to 1,399', 'student-reviewed', 'Successive freezes as more batches were reviewed (v006: 1,296; v008: 1,369; v009: 1,399, 669 cat / 730 dog)'),
 ('v010', '1,929 (880 / 1,049)', 'student-reviewed', 'All 50 personally reviewed batches'),
 ('v011', '5,536 (2,375 / 3,161)', 'AI-assisted', 'v010 + 3,607 AI-found rows (screen pass, about 74% valid on agent review)'),
 ('v012', '4,610 (1,962 / 2,648)', 'AI-assisted', 'v010 + 2,681 AI rows kept by strict agent review'),
 ('v013', '4,922 (2,120 / 2,802)', 'AI-assisted', 'v012 + 351 recovered framing rejects (minus duplicates)'),
 ('v014 (final)', '8,029 (3,096 / 4,933)', 'AI-assisted', '1,929 personal + 6,100 agent-reviewed AI rows under the exact handout specification'),
]


def cell(name):
    r = DATA[name]
    marks = int(r['best_marks'])
    mem = float(r['memorization']) * 100
    first = f"{marks}/15" if marks else '0 (limit)'
    return f"{first}, acc {float(r['accuracy']):.3f}, FID {float(r['fid']):.1f}, recall {float(r['recall']):.3f}, mem {mem:.2f}%"


def main():
    lines = ['# Experiments index',
             '',
             'Every experiment folder under `results/experiments/` (92 in total), with its **best official evaluation** (most marks, then lowest FID; '
             '"0 (limit)" means every evaluation broke a hard limit, in every such case memorization above 5%) and a one-line verdict. '
             'Numbers are read from the recorded result files by `scripts/build_experiments_index.py` (`results/experiments_index.csv`); '
             'this file is generated by `scripts/build_experiments_md.py`. All evaluations use the unchanged official evaluator on the dev reference. '
             'Ranges and per-seed numbers are in each folder; decisions are in `results/fullpool_20261119/DECISION_LOG.md`.',
             '',
             '## What did not work (summary for the report)',
             '',
             '1. **Memorization at about 1.2k images** was the binding limit: 13 of 92 folders have 0 marks because memorization exceeded 5% (up to 73.8%).',
             '2. **Alternative objectives and architectures** (Min-SNR, EDM, velocity, patches, coarse loss, flow matching, Haar) were all worse than the plain epsilon U-Net at matched budget (FID 87 to 220 vs 71.6).',
             '3. **Extra regularization** (dropout 0.15/0.25, self-conditioning, weight decay) gave no gain.',
             '4. **Sampler tricks** (FreeU, PAG, temporal score, class-noise annealing) each gained a few FID points and are in the final sampler, but could not get below FID 60 with about 1.4k images.',
             '5. **Mirroring and more data** did: mirroring took v010 from 13 to 14 marks and removed memorization; 5.5k and 8k images reached FID about 47 and 15/15 on most seeds.',
             '6. **A wider model** was sharper (precision up) but had lower recall and, on small data, more memorization.',
             '7. **Cleaner but smaller data** (v012, v013) lowered recall (the binding metric) compared with the larger v011/v014 sets.',
             '8. **Seed noise matters**: recall varies by about 0.02 between seeds, so a single seed can hide a failure; the final model has one 14/15 seed in 61.',
             '',
             '## Data versions', '',
             '| Manifest | Training rows | Origin | Note |', '|---|---|---|---|']
    lines += [f'| {a} | {b} | {c} | {d} |' for a, b, c, d in DATA_ROWS]
    seen = set()
    for title, rows in GROUPS:
        lines += ['', f'## {title}', '', '| Folder | What was tried | Best official result | Verdict |', '|---|---|---|---|']
        for name, purpose, verdict in rows:
            if name in seen or name not in DATA or not verdict:
                continue
            seen.add(name)
            lines.append(f'| [{name}](results/experiments/{name}) | {purpose} | {cell(name)} | {verdict} |')
    missing = sorted(set(DATA) - seen)
    lines += ['', '## In progress (21 Nov 2026)', '',
              '`v014_wide_ema9995_flip` (10.7M parameters) and `v014_mid_ema9995_flip` (about 8M) on the frozen v014 manifest: a bounded attempt to improve image quality; '
              'replacement rule fixed beforehand in decision log entry 75. Results are not in the table yet.',
              '', '## Checkpoints',
              '',
              'On 21 Nov 2026 about 141.6 GB of unevaluated intermediate checkpoints were deleted (ledger `results/pruning/ledger_*.json`, decision log entries 76-77). '
              'Every checkpoint cited by any result, the last checkpoint of each run, the final model and the receipts of all checkpoints remain. '
              'The kept Linux-era set is in private release `linux-era-backup-2026-11-21`, older runs in `migration-2026-11-19`.']
    if missing:
        lines += ['', f'Folders not described above: {", ".join(missing)}']
    Path('EXPERIMENTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(len(seen), 'folders described;', len(missing), 'not described:', missing)


if __name__ == '__main__':
    main()

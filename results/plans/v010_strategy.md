# v010 strategy: closing the FID and recall gap (agent analysis, 2026-11-19)

Written by the coding agent during a 12-hour window in which Paco delegated model-choice and
refinement decisions. Every claim below cites existing receipts; nothing here is a new measured
result until the corresponding official evaluation exists under `results/experiments/v010_*`.

## Where the marks are lost

Selected v009 attention model, 34k updates, FreeU 2.0, seeds 0/101/202
([receipts](../experiments/v009_attention16_extension34k/)):

| Metric | Range | Excellent | Marks |
|---|---|---|---|
| Accuracy | 0.925–0.936 | ≥ 0.90 | 5 |
| FID | 60.67–61.16 | ≤ 50 | 3 (Pass; Good needs ≤ 60) |
| Recall | 0.311–0.345 | ≥ 0.40 | 4 |
| Memorization | 3.85–4.55% | ≤ 5% hard limit | — |

## Diagnosis

Across 221 stored official evaluations, every recipe that stays under the 5% memorization
limit lands at FID 60–72 and recall 0.31–0.35. This covers about forty sampler and objective
variants (PAG, FreeU, CFG, DPM-Solver++, EDM, flow matching, Haar packing, self-conditioning,
dropout, learning rate). The only runs below FID 53 or above recall 0.40 memorized 34–74% of
samples (e.g. `optimization_v001_seed17/training/lr_0.0003_u050000`, FID 51.4, recall 0.445,
memorization 61.9%). The v009 runs show memorization rising by roughly 0.5 points per 1k updates
near 34k (34k 4.25%, 35k 4.70%, uniform 36k 5.05%). The model sees each of its 1,399 crops about
3,100 times before the memorization cap stops training. The binding constraint is
the number of distinct training targets, not the sampler.

The official memorization test (`starter_kit/evaluate.py`, `catdog_eval/metrics.py`) is a
pixel-space nearest-neighbour ratio `d1/d2 < 0.5` against the deduplicated training images
**plus their mirror images**. More distinct targets both delay copying and shrink `d2`.

## Decisions

1. **Use all 50 personally reviewed batches (v010).** Freeze with
   `--preserve-split-from manifests/v009.csv`: every v009 training row and the byte-identical
   63-row holdout are retained, and the remaining kept crops join training (about 1,929 rows,
   +38%). The freeze re-verifies personal verdict provenance, duplicate-family decisions,
   preview hashes and occupancy; the cache is checked against the official reconstruction.
2. **Allow whole-crop horizontal mirroring during training (p = 0.5).**
   Reconciliation with the handout: the handout requires training "only on the images in your
   manifest" and forbids "augmented or transformed versions" of *dev* images; it does not forbid
   transforming manifest crops. The official evaluator explicitly anticipates this: "Mirror
   images count as copies too (horizontal flips are a common augmentation)" and it adds every
   training mirror to the memorization reference. A mirrored crop is a pixel-exact permutation
   of a manifest row, with no resampling, synthesis or external data. No other augmentation is
   used. The option defaults to off and draws no random numbers when off, so all earlier runs
   remain reproducible (`tests/test_horizontal_flip.py`).
3. **Experiment order.** (a) v010 + mirroring with the v009 attention recipe, longer budget
   (80k, checkpoints every 4k), because it attacks the binding constraint most directly;
   (b) matched v010 without mirroring as the data-only control; (c) a wider U-Net with mirroring
   only if (a) shows memorization headroom at its best FID. Sampler retuning (guidance, FreeU)
   only on a selected checkpoint.
4. **Selection.** Official seed 0 screens across checkpoints; reject any hard-check failure;
   rank by tier total, then the minimum normalized margin, then memorization. Seeds 101/202 for
   selection; freeze before confirmation seeds 303/404. Internal target memorization ≤ 3.5% to
   leave margin for the hidden seed.

## Environment note

Training moved from the Windows RTX 5060 Ti to a shared Linux RTX 5090 (Python 3.11,
torch 2.9.1+cu128). GPU work waits for Paco's own jobs to finish. New-hardware results are
re-measured, not assumed identical to earlier receipts.

# Evidence outline — incomplete course report

This outline records engineering and development evidence. The final personal audit, independent checks and final submission results remain pending. The student must check sources and conclusions before submission.

Planned main-text allocation: summary/problem 0.5 page; data/audit 1.25; methods 1.25; results/ablations 1.75; limitations 0.5; agent verification/conclusion 0.75. References and appendix are separate.

## Data: C001

All 76,250 supplied pool images passed the recorded integrity scan. [claim:C001] [evidence:E001]

## Data: C002

The full detector run produced 125,941 proposals and 47,855 valid square proposals from 26,925 sources; these are not student acceptances. [claim:C002] [evidence:E002]

## Method: C003

The implemented channel fallback has 10,664,371 parameters. [claim:C003] [evidence:E005,E006,E007]

## Verification: C004

The separate two-image diagnostic reduced fixed-probe noise MSE from 1.155416 to 0.017217 in 2,000 updates; it is not evidence of generalization. [claim:C004] [evidence:E003]

## Verification: C005

The actual-architecture diagnostic benchmark measured 0.10579 steady seconds per update; its 50,000-update projection excludes evaluation and full-dataset overhead. [claim:C005] [evidence:E004]

## Verification: C006

The recorded CUDA interface stress test measured a maximum output difference of 0.0 when logical microbatch sizes changed while physical execution size stayed fixed. [claim:C006] [evidence:E005]

## Data: C007

The personally reviewed v001 split contains 583 cat and 614 dog training crops and a disjoint holdout of 31 cat and 32 dog crops. [claim:C007] [evidence:E009,E010,E011]

## Development results: C008

The full-width seed-17 baseline at 50,000 updates reached 90.15% accuracy and 52.90 FID on evaluator seed 0. Its 57.80% memorization exceeded the 5% limit. [claim:C008] [evidence:E012,E013]

## Development results: C009

The 21,000-update EMA development candidate using 50 DDIM steps and CFG 2.5 passed the hard conditions on 5 of 5 seeds. Accuracy was 88.75%–89.95%, FID 77.94–79.85, recall 22.35%–28.54%, and memorization 4.45%–4.75%. [claim:C009] [evidence:E014,E015,E016,E017,E018,E019,E020]

## Development results: C010

The 30,000-update smaller EMA candidate passed on 5 of 5 seeds. Accuracy was 89.65%–91.35%, FID 71.99–74.23, recall 27.43%–31.42%, and memorization 3.45%–4.45%. [claim:C010] [evidence:E021,E022,E023,E024,E025,E026,E027]

## Matched data quantity comparison: C011

At 50,000 updates, the balanced 1,000-crop subset reached 93.35% accuracy, FID 51.75, and 44.47% recall on seed 0. Memorization was 73.75% for the subset and 57.80% for the 1,197-crop parent. Both exceeded the 5% limit. [claim:C011] [evidence:E012,E028,E029,E030]

## Training-length and sampler comparison: C012

The 32,000-update small candidate with 50 DDIM steps and CFG 2.0 reached 36.62% recall on seed 0. Seed 101 had 100 memorization flags among 2,000 generated crops. The official evaluator stored the rate as 0.05000000075 and rejected the candidate. [claim:C012] [evidence:E031,E032,E033]

## Matched residual dropout comparison: C013

At 22,000 updates on seed 0, increasing dropout from 0.10 to 0.25 changed memorization from 8.05% to 0.20%, FID from 72.89 to 119.97, and recall from 27.32% to 19.91%. The larger dropout reduced memorization and worsened sample quality at the matched budget. [claim:C013] [evidence:E034,E035,E036,E037,E038]

## Raw and EMA sampler screening: C014

On the fixed 20,000-update checkpoint, all 12 raw-weight sampler screens exceeded the 5% memorization limit. Their rates were 36.4% to 57.8%, compared with 0.2% to 2.4% for EMA. Each setting generated 500 images on seed 0. [claim:C014] [evidence:E039]

## Recorded review geometry: C015

The frozen development split recorded median source head widths of 96.9 pixels for cats and 91.3 pixels for dogs. Recomputed width divided by side flagged 1 of 1,260 kept crops for a personal boundary recheck. [claim:C015] [evidence:E040,E041]

## Fresh training-seed comparison: C016

Training seed 17 produced FID 71.99–74.07, recall 27.43%–31.42%, and memorization 3.55%–4.45%. Training seed 29 produced FID 93.07–93.78, recall 25.00%–28.76%, and memorization 3.20%–3.45%. Training seed 43 produced FID 73.65–74.53, recall 30.53%–31.08%, and memorization 4.00%–5.00%. Each run used the same 30,000-update recipe and three 2,000-image development evaluations. [claim:C016] [evidence:E042,E043,E044,E045,E046,E047,E048,E049,E050,E051,E052]

## Corrected-manifest reference: C017

The fresh 30,000-update v002 reference passed all hard conditions on 3 of 3 development seeds, with scores 11, 11, 11 of 15. Accuracy was 89.40%–89.95%, FID 71.59–72.80, recall 30.20%–32.30%, and memorization 4.10%–4.60%. [claim:C017] [evidence:E053,E054,E055,E056,E057,E058]

## Matched velocity target: C018

The matched v002 comparison scored 0, 0, 0 out of 15 across the three official development seeds, with FID 93.43–95.45, recall 24.12%–24.45%, and memorization 37.90%–39.00%. [claim:C018] [evidence:E059,E060,E061,E062,E063]

## Matched sampler endpoint: C019

The matched v002 comparison scored 10, 11, 11 out of 15 across the three official development seeds, with FID 71.22–73.19, recall 29.20%–32.74%, and memorization 3.75%–4.35%. [claim:C019] [evidence:E064,E065,E066,E067,E068]

## Matched Min-SNR weighting: C020

At 20,000 updates, FID was 207.06, memorization was 0.00%, and the score was 3/15. At 24,000 updates, FID was 139.76, memorization was 2.75%, and the score was 7/15. At 28,000 updates, FID was 101.63, memorization was 19.65%, and the score was 0/15. At 30,000 updates, FID was 90.97, memorization was 30.60%, and the score was 0/15. [claim:C020] [evidence:E069,E070,E071,E072,E073,E074]

## Matched DDIM step count: C021

At 100 steps, seed 0 scored 10/15 with FID 73.91, recall 29.20%, and memorization 4.35%. [claim:C021] [evidence:E075,E076,E077]

## Evidence still required

- Planned equal-budget ablations with exact checkpoint, manifest and code lineage.
- Full official five-seed metric table: accuracy, FID, KID and variability, precision, recall, memorization; parameter count, hardware, measured training/sampling time.
- Personal official 200-row audit with Wilson intervals, concrete errors, and class denominators.
- Predeclared sample grid and nearest-neighbor diagnostics, with no selected best images.
- Three authentic student independent checks, actual failures/agreements and resulting rules.
- Actual student name/ID, source verification, final page count, figure selection, and rehearsal.

The completed placeholder evaluator smoke only establishes environment operation. Its metrics must never enter the project result table.

Scientific Agent Skills materially informed evidence handling and visualization. Its verified bibliographic record is retained as E107 and in references.bib; human confirmation remains pending.

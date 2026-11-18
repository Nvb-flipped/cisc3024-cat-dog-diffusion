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

## Stochastic DDIM comparison: C022

At eta 0.35, accuracy was 90.80%, FID 76.47, recall 34.29%, and memorization 5.15%. At eta 0.70, accuracy was 92.00%, FID 78.44, recall 31.42%, and memorization 7.60%. [claim:C022] [evidence:E078,E079,E080]

## Reviewed data expansion: C023

The v003 freeze contains 1,205 training crops (587 cats and 618 dogs) and preserves the same 63 holdout crops. All cached labels and crop pixels matched the official reconstruction. [claim:C023] [evidence:E081,E082,E083]

## EDM integration diagnostic: C024

The two-crop EDM diagnostic reduced fixed-probe MSE from 1.19148 to 0.01261 in 2,000 updates (98.94% reduction). [claim:C024] [evidence:E084,E110]

## Matched EDM comparison: C025

At 24,000 updates, accuracy was 87.05%, FID 147.67, recall 6.86%, memorization 1.10%, and the score 5/15. At 28,000 updates, accuracy was 89.95%, FID 116.20, recall 14.27%, memorization 8.55%, and the score 0/15. At 30,000 updates, accuracy was 90.85%, FID 103.11, recall 18.14%, memorization 14.75%, and the score 0/15. [claim:C025] [evidence:E085,E086,E087,E088,E089]

## Matched patch-grid comparison: C026

At 24,000 updates with 4×4 patches, accuracy was 66.85%, FID 283.17, recall 0.00%, memorization 0.00%, and the score 2/15. At 30,000 updates with 4×4 patches, accuracy was 65.65%, FID 220.25, recall 0.22%, memorization 0.00%, and the score 2/15. At 24,000 updates with 8×8 patches, accuracy was 53.80%, FID 276.45, recall 0.00%, memorization 0.00%, and the score 1/15. At 30,000 updates with 8×8 patches, accuracy was 56.55%, FID 237.26, recall 0.00%, memorization 0.00%, and the score 1/15. [claim:C026] [evidence:E090,E091,E092,E093,E094,E095,E096]

## Matched pool coverage comparison: C027

At 28,000 training updates, generation seed 0 scored 11/15 with accuracy 86.85%, FID 79.22, 31.31% recall, and 2.05% memorization. At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 88.10%, FID 74.42, 33.52% recall, and 3.95% memorization. [claim:C027] [evidence:E120,E121,E122,E123]

## Matched DPM-Solver++ comparison: C028

At 30,000 training updates, generation seed 0 scored 10/15 with accuracy 88.10%, FID 79.46, 29.20% recall, and 4.25% memorization. [claim:C028] [evidence:E125,E126,E127,E111]

## Larger matched pool coverage comparison: C029

At 28,000 training updates, generation seed 0 scored 10/15 with accuracy 88.70%, FID 75.62, 24.67% recall, and 1.35% memorization. At 30,000 training updates, generation seed 0 scored 10/15 with accuracy 89.50%, FID 72.13, 25.88% recall, and 2.55% memorization. [claim:C029] [evidence:E129,E130,E131,E132]

## Bounded v006 continuation: C030

At 32,000 training updates, generation seed 0 scored 10/15 with accuracy 90.00% after rounding (below the official 90% threshold), FID 70.85, 28.43% recall, and 4.05% memorization. [claim:C030] [evidence:E134,E135,E136]

## Matched self-conditioning comparison: C032

At 28,000 training updates, generation seed 0 scored 1/15 with accuracy 52.45%, FID 286.27, 0.00% recall, and 0.00% memorization. At 30,000 training updates, generation seed 0 scored 1/15 with accuracy 52.65%, FID 268.22, 0.00% recall, and 0.00% memorization. [claim:C032] [evidence:E143,E144,E145,E146]

## High-noise dynamic thresholding: C033

At 30,000 training updates, generation seed 0 scored 12/15 with accuracy 90.65%, FID 74.20, 31.64% recall, and 3.95% memorization. [claim:C033] [evidence:E148,E149,E150]

## Same-trajectory EMA snapshot average: C034

At 30,000 training updates, generation seed 0 scored 10/15 with accuracy 89.85%, FID 73.94, 29.98% recall, and 3.55% memorization. [claim:C034] [evidence:E153,E154,E155]

## Matched residual dropout comparison: C035

At 28,000 training updates, generation seed 0 scored 10/15 with accuracy 89.65%, FID 75.94, 23.78% recall, and 0.65% memorization. At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 90.20%, FID 72.30, 23.67% recall, and 1.05% memorization. [claim:C035] [evidence:E158,E159,E160,E161]

## Restricted CFG with rescale 0.25: C036

At 30,000 training updates with late_off, generation seed 0 scored 11/15 with accuracy 88.45%, FID 77.85, 30.20% recall, and 3.90% memorization. At 30,000 training updates with ends_off, generation seed 0 scored 11/15 with accuracy 87.60%, FID 77.68, 30.53% recall, and 3.50% memorization. [claim:C036] [evidence:E162,E163,E164,E165]

## Matched 4×4 block-average epsilon loss: C037

At 28,000 training updates, generation seed 0 scored 8/15 with accuracy 87.20%, FID 93.00, 16.92% recall, and 3.15% memorization. At 30,000 training updates, generation seed 0 scored 9/15 with accuracy 88.30%, FID 87.62, 22.12% recall, and 4.90% memorization. [claim:C037] [evidence:E166,E167,E168,E169]

## Horizontal reflection denoiser: C038

At 30,000 training updates, generation seed 0 scored 7/15 with accuracy 88.30%, FID 100.63, 17.15% recall, and 0.05% memorization. [claim:C038] [evidence:E170,E171,E172]

## Temporal score rescaling: C039

At 30,000 training updates with broader, generation seed 0 scored 11/15 with accuracy 89.45%, FID 77.44, 30.53% recall, and 3.80% memorization. At 30,000 training updates with sharper, generation seed 0 scored 12/15 with accuracy 90.70%, FID 70.10, 31.75% recall, and 4.70% memorization. [claim:C039] [evidence:E173,E174,E175,E176]

## Perturbed-attention guidance: C040

At 30,000 training updates, generation seed 0 scored 0/15 with accuracy 90.60%, FID 65.18, 32.19% recall, and 11.15% memorization. [claim:C040] [evidence:E177,E178,E179]

## Matched Batch 19 expansion: C041

At 28,000 training updates, generation seed 0 scored 10/15 with accuracy 89.20%, FID 78.05, 28.98% recall, and 0.85% memorization. At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 88.45%, FID 74.92, 30.75% recall, and 1.60% memorization. [claim:C041] [evidence:E180,E181,E182,E183]

## Half-strength PAG on v007: C042

At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 88.85%, FID 70.16, 32.52% recall, and 3.05% memorization. [claim:C042] [evidence:E184,E185,E186]

## CFG 3.0 with half-strength PAG on v007: C043

At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 89.75%, FID 67.67, 32.52% recall, and 3.95% memorization. [claim:C043] [evidence:E187,E188,E189]

## Matched AdamW weight decay: C044

At 28,000 training updates, generation seed 0 scored 10/15 with accuracy 88.80%, FID 76.59, 28.10% recall, and 1.75% memorization. At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 89.50%, FID 71.65, 31.08% recall, and 3.25% memorization. [claim:C044] [evidence:E190,E191,E192,E193]

## CFG 3.25 with half-strength PAG on v007: C045

At 30,000 training updates, generation seed 0 scored 11/15 with accuracy 89.90%, FID 66.68, 33.74% recall, and 4.40% memorization. [claim:C045] [evidence:E194,E195,E196]

## Bounded v007 continuation: C046

At 32,000 training updates, generation seed 0 scored 0/15 with accuracy 91.05%, FID 64.85, 34.62% recall, and 5.65% memorization. [claim:C046] [evidence:E197,E198,E199]

## v007 with PAG 0.30: C047

At 32,000 training updates, generation seed 0 scored 12/15 with accuracy 90.50%, FID 66.32, 33.30% recall, and 4.50% memorization. [claim:C047] [evidence:E200,E201,E202]

## Matched v007 input noise perturbation: C049

At 32,000 training updates, generation seed 0 scored 12/15 with accuracy 90.05%, FID 70.88, 31.97% recall, and 3.35% memorization. [claim:C049] [evidence:E208,E209,E210]

## v007 temporal sharpening 1.05: C051

At 32,000 training updates, generation seed 0 scored 12/15 with accuracy 90.25%, FID 65.96, 33.63% recall, and 4.15% memorization. [claim:C051] [evidence:E217,E218,E219]

## Three-seed CFG rescale follow-up: C031

With the exact v002 EMA weights and 0.25 rescale, the official marks were 12/12/11. FID was 70.53–72.16, recall 30.75%–34.62%, and memorization 4.30%–4.85%. [claim:C031] [evidence:E137,E138,E139,E140,E141,E112]

## Three-seed v007 selection: C048

The v007 32k EMA with CFG 3.25 and PAG 0.15 scored 12/12/12. Accuracy was 90.30%–91.35%, FID 64.52–67.08, recall 32.19%–33.85%, and memorization 4.00%–4.45%. [claim:C048] [evidence:E203,E204,E205,E206,E207]

## Three-seed v007 temporal sharpening 1.03: C050

The retained temporal setting scored 12/12/12. Accuracy was 90.20%–91.90%, FID 63.71–66.28, recall 31.08%–34.85%, and memorization 4.05%–4.65%. [claim:C050] [evidence:E211,E212,E213,E214,E215,E216]

## v007 class-embedding noise annealing: C051

Development scores out of 15 were 12, 12, 12. Accuracy was 91.90%–93.35%, FID 64.49–65.05, recall 30.64%–34.29%, and memorization 3.60%–4.30%. [claim:C051] [evidence:E217,E218,E219,E220,E221,E222]

## v008 1,369-crop fit: C052

Development scores out of 15 were 12. Accuracy was 93.85%–93.85%, FID 66.82–66.82, recall 34.29%–34.29%, and memorization 3.00%–3.00%. [claim:C052] [evidence:E223,E224,E225]

## v009 1,399-crop fit: C053

Development scores out of 15 were 11. Accuracy was 91.10%–91.10%, FID 71.02–71.02, recall 28.76%–28.76%, and memorization 2.10%–2.10%. [claim:C053] [evidence:E226,E227,E228]

## v009 appearance-weighted fit: C054

Development scores out of 15 were 11. Accuracy was 92.15%–92.15%, FID 71.38–71.38, recall 29.42%–29.42%, and memorization 2.80%–2.80%. [claim:C054] [evidence:E229,E230,E231]

## v009 Haar fit: C055

Development scores out of 15 were 7. Accuracy was 80.35%–80.35%, FID 121.42–121.42, recall 19.69%–19.69%, and memorization 3.50%–3.50%. [claim:C055] [evidence:E232,E233,E234]

## v009 rectified-flow fit: C056

Development scores out of 15 were 0. Accuracy was 91.00%–91.00%, FID 90.96–90.96, recall 28.87%–28.87%, and memorization 34.65%–34.65%. [claim:C056] [evidence:E235,E236,E237]

## v009 36k extension: C057

Development scores out of 15 were 0. Accuracy was 92.35%–92.35%, FID 67.12–67.12, recall 33.19%–33.19%, and memorization 5.05%–5.05%. [claim:C057] [evidence:E238,E239,E240]

## v009 36k extension with CFG 3.0: C058

Development scores out of 15 were 12. Accuracy was 91.95%–91.95%, FID 68.12–68.12, recall 31.42%–31.42%, and memorization 4.65%–4.65%. [claim:C058] [evidence:E241,E242,E243]

## v007 decoder feature reweighting: C059

Development scores out of 15 were 12, 12, 0. Accuracy was 92.75%–93.10%, FID 61.47–62.00, recall 31.08%–32.08%, and memorization 4.30%–5.10%. [claim:C059] [evidence:E244,E245,E246,E247,E248,E249]

## v007 decoder feature reweighting with CFG 3.10: C060

Development scores out of 15 were 12, 12, 12. Accuracy was 92.65%–93.80%, FID 62.17–62.75, recall 31.31%–33.96%, and memorization 4.25%–4.95%. [claim:C060] [evidence:E250,E251,E252,E253,E254,E255]

## v009 16×16 encoder attention: C061

Development scores out of 15 were 12. Accuracy was 91.90%–91.90%, FID 67.86–67.86, recall 30.31%–30.31%, and memorization 2.25%–2.25%. [claim:C061] [evidence:E256,E257,E258]

## v007 FreeU with 100 DDIM steps: C062

Development scores out of 15 were 0. Accuracy was 92.85%–92.85%, FID 62.63–62.63, recall 34.85%–34.85%, and memorization 5.35%–5.35%. [claim:C062] [evidence:E259,E260,E261]

## v009 attention with FreeU decoder reweighting: C063

Development scores out of 15 were 12. Accuracy was 93.10%–93.10%, FID 64.25–64.25, recall 31.08%–31.08%, and memorization 3.15%–3.15%. [claim:C063] [evidence:E262,E263,E264]

## v009 attention with stronger FreeU reweighting: C064

Development scores out of 15 were 12. Accuracy was 93.10%–93.10%, FID 62.52–62.52, recall 31.19%–31.19%, and memorization 2.90%–2.90%. [claim:C064] [evidence:E265,E266,E267]

## v009 attention continuation to 34,000 updates: C065

Development scores out of 15 were 12. Accuracy was 93.55%–93.55%, FID 60.67–60.67, recall 34.51%–34.51%, and memorization 4.25%–4.25%. [claim:C065] [evidence:E268,E269,E270]

## v009 attention continuation to 35,000 updates: C066

Development scores out of 15 were 12. Accuracy was 94.00%–94.00%, FID 60.51–60.51, recall 33.96%–33.96%, and memorization 4.70%–4.70%. [claim:C066] [evidence:E271,E272,E273]

## 34k sampler CFG 3.10: C067

Development scores out of 15 were 12. Accuracy was 93.85%–93.85%, FID 61.58–61.58, recall 33.30%–33.30%, and memorization 4.40%–4.40%. [claim:C067] [evidence:E274,E275,E276]

## 34k sampler PAG 0.10: C068

Development scores out of 15 were 12. Accuracy was 93.65%–93.65%, FID 60.90–60.90, recall 32.19%–32.19%, and memorization 4.30%–4.30%. [claim:C068] [evidence:E277,E278,E279]

## 34k sampler FreeU 1.80: C069

Development scores out of 15 were 12. Accuracy was 93.95%–93.95%, FID 60.88–60.88, recall 33.08%–33.08%, and memorization 4.35%–4.35%. [claim:C069] [evidence:E280,E281,E282]

## 34k sampler FreeU 2.20: C070

Development scores out of 15 were 12. Accuracy was 93.65%–93.65%, FID 60.52–60.52, recall 33.96%–33.96%, and memorization 4.25%–4.25%. [claim:C070] [evidence:E283,E284,E285]

## 34k late learning-rate reduction: C071

Development scores out of 15 were 12. Accuracy was 93.80%–93.80%, FID 60.42–60.42, recall 34.29%–34.29%, and memorization 4.50%–4.50%. [claim:C071] [evidence:E286,E287,E288]

## 35k late learning-rate reduction: C072

Development scores out of 15 were 0. Accuracy was 93.80%–93.80%, FID 60.09–60.09, recall 32.96%–32.96%, and memorization 5.05%–5.05%. [claim:C072] [evidence:E289,E290,E291]

## 35k lower-rate fit with FreeU 2.20: C073

Development scores out of 15 were 12. Accuracy was 93.80%–93.80%, FID 60.00–60.00, recall 32.30%–32.30%, and memorization 4.85%–4.85%. [claim:C073] [evidence:E292,E293,E294]

## Three-seed 34k attention stability: C074

Scores were 12,12,12 out of 15. Accuracy was 92.50%–93.55%, FID 60.67–61.16, recall 31.08%–34.51%, and memorization 3.85%–4.55%. [claim:C074] [evidence:E295,E296,E297,E298,E299,E300]

## 34k development reselection: C075

After review removed the added worst-recall veto, the original minimum-tier and normalized-margin rule selected 34k attention. The recorded worst margins are -1.1161 for 34k attention and -1.2751 for the previous FreeU candidate. Reserved confirmation and student checks remain pending. [claim:C075] [evidence:E301]

## Evidence still required

- Planned equal-budget ablations with exact checkpoint, manifest and code lineage.
- Full official five-seed metric table: accuracy, FID, KID and variability, precision, recall, memorization; parameter count, hardware, measured training/sampling time.
- Personal official 200-row audit with Wilson intervals, concrete errors, and class denominators.
- Predeclared sample grid and nearest-neighbor diagnostics, with no selected best images.
- Three authentic student independent checks, actual failures/agreements and resulting rules.
- Actual student name/ID, source verification, final page count, figure selection, and rehearsal.

The completed placeholder evaluator smoke only establishes environment operation. Its metrics must never enter the project result table.

Scientific Agent Skills materially informed evidence handling and visualization. Its verified bibliographic record is retained as E107 and in references.bib; human confirmation remains pending.

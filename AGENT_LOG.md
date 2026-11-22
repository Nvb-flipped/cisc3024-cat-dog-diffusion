# Independent student checks

Three independent checks of the agent's data, results and interface work, each written when it was done (context, my own check, outcome, evidence).

## Check 1 – Data (20 Nov 2026, 23:28–23:56 GMT+8)

**Context.** I asked the agent to find all valid cat/dog head crops in the rest of the pool and add them to the training set. It reported that the 6,100 AI-found crops in the final manifest had passed its visual review against the handout's data specification, and its own spot check of 96 crops found only about 3% problems.

**Check.** I tested this claim myself by labelling 200 random crops from the final manifest against the handout specification, without seeing any of the agent's verdicts. The crops were drawn by the official `audit.py` from my student ID, so the agent could not choose which ones I saw. I judged them from the sheets that the tool generates (each crop at 96×96 and at the 32×32 size the model sees) and wrote a reason for every bad one.

**Outcome.** Only 160 of the 200 crops (80.0%, 95% interval 73.9–85.0%) were good, much lower than the agent's estimate. 33 of the 40 bad crops were "ears cut", which the agent's review had missed; the other 7 were head under 60% of the side (4), head or ears covered (2) and head cut off (1). I added a rule to AGENTS.md (line 20) so that spot checks view crops at a larger size and test every handout criterion, and the data card now reports the 80% and these error types.

**Evidence.** `audit/audit.csv` and `results/audit/breakdown.json`. Two cells in my first CSV had typing slips (a label cell and a verdict/reason cell); they were corrected after I confirmed the intended values, and my original file is kept in `private_build/audit_backup/`.

## Check 2 – Results (21 Nov 2026, 00:38–00:57 GMT+8)

**Context.** The agent reported that the final model (`submission/`, EMA weights after 80,000 updates) scored 15/15 on every official seed it had tried (60 out of 60) with 0% memorization. It also described the sample grids as having only about 6–8 blurry or distorted dog tiles out of 64.

**Check.** I reran the unchanged official evaluator (`starter_kit/evaluate.py`: 1,000 images per class, batch size 500, grading mode) myself on `submission/`. I ran it directly, not through the agent's own evaluation wrapper. Before running, I recorded the SHA-256 of `weights.pt` and `manifest.csv` and confirmed they are the hashes the agent had reported (`e26f85b0…` and `693b3270…`). I ran two seeds: seed 0, to compare with the agent's file `results/final/seed_0.json`, and seed 31415, a seed I chose myself that the agent had never used. For each run I read the evaluator's own summary and its JSON (marks, violations, manifest rebuild, determinism and batch checks). I also inspected the saved sample grid (16 columns × 8 rows: the top four rows are cats, the bottom four rows are dogs) and noted the tiles that I judged to have failed. A tile failed if I could hardly or not at all tell whether it shows a cat, a dog or even an animal, because of distortion or misplaced head components.

**Outcome.**

- **Seed 0 matches the agent's result.** Accuracy 0.9720, precision 0.7965, recall 0.4192 and memorization 0.00% are identical to the agent's file. FID (46.5682) and KID (0.0073) differ from it only at the 7th digit or beyond, which is floating-point noise. The run gave 15/15 marks and no violations. The evaluator rebuilt all 8,029/8,029 manifest rows, reported deterministic=True and seed_sensitive=True, single/mixed batch accuracy 0.95/0.96, 5,098,915 parameters and 19.75 MB.
- **My own seed 31415 did not give 15/15.** Accuracy 0.9745, FID 46.40, KID 0.0068, precision 0.7754 and memorization 0.00% are all fine, but recall is 0.3949, just below the 0.40 threshold. Recall therefore earns 4 marks instead of 5, and the total is **14/15** (no violations). So the agent's claim that every seed scores 15/15 is not true in general; the correct statement is 15/15 on 60 of 61 seeds. My seed is about 2 standard deviations below the agent's mean recall of 0.433 (SD 0.018), which is unlucky but possible; the agent had itself warned that recall varies by about ±0.02 between seeds. The agent reran seed 31415 in its own container and got identical numbers.
- **Sample grids.** The tiles I judged as failed (image numbers per row, left to right):

  | Row | Seed 0 | Seed 31415 |
  |---|---|---|
  | 1 (cats) | 1, 9, 10 | 2, 6, 9, 10, 12 |
  | 2 (cats) | 7, 13 | 1, 5, 6, 10 |
  | 3 (cats) | 13, 15 | 1, 2, 11, 15, 16 |
  | 4 (cats) | 4, 6, 7 | 3, 4, 7, 8, 11, 13, 16 |
  | 5 (dogs) | 4, 5, 7, 14, 15 | 1, 2, 4, 6, 9, 11, 12, 14, 15, 16 |
  | 6 (dogs) | 1, 2, 7, 8, 9, 11, 12, 14, 15, 16 | 1, 5, 8, 9, 10, 12, 14, 15, 16 |
  | 7 (dogs) | 1, 2, 4, 11, 12, 14, 15 | 1, 7, 8, 9, 10, 11, 12, 13, 15 |
  | 8 (dogs) | 2, 3, 7, 8, 9, 11, 16 | 1, 2, 5, 6, 8, 9, 10, 12, 13, 15, 16 |
  | **Total** | **39 of 128** (cats 10/64, dogs 29/64) | **60 of 128** (cats 21/64, dogs 39/64) |

  By this criterion, about a third of the tiles in seed 0 and nearly half in seed 31415 are unrecognizable. That is much worse than the agent's description of about 6–8 dog tiles out of 64, and dogs fail more often than cats in both seeds. The evaluator's numbers do not show this: its classifier accuracy is 97% and its precision is about 0.78–0.80 (roughly a fifth of the samples outside the real-image distribution), so the official metrics under-count this kind of failure. The tiles I did not mark are still not fully sharp, but I can tell cat from dog in them.
- **What changed.** I gave the agent my evidence. It reproduced the seed-31415 result, recorded it (decision log entry 73), and corrected its claim to 60 of 61 seeds at 15. I added two rules to AGENTS.md (lines 21 and 22). The first is about robustness claims: write "X out of Y seeds" instead of "all seeds", report the spread of key metrics across seeds, and flag any metric that can fall below its grading threshold (such as recall below 0.40) as a known risk. The second is about visual quality: official metrics do not replace looking at the grids, so I count the exact number of unrecognizable tiles and report cat and dog failure rates separately. The report must state the 60-of-61 result and its risk, and describe the sample quality honestly instead of "6–8 distorted dogs".

**Evidence.** `student_checks/evidence/`: `input_hashes.txt`, `my_seed0.json`, `my_seed0.log`, `my_seed0.png`, `my_seed31415.json` and `my_seed31415.png`. `my_seed31415.log` now holds a CUDA out-of-memory error from a later retry (the GPU is shared), which overwrote the log of the successful 00:45 run; that run's JSON and grid are intact. `my_seed31643.log` is an empty file from a mistyped seed number.

## Check 3 – Model/interface (22 Nov 2026, 20:28–20:37 GMT+8)

**Context.** The agent claimed that `submission/` follows the course interface: the same labels and seed always give the same images, a different seed gives different images, each image depends only on its own label and the seed (not on the other labels or the batch size), the output is `[N,3,32,32]` in [0,1] without NaN, the caller's random state is preserved, and the noising and recovery functions in `submission/diffusion.py` follow the standard DDPM arithmetic.

**Check.** I tested this on the final files, whose SHA-256 I recorded first (`interface_input_hashes.txt`): `weights.pt` e26f85b0b7ff…23d089f and `manifest.csv` 693b32705b7b…2af930 on the GPU (torch 2.9.1, CUDA).
- *Interface probes.* I wrote the first version of six probes and the agent suggested refinements (a control for each "must differ" case, larger batches, more random states). Each prints the maximum absolute difference, not just pass or fail: (1) the same labels and seed twice; (2) seed 5 against seed 6, per image; (3) image 0 when the other labels change (`[0,0,0,0]` against `[0,1,1,1]`), plus a control that a different label at slot 1 does change the image; (4) a single label, and the first four images of batches of 40 and 65 (larger than the internal batch of 32), against the same four labels alone; (5) shape, dtype, range and NaN/Inf; (6) the random state of torch (CPU and CUDA), Python and NumPy before and after generation.
- *Hand calculation.* I worked one example by hand, using a calculator for the square roots and decimals, clean value 0.4, cumulative alpha 0.81, noise −0.2, using $x_t=\sqrt{\bar\alpha}\,x_0+\sqrt{1-\bar\alpha}\,\epsilon$ and $\hat x_0=(x_t-\sqrt{1-\bar\alpha}\,\epsilon)/\sqrt{\bar\alpha}$: the noisy value is 0.9·0.4 + 0.43589·(−0.2) = 0.27282 and the recovered value is (0.27282 + 0.08718)/0.9 = 0.4. I then fed the same numbers to `add_noise` and `reconstruct_clean` in `submission/diffusion.py`. The code returned 0.2728220211291866 for the noisy value (my hand value 0.2728220211291865, difference 5.6e-17, floating-point rounding) and 0.4 for the recovered value (difference 0), and my script printed MATCH.

**Outcome.** All six probes passed, and every comparison was exactly identical (difference 0), not merely small: determinism, image 0 with other labels changed, and one label against a batch of 4, 40 and 65. The known GPU rounding problem from AGENTS.md (differences near 1e-5 when the physical batch changes) did not appear. The controls show that the probes can detect a change: a different seed changed every image (mean difference 0.20 to 0.42 per image) and a different label changed the image (0.13). The output was float32, finite, within [0, 1] and of the right shape, and none of the four random states changed. The hand calculation recovered the original value exactly. I found no mistake, so the agent's interface claims held and I changed no code or tests. Limits: the probes use four images and one pair of seeds, so they support but do not prove the claims; the official evaluator's own checks agree (deterministic, seed-sensitive, single-class accuracy 0.95 against 0.96 mixed).

**Evidence.** `student_checks/evidence/`: `interface_input_hashes.txt`, `my_interface_probe.py`, `my_interface_probe.log`, `hand_calculation.md`, `my_hand_check.py` and `my_hand_check.log`.

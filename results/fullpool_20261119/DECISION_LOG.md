# Full-pool curation and model decision log (agent decisions under Paco's delegation)

Times UTC, 19 Nov 2026. Each entry: decision, evidence, what would change it.

1. **12:05 Geometry rescue within the specification.** DINO proposals rejected only as
   `square_outside_source` are re-squared at occupancy 0.70–0.80 with the head centre within 8%
   of the square centre, angle 0, no padding. Evidence: 23,499 proposals become valid (15,240 merged
   heads; 9,823 previously negative sources). Changes if audits show rescued crops are off-centre.
2. **12:10 Rescue detector = targeted Grounding DINO, not YOLO-World.** Internet ~1–2 MB/s and the
   Ubuntu mirror unreachable; DINO is already local and pinned. Changes if DINO rescue yields few heads.
3. **12:15 CLIP source flag at P(real cat/dog) ≥ 0.10** for DINO rescue: 13,943 of 39,502
   candidate-less sources; recall 99.75% on sources with a personally confirmed head (optimistic,
   shortlist selection used related CLIP scores). The 25,559 unflagged sources need a sampled miss audit.
4. **13:05 Qwen3.5-4B served in BF16** (the other user's 24 GB job ended), 45% VRAM cap.
5. **13:20 Prompt v2 primary pass started** (calibration: precision 96.1% with geometry gate,
   recall 90.6% on personal verdicts).
6. **13:40 v2 alone rejected as a sole authority.** Real-queue agent audits: 36/48 and 42/64 (blind)
   accepted crops valid. v4 looked like 97% on the sheet it was designed on but 74% on the blind sheet
   (overfitting caught by the blind check). CLIP crop gate kept 22/77 good crops: rejected.
7. **14:00 Detector-derived size hint removed.** 4B answered "60–80%" for 9,496/9,525 candidates,
   copying `implied_head_fraction_of_side`. All v2-v4 ledgers are treated as unverified pre-screens;
   prompt v5 (no box hints) is the escalation prompt.
8. **14:00 Mirroring adopted for generator training.** Matched 48k: FID 58.00→52.22, recall
   0.319→0.363, memorization 1.65%→0.00%. The evaluator counts mirrors as copies.
9. **14:25 Flip continuation to 160k started** from the verified 80k checkpoint, because 64k reached
   FID 50.68 / recall 0.393 with 0% memorization; whether it is used depends on screens (§ Track A).
10. **Rule for AI-curated rows (engineering judgement, not a hard threshold):** no VLM-accepted crop
    enters a training manifest unless a fresh, representative, blind sample (accepted, rejected and
    uncertain decisions) shows high accepted precision with acceptable diversity, and there is time
    to retrain and evaluate. Otherwise the submission uses v010 (+ mirroring) and the curation is
    reported as a measured result.
11. **14:40 Blind representative sample of 200 queue candidates** (agent-labelled before any VLM
    run; `private_build/fullpool_qa/blind200/`): 19 valid (9.5%, Wilson 6.2–14.4%); dino_valid 6.6%,
    geometry rescue 15.6% (the rescue is the richer stratum). At this base rate a filter needs ~99.5%
    specificity for ~95% precision.
12. **14:45 Qwen3.5-9B BF16 + prompt v5 (no hints):** with the prompt's own rule it rejected all 200
    (it rates every close-up head "over 80%"). Ignoring the VLM size band (geometry already fixes
    occupancy at 0.70–0.80 of the detector box): 12/19 valid kept, 4/181 invalid passed (precision
    ~75%, specificity 97.8%). Errors are framing (lying/whole body/small head). This rule was chosen
    after seeing the sample, so it is not yet validated. Not sufficient for training rows.
13. **14:30 Flip run plateau:** FID 50.68/50.54/50.96 and recall 0.393/0.389/0.361 at 64k/72k/80k
    (0% memorization). Priority moves to sampler guidance (sweep 2.5/2.75/3.0 on 64k and 72k).
14. **15:20 Paco's instruction: full-pool curation continues regardless of generator score.** Even
    at 15/15 the project must add more valid images from the whole pool. Quality gates decide which
    images are added, never whether curation proceeds. Current pipeline: 9B v5 over all 63,115 head
    candidates (47,605 DINO + 15,510 rescue), then head-fraction re-measurement, then an accepted-
    sample precision audit, then v011 training on the expanded manifest compared with v010.
15. **15:15 Head-fraction gate (DINO re-measured inside the square, ≥ 0.60):** all 19 valid blind
    crops measure 0.61–0.93; combined with the 9B rule it cut false positives 4→2 (12/14 on the blind
    sample, tuned there; must be confirmed on an accepted-sample audit).
16. **15:10 Wide 10.7M model not preferred:** precision 0.80–0.82 but FID 51.8–52.3, recall
    0.29–0.35 and memorization 1.0%→3.4% rising (48k–64k).
17. **15:05 DINO rescue complete:** 13,943 sources, 20,512 geometry-valid proposals in 9,262
    previously negative sources, merged into 15,510 rescue head candidates.
18. **17:45 The 160k continuation is not used.** Seed 0, v009 sampler: 96k FID 50.90 / recall
    0.363; 112k 52.31 / 0.362; 128k 52.42 / 0.357; 144k 52.81 / 0.360; 160k 53.13 / 0.334
    (memorization ≤ 0.05%). Longer training degrades FID and recall; the 64k checkpoint stays the
    best v010 candidate (seeds 0/101/202 at guidance 3.0: 14/13/13; at 3.25: 13/13/13).
19. **17:45 9B screen of the 15,510 rescue candidates done:** 2,337 PASS (1,479 rescue-valid, 858
    geometry rescues). Head-fraction re-measurement of these passes running on GPU.
20. **18:15 9B v5 screen of all 63,115 head candidates complete:** main queue 4,381/47,605 PASS
    (2,638 original geometry, 1,743 geometry rescue); rescue queue 2,337/15,510 PASS. Rescue head
    fractions ≥ 0.60: 1,916/2,337. Main-queue head measurement running; then a random 150-crop
    accepted-sample audit decides whether the expanded manifest is trained.
21. **19:00 Accepted-sample audit (random 151 of 5,275 v5-PASS & head ≥ 0.60):** 111 valid
    (73.5%, Wilson 66–80%). Errors: framing (small head/whole body, ears cut, looking up/down),
    second animal/person, graphics. Head fraction did not separate further.
22. **19:10 9B framing pass v6 on the 5,290 accepted:** only `head_size` is informative (ears and
    other-animal fields were noise). Rule `head_size=most_of_image & gaze=camera_or_side` (chosen
    on the 151): 3,805 eligible. Fresh validation sample of 100: 79 valid (Wilson ~70–86%);
    failures: ears cut by close face boxes (~8), added graphics/frames/text (~9), second animal (3).
23. **19:20 Ear-margin re-squaring (box/side 0.65) rejected:** visually no help — the detector box
    often covers only the face; a margin large enough for ears would push heads whose box already
    includes ears below 60%. Not used.
24. **19:30 Graphics/second-subject pass v7 gives little:** removes 2 validation false positives;
    not used as a gate.
25. **19:45 v011 frozen (AI-assisted, origin ai_assisted_reviewed_manifest):** v010's 1,929 personal
    rows + 3,607 AI rows (9B v5 PASS, head fraction ≥ 0.60, v6 framing rule) = 5,536 training rows
    (2,375 cat, 3,161 dog), byte-identical 63-row holdout, cache parity exact. A first freeze had 55
    same-photo square overlaps (detector-box check); it was never used, is preserved under
    `private_build/superseded_manifests/`, and the check now compares final squares. Estimated AI-row
    precision ~80% (validation sample), so roughly 700 imperfect rows (mostly ear tips cut on close
    faces, small graphics). Recorded trainer amendment: `results/plans/v010_input_amendments.json`.
26. **19:50 v011 training** with the exact v010+mirroring config, 120k updates; screens chained.
    In parallel: agent visual review of all AI rows to build a cleaner v012 for comparison.
27. **Correction (18:45 UTC, read from the system clock):** the times written in entries 21–26 were
    estimates and are too late by about an hour; those steps actually happened between ~18:15 and
    18:45 UTC. v011 training started at 18:41 UTC (supervisor log).
28. **20:00 v011 (5,536 rows incl. 3,607 AI rows) official seed 0, guidance 3.0:** 48k 13/15 (FID
    53.97, recall 0.375, acc 0.971); **64k 15/15 (FID 49.85, recall 0.405, accuracy 0.981, precision
    0.802, memorization 0.0%)**. Agent visual check of the 64k grid: dogs clearly better than v010
    (~8/64 distorted vs ~25/64), cats similar. Seeds 101/202 queued before any selection.
29. **20:00 Agent visual review of all v011 AI rows in progress:** sheets 0–9 (1,000 rows): 228
    rejected (22.8%) — consistent with the ~80% validation estimate. Ledger:
    `private_build/fullpool_qa/review_v011_ai/agent_review.jsonl` (agent_visual_review, not personal).
30. **20:45 v011 checkpoint screens (seed 0, guidance 3.0):** 80k 15/15 (FID 48.78, recall 0.405),
    96k 14 (49.03, 0.365), 112k 14 (49.24, 0.397), 120k 15 (49.13, 0.419). 64k seeds 0/101/202:
    15/13/15 (recall 0.405/0.367/0.403). FID now < 50 from 80k; recall is the margin metric. Next:
    seeds 101/202 for 80k and 120k, and guidance 2.75 at both (accuracy headroom ~8 points).
31. **20:55 v011 multi-seed (seeds 0/101/202):** 80k g3.0 15/14/14; 120k g3.0 15/14/14 (recall falls
    with longer training); 80k g2.75 15/14/15 (FID 48.5/48.8/48.1, recall 0.409/0.392/0.402);
    80k g2.5 14/14/15 (recall 0.397/0.384/0.412). Mean recall barely depends on guidance in 2.5–3.0;
    FID has ~1–2 points margin; recall sits at ~0.40. Stochastic DDIM rejected (incompatible with
    class-noise annealing; earlier v002 eta runs lowered recall). Testing limited-interval guidance
    (Kynkäänniemi et al. 2024): t ≤ 850 or ≤ 700 at guidance 3.0, and ≤ 700 at 3.5.
32. **21:05 Limited-interval guidance rejected:** at 80k seed 0, t≤850/g3.0 FID 50.19 rec 0.406;
    t≤700/g3.0 50.26/0.400; t≤700/g3.5 50.52/0.420 — recall gains only by losing FID/precision.
33. **21:15 Agent visual review of all 3,607 v011 AI rows complete:** 2,681 kept (74.3%), 926
    rejected: lying/sleeping 243, text/graphics/frames 208, dark/blurry/faded 148, pose/cut/profile
    98, other animal 82, person/hand 73, small/whole body 52, repost duplicates 22.
34. **21:20 v012 frozen and training:** v010's 1,929 personal rows + 2,681 agent-approved AI rows =
    4,610 (1,962 cat, 2,648 dog), same holdout, parity exact; identical recipe to v011; screens at
    guidance 2.75 chained. Compares quantity (v011, ~74% precise AI rows) with quality (v012).
35. **21:50 Missed-head audit of unflagged detector-negative sources** (CLIP P<0.10, 25,559 sources,
    no rescue; random 200, agent-viewed thumbnails, `private_build/fullpool_qa/miss_audit_unflagged/`):
    2/200 contain a real cat/dog (a small dog in a sweater, a tiny kitten in a tunnel), neither a
    clear valid head crop. Valid-head miss rate ≈ 1% (Wilson 0.3–3.6%); ≤ ~900 sources at the upper
    bound, mostly low quality. Reported as a limitation, not re-processed.
36. **22:45 Agent error, fixed:** a cosmetic docstring edit to `scripts/run_v012.py` after its plan was
    frozen changed the file hash, so the chained v012 screens failed all 10 retries (22:10–22:24) and
    were only noticed at the next hourly check. The original bytes were restored (hash verified against
    the frozen plan) and the screens relaunched. Lesson: never edit a plan-hashed driver after
    freezing; record an amendment or create a new driver instead; also check chained jobs soon after
    their trigger time.
37. **23:45 v012 seed-0 screens (g2.75):** 64k 14 (FID 48.54, rec 0.382), 80k 14 (48.14, 0.393),
    96k 14 (48.21, 0.372), 120k 14 (48.41, 0.371). Versus v011 80k/120k (15/15, rec 0.409/0.418):
    cleaner-but-smaller data gives similar/better FID and precision but lower recall on seed 0.
    Seeds 101/202 for v012 80k running before any conclusion.
38. **00:00 (20 Nov) v012 80k g2.75 seeds 0/101/202:** 14/14/15, recall 0.393/0.347/0.402 (mean
    0.381), FID mean 48.0, precision ~0.84. v011 same setting: 15/14/15, recall mean 0.401, FID 48.5.
    Cleaner-but-smaller data improves FID/precision slightly but recall (the binding metric) falls;
    differences are within ~1–2 seed SDs. Direction: grow *correct* data (agent review of the 1,364
    v6-framing-rejected crops: ~25% valid so far) and test config-only levers on v011 data:
    wide 10.7M (`v011_wide_flip`) and EMA 0.9995 (`v011_ema9995_flip`), 100k each, screens chained.
39. **01:00 Agent review of the 1,364 v6-framing-rejected crops:** 351 valid (25.7%) recovered
    (`results/fullpool_20261119/agent_review/v6_rejected_agent_review.jsonl`).
40. **01:05 v013 frozen and training:** v012 rows + recovered crops = 4,922 training rows (2,120 cat,
    2,802 dog; 39 recovered crops removed as same-photo/personal duplicates), same holdout, parity
    exact; identical recipe; screens at g2.75 chained. All AI rows in v013 passed agent visual review.
41. **01:50 EMA 0.9995 on v011 data (g2.75, seed 0):** 64k 15/15 FID 48.36 recall 0.440; 80k 15/15
    FID 47.86 recall 0.435 (vs EMA 0.9999 80k: 48.52 / 0.409). Seeds 101/202 queued. EMA does not
    change optimization, so v013 is also being trained with EMA 0.9995 (`v013_ema9995_flip`, 100k).
42. **02:10 Paco: the pool holds ~9,000–10,000 valid images; find and use them all.** Estimates agree
    (~8.5k at agent strictness). Recovery plan: (a) 9B prompt v8 (recall-oriented "usable head shot?",
    watermarks/minor flaws tolerated, lying body OK if head upright) over the 29,131 9B-rejected
    candidates that 9B called cat/dog (blind sample: ~6–8% valid there), then head measure and agent
    review of v8 passes; (b) agent re-review of its own lying/resting rejections under the exact
    spec (head roughly upright): 93/246 recovered — the earlier rule was stricter than the handout.
43. **02:10 Config levers plateau:** EMA 0.9995 80k seeds 0/101/202 15/14/15, recall 0.435/0.384/0.405
    (mean 0.408 vs 0.401 for EMA 0.9999); 64k 0.440, 100k 0.387 (seed 0). Wide 64k seed 0: 14, recall
    0.399, precision 0.847. Recall is ~0.40 ± 0.02 across settings on v011 data; the remaining lever
    is more valid data (target 9–10k images).
44. **02:45 Wide 10.7M on v011 (seed 0, g2.75):** 64k/80k/100k = 14/14/14; FID 48.75/48.66/48.88;
    recall 0.399/0.385/0.383; precision 0.847/0.863/0.878. Sharper samples, lower recall; not adopted.
45. **02:50 v8 is too permissive as a judge:** 12,669/17,684 PASS (72%) in a pool that is ~6–8% valid.
    Used only as a coarse filter: v8 PASS ∧ head fraction ≥ 0.60, then a random-100 agent yield audit
    decides whether to eye-review the whole filtered set (expected ~80 sheets) or build a sharper filter.
46. **03:50 v8 complete:** 22,043/29,131 PASS. Yield audit of a random 100 from v8 PASS ∧ head ≥ 0.60
    (`private_build/fullpool_qa/v8_yield100/`): 7 valid (v5 near-miss-real bucket 5/52, v5 not-real
    2/48). The remaining ~2,000 valid heads are spread at ~7% density over ~29k mostly-junk crops.
47. **03:55 Ranking for review, not acceptance:** linear probe on local CLIP ViT-B/32 crop embeddings
    (`src/curation/rank_classifier.py`) trained on recorded pool-crop spec judgements (personal
    verdicts, agent reviews, audits); unreviewed candidates are reviewed best-first by the agent and
    review stops when sheet yield becomes low. Pool images only; no dev data; evaluator unused.
48. **04:15 Ranked review works:** CLIP linear probe (5-fold AUC ~0.85 on hard labelled crops) puts
    valid heads first: ranked sheets 000/001/002 had 83/55/32 valid as cropped. Many remaining "ears"
    rejections are face-only detector boxes; `src/curation/recrop.py` re-squares them (box/side 0.50,
    else 0.55/0.60; offset ≤ 8%, no padding) and the re-crops are reviewed: 87/101 valid. Ranked
    review continues best-first until sheet yield (incl. re-crops) falls below ~30%.
49. **04:20 v013 seed-0 screens (g2.75):** 64k/80k/96k/120k = 14 each, recall 0.330/0.353/0.365/0.353,
    FID 49.99/49.02/49.01/49.42. Recall order v011 (5,536 rows, ~75% strict-valid AI rows) > v012 >
    v013 (strictly reviewed) — the agent's review was stricter than the handout specification.
50. **04:20 Review rule changed to the handout specification exactly:** reject only non-photos/prints,
    wrong species, a second head/face (animal or person), head too small/cut off/not upright, heavy
    graphics/frames/collages, duplicates. Lying bodies with upright heads, dark-but-visible heads,
    profiles, watermarks, small text and hands are acceptable. Earlier marginal rejections will be
    re-reviewed under this rule; the next manifest maximises spec-valid images (Paco's 9–10k target).
51. **05:10 Missed chain:** v013 EMA 0.9995 finished at 04:01 without chained screens (agent omission);
    screens (64k/80k/100k, g2.75) launched at 05:10. Nine parallel forked agent reviewers (decision 50
    rubric, separate ledgers) are reviewing ranked sheets 008–097 and 32 re-review sheets; reviewed
    re-crops so far 253/273 valid. `src/curation/freeze_reviewed.py` (acceptance = agent visual review)
    dry-run on available ledgers: 1,929 personal + 3,554 AI rows = 5,483 before the forks' results.
52. **06:05 v013 EMA 0.9995 (seed 0, g2.75):** 64k/80k/100k = 14 each; FID 48.70/48.62/48.78; recall
    0.388/0.383/0.365. The strictly reviewed v013 data stays below v011 in recall regardless of EMA,
    supporting decision 50 (maximise spec-valid data). Re-review forks 7 and 9 done: 468/1,100 and
    334/921 valid under the exact spec.
53. **06:50 All review queues closed:** ranked sheets 000–097, re-review sheets, and re-crop sheets
    000–017 (1,485 re-crops, 1,116 valid) are reviewed. The 249 remaining "too tight" rejections
    cannot be re-squared inside the image (recrop.resquare returns None), so re-crop coverage is
    complete. Latest-verdict count: 7,339 AI-found good + 1,929 personal = 9,268 before dedup.
    First v014 freeze: 8,295 rows (1,929 personal + 6,366 AI; 3,202 cat / 5,093 dog; 781 same-photo,
    181 duplicate-of-personal, 22 family duplicates, 5 holdout-source exclusions).
54. **07:00 Label spot-check found a one-head gap; v014 set aside before any use:** CLIP zero-shot on
    stored crop embeddings disagreed with 11/6,366 VLM species labels (none strong); all 11 VLM labels
    are visually correct, but 6 of the 11 crops show part of a second animal head at the edge
    (exactly-one-head violation). A detector-overlap scan (another detected head box ≥25% inside the
    square) flagged 1,160 of the 6,366 AI rows; a random 48 had ~6 real violations. All 1,160 are
    being re-checked for the one-head rule by three forked reviewers (25 sheets of 48,
    `private_build/fullpool_qa/rereview_second_head/`); two misses from the label sheet are recorded
    in `label_check/agent_review.jsonl`. The unused first v014 files moved to
    `private_build/superseded_manifests/v014_pre_second_head/`; v014 will be refrozen with these
    ledgers last. Drivers `scripts/run_v014.py` (160k) and `scripts/run_v014_variants.py`
    (EMA 0.9995, 160k) are sed copies of the v013 drivers.
55. **07:35 One-head re-check result and final v014:** the three re-check reviewers rejected 295 of
    the 1,160 flagged rows (second animal head 92, ears cut 80, person 36, screen/non-photo 21,
    small head 21, not upright 20, other 25). The 80 ears-cut rejections were confirmed on a
    300-px sample (face-only squares the earlier small-tile reviews missed; the overlap scan caught
    them because the full-head box was a separate detection). Re-crop batch 6 re-squared 63 of them
    (15 were already re-crops, 2 do not fit); agent review kept 28. A random 96 of the 5,206
    unflagged rows had 0 second heads and 3 non-photos (painting, figurine, mug print; recorded) —
    residual non-photo rate ~3% (Wilson 95% 1–9%), accepted and left to the personal audit.
    **v014 = 8,029 rows** (1,929 personal + 6,100 AI; 3,096 cat / 4,933 dog; byte-identical 63-row
    holdout), sha256 693b3270…; cache parity exact (8,092 crops). Species labels: CLIP cross-check
    disagreed on 11/6,366, all VLM labels visually correct.
56. **07:40 v014 trainings launched (restricted containers, 1k checkpoints, chained seed-0 g2.75
    screens at 80k/100k/120k/140k/160k):** `v014_flip` (EMA 0.9999) and `v014_ema9995_flip`, both
    160k updates, seed 17. Selection rule unchanged: hard checks first, then tier total, then
    recall margin; best candidates go to seeds 101/202, then confirmation 303/404, with grid
    inspection for head realism/distortion before any selection.

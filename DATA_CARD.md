# Data card

**Specification.** 32×32 RGB real photo; exactly one centred, roughly upright cat/dog head, ears included, about 60–80% of the square's side; head at least 32 px wide in the pool image; label cat = 0, dog = 1. **Final training set:** `submission/manifest.csv` (v014), 8,029 exact crops (3,096 cat, 4,933 dog), no rotation or padding, plus a separate 63-crop pool-only holdout. Targets come only from the course pool (no dev images or features). The generator trains from scratch; its only augmentation is left–right mirroring.

**Pipeline (the review also covered crops the screen rejected, so the steps are not a strict funnel).**

| Step | Count |
|---|---:|
| Raw pool | 76,250 images |
| Candidate head crops: Grounding DINO Tiny squares, geometry rescue (60–80% rule), targeted rescue of detector misses | 63,115 |
| Qwen3.5-9B screen: crops passed (a looser second pass only ordered the review) | 6,718 |
| Agent visual review of 17,174 crops under the handout rules (CLIP-ranked order) | 7,083 valid |
| Remove same-photo, near-duplicate and holdout-source rows | 6,100 AI-found |
| Add the student's personally reviewed crops (50 batches) | **8,029 final** |

AI-assisted cleaning was used. Agent verdicts are not the student's; the audit below is the student's own.

**Audit precision.** The student personally labelled the 200 crops drawn by `audit.py`: **160 good = 80.0%** (Wilson 95% interval 73.9–85.0%). Cats 63/75 = 84.0% (74.1–90.6%); dogs 97/125 = 77.6% (69.5–84.0%). Student-reviewed rows 43/49 = 87.8% (75.8–94.3%); AI-found rows 117/151 = 77.5% (70.2–83.4%); the intervals overlap, so no reliable difference. Numbers: `results/audit/breakdown.json`.

**Main remaining errors (40 bad).** Ears cut by the square: 33 (82%). Head under 60% of the side: 4. Head or ears covered: 2. Head cut off: 1. No crop was marked as a cartoon, two heads or a wrong label. Ear clipping is worst in the 18 audited re-crops (squares re-made from face-only detector boxes): 7 of 18 had ears cut, versus 22 of 133 other AI-found rows. Likely causes: detector boxes often cover only the face, and the agent judged ear margins on small thumbnails. A missed-head sample of 200 unflagged sources found about 1% real heads (Wilson 0.3–3.6%).

**Next.** (1) Re-review every row at 300 px or larger with an explicit ear-tip margin check (a 300 px sample showed these failures are visible at that size) and re-crop wider or drop failures. (2) Treat re-crops as high risk: re-square to the full head width including ears. (3) Retrain from scratch on the cleaned manifest and redraw a fresh audit to measure the gain. Not done for this submission: it needs re-review of about 8,000 crops, a full retraining and a new 200-row audit.

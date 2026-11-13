# Course requirements and evidence

Source: supplied `project_handout.pdf`, four pages. The handout takes precedence over starter comments. This is a living implementation map, not a claim of full marks. All paths are relative to the project root.

| Requirement | Source | Evidence / implementation | Current state / owner |
|---|---|---|---|
| Individual course-only work; coding agent is the main development tool | pp. 1, 4 | AGENTS.md, Git history, local-only data layout | Engineering in progress; student owns decisions and explanation |
| Preserve official kit | p. 1 | scripts/setup_inputs.py --verify-only; archive hashes in results/setup/source_hashes.json | Verified; evaluator wrapper checks again |
| Real 32×32 RGB photo, one centered upright cat/dog head including ears, 60–80% occupancy, source head width ≥32 px | pp. 1–2 | src/curation/review.py and review.js; crop parity tests; replayable decision log | Personal crop and duplicate decisions saved through Batch 17 and frozen for v005. Independent data check and official audit pending |
| No dev images or individual dev features for training, auxiliary models or pool selection | pp. 1–2, 4 | Pool-only curation inputs; disjoint source holdout; separate unchanged evaluator | Implemented boundary; retain review of future changes |
| Pretrained models only for curation; classifier/Inception forbidden in curation/training/generation | pp. 2, 4 | Pinned Grounding DINO/CLIP provenance; separated evaluator cache | Implemented; pretrained weights excluded from generator |
| Manifest describes every training image using exact official geometry | p. 2 | src/curation/freeze.py; manifests/v005.provenance.json; results/curation/v005_cache_parity.json; checkpoint/export hash binding | v005 has 1,249 valid training rows with no warnings. Both splits match the official reconstruction pixel for pixel. v002 remains the reference model's data |
| Aim for ≥500/class | p. 2 | Training enforces ≥500/class after holdout | v005 contains 606 cat and 643 dog training crops |
| Personal official 200-row audit; every bad row has a reason | p. 2 | src/evaluation/audit.py invokes unchanged audit.py; exact sample/ID/hash checks | Student only; final manifest/ID needed later |
| One-page data card, stage counts, audit precision/uncertainty, errors and next steps | p. 2 | DATA_CARD.md; full proposal summary; Wilson interval utility | Provisional; final audit and page review pending |
| Generator trained from random initialization, exact manifest targets, no pretrained autoencoder | pp. 2, 4 | submission/model.py; src/training/train.py and step.py; results/plans/v004_coverage.json | Fresh matched 30,000-update v002 and v004 noise-target runs verified. v002 remains the selected reference |
| One nn.Module; offline self-contained submission; deterministic private seed and own-label isolation | pp. 2–3 | submission/generate.py and diffusion.py; actual GPU stress evidence; offline checker | Implementation verified diagnostically; final weights need recheck |
| ≤20M parameters and ≤100MB; memorization ≤5% | p. 3 | Internal 18M/90,000,000-byte caps; final scorer/readiness | The v001 reference has 5,082,147 parameters and is 20,450,122 bytes with the current sampler. Final bytes and memorization pending |
| Output [N,3,32,32], finite in [0,1]; no batch-composition dependence | pp. 2–3 | Official tests, project interface tests, fixed physical execution batches | Diagnostic checks passed; final CPU/GPU matrix pending |
| Official accuracy/FID/KID/precision/recall/memorization and full training rebuild | pp. 2–3 | src/evaluation/run.py, sweep.py, select.py; results/experiments/capacity_32_seed17_v002/comparison.json | v002 noise-target reference passed every hard condition on three full development seeds. Final export and confirmation pending |
| Excellent tier: accuracy ≥.90, FID ≤50, recall ≥.40, independently graded | p. 2 | Prespecified five generation seeds and conservative ranking in PROJECT_PLAN.md | v002 noise-target reference scored 11/15 on each seed. Accuracy 89.40–89.95%, FID 71.59–72.80, recall 30.20–32.30% |
| Crop parity, output/range, fixed-seed and single-batch overfit pytest checks | p. 3 | tests/ plus unchanged starter tests; results/setup/*.xml; two-image fixed 2,000-update diagnostic | Tests pass; diagnostic is not generalization evidence |
| One reproduction command per report number; Git commits | p. 3 | README.md, run metadata, code/manifest hashes, segment logs, Git history | Infrastructure implemented; final report commands pending actual experiments |
| Six-page main report with actual methods/equations, full result table/grid, failures and agent verification | p. 3 | report/evidence_outline.md, source_manifest.json, claims.csv, references.bib | Incomplete evidence outline; no invented baseline/results |
| Three contemporaneous independent student checks, at least one on data | p. 3 | AGENT_LOG.md pending instructions | Student only; none claimed complete |
| 4-minute talk/demo + 1–2-minute questions; student understanding | p. 4 | PROJECT_PLAN.md timing; final-checkpoint demo planned | Deck/demo/rehearsal pending |
| Slide ownership rule | p. 4 and explicit student clarification | Student reports instructor's in-person exemption permitting AI generation, with personal revision; recorded in plan | No written exemption claimed; student must own/explain final talk |
| report.pdf with actual name/ID, plus clean ZIP and presentation slides | p. 4 | src/release/package.py and verify_archive.py; explicit allowlist and hash receipts | No final ZIP or upload yet |
| Exclude standalone images, raw pool/data, environments, alternate checkpoints | p. 4 | Archive allowlist, embedded-image checks, clean extraction tests | Software checks implemented; actual final archive still required |
| Deadline Nov 26 2026 23:55 Asia/Shanghai | pp. 1, 4 | PROJECT_PLAN.md targets stable upload Nov 25 18:00 | Future student upload/receipt |

The requested SVG/PNG/PDF/draw.io/XML/native PPTX figure package and Beamer deck are additional agreed deliverables in PROJECT_PLAN.md. F0 has structural parity evidence and a limited local draw.io edit/save/reopen check. Human F1/F2 brief/exemplar/A/B decisions, connected-node edits, and actual Office rendering remain pending.

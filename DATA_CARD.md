# Data card — provisional machine record, not submission-ready

**Task.** 32×32 RGB photographs of one centered, roughly upright cat/dog head, ears included; 60–80% head occupancy, original head width ≥32 px. Labels: cat=0, dog=1. Training targets must be exact manifest crops of the supplied course pool. Data stay local and course-only.

| Stage | Current count | Evidence / limitation |
|---|---:|---|
| Indexed pool | 76,250 IDs, 22 shards | `results/setup/source_hashes.json` |
| Integrity scan | 76,250 decoded; 0 failed/missing/repeated IDs | `results/curation/integrity.json`; CRC, dimensions and full decode |
| Exact RGB duplicate excess | 0 | Does not rule out near-duplicates |
| Deterministic detector pilot | 2,000 sources | Pinned Grounding DINO Tiny; pool-only |
| Nine-setting union | 3,352 proposals | Counts proposals, including overlapping/conflicting candidates |
| Valid square geometry | 1,269 proposals from 697 sources | Geometry is not specification compliance |
| Full-pool proposal precomputation | 125,941 proposals from all 76,250 sources | Zero detector errors; all nine settings retained |
| Full-pool valid square geometry | 47,855 proposals from 26,925 sources | No accepted training heads; pool-only review-priority scores are exploratory |
| Student calibration / duplicate confirmation | Pending | 200 proposed crops + 200 rejected sources selected; no human verdicts recorded by the agent |
| Frozen train / pool-only holdout | Pending | Planned class-stratified, duplicate-family-disjoint 5% holdout |
| Final official audit | Pending | Actual student ID and frozen final manifest required |

**Pipeline.** Delivered-coordinate RGB decoding → frozen detector proposals → student correction/rejection → duplicate-family review → immutable six-column manifest → official reconstruction. No EXIF reorientation, padding, generative editing, external training images, dev-based image selection or evaluation-network curation. Final targets receive no image augmentation. Detector and evaluator weights remain separate from the generator.

**Quality and uncertainty.** Audit precision and its 95% Wilson interval are **not available** until the student labels the exact official 200-row audit. No detector accuracy, final class count or compliance percentage is claimed. The rejected-source sample diagnoses misses; it is not an unweighted estimate of overall pool accuracy.

**Observed pilot limitations.** Agent-inspected diagnostic sheets contain drawings, toys, prints on products, wrong species labels, loose boxes and boundary failures. These observations are exploratory and do not replace student verdicts. Source dHash pairs are candidates for visual/aligned-pixel duplicate confirmation, not automatic duplicate labels.

**Next action.** Complete calibration, record original-versus-corrected decisions and uncertainty, select a defensible setting, and review full-pool candidates before baseline training. Replace this provisional record with final stage counts, per-class audit denominators, remaining error types and one evidence-based improvement. Student independent verification remains pending.

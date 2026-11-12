# Data card — provisional pending the official audit

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
| Full-pool valid square geometry | 47,855 proposals from 26,925 sources | `results/curation/full/summary.json` |
| Student calibration | 200 proposals and 200 rejected sources | 14 proposed-set good verdicts. 23 rejected-set missed heads. `results/curation/calibration.json` |
| Duplicate review and retained crops | 1,260 retained from 1,260 personally approved crops | 614 cats and 646 dogs. `manifests/v002.provenance.json` and `annotations/duplicate_decisions.jsonl` |
| Frozen training crops | 1,197 | 583 cats, 614 dogs. `manifests/v002.csv` |
| Pool-only holdout | 63 | 31 cats, 32 dogs. Class-stratified and duplicate-family-disjoint. `manifests/v002.validation.csv` |
| Final official audit | Pending | Actual student ID and frozen final manifest required |

**Pipeline.** Delivered-coordinate RGB decoding → frozen detector proposals → student correction/rejection → duplicate-family review → immutable six-column manifest → official reconstruction. No EXIF reorientation, padding, generative editing, external training images, dev-based image selection or evaluation-network curation. Final targets receive no image augmentation. Detector and evaluator weights remain separate from the generator.

**Quality and uncertainty.** The official manifest check rebuilt all 1,197 training crops. Recalculated occupancy is 60–80% throughout the frozen split, using recorded widths. Minimum recorded training head widths are 32.8 px for cats and 36.0 px for dogs. The highest observed detector fallback, box threshold 0.40 and text threshold 0.30, yielded 5 compliant unchanged proposals among 26 reviewed (19.23%, 95% Wilson interval 8.51–37.88%). The 23 missed heads are from a diagnostic rejected-source sample. Audit precision and its Wilson interval await the student labels.

**Observed pilot limitations.** Agent-inspected diagnostic sheets contain drawings, toys, prints on products, wrong species labels, loose boxes and boundary failures. These observations are exploratory and do not replace student verdicts. Source dHash pairs are candidates for visual/aligned-pixel duplicate confirmation, not automatic duplicate labels.

**Next action.** Complete the student's independent data check and the final official audit. Add the audit's per-class denominators, observed error types and an improvement tied to the actual errors.

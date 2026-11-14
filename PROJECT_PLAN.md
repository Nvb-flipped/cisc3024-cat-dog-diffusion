# CISC3024 course project: technical plan targeting 100/100

Prepared 30 October 2026. Submission deadline: **26 November 2026, 23:55, Asia/Shanghai**. The presentation takes place in the lab session that day.

**Current working deadline: 22 November 2026, Asia/Shanghai, per Paco's November 14 update.** Use this earlier date for the release. The schedule below records the original plan. Keep room before the 22nd for the student audit, independent checks, final rehearsal and a clean archive verification. Use the existing trained weights for bounded, evidence-driven comparisons wherever possible, and preserve weekly Codex allowance for final verification and release.

## 1. Objective, evidence, and fixed decisions

Build a class-conditional generator of new 32 × 32 RGB cat and dog head photographs, trained from scratch on a defensible training set curated exclusively from the supplied pool. Deliver reproducible code, an honest audit, a six-page report, and a rehearsed presentation in both academic Beamer and editable PowerPoint formats.

**100/100 is the acceptance objective.** All controllable rubric requirements must have evidence. Hidden-test results and the instructor's qualitative judgments cannot be guaranteed in advance. Do not call the project full-mark-ready merely because training completes, a sample grid looks attractive, or one development evaluation passes.

### Reviewed sources and current state

- [Course handout](<C:/Users/csmen/Documents/Codex/Pattern-Recognition-Project/project_handout.pdf>): all four pages, including data restrictions, evaluation tiers, disqualification conditions, and submission rules.
- [Project briefing](<C:/Users/csmen/Documents/Codex/Pattern-Recognition-Project/Project Briefing.pptx>): all 15 slides and available speaker notes.
- [Starter-kit archive](<C:/Users/csmen/Documents/Codex/Pattern-Recognition-Project/Course_Project_Resources/starter_kit.zip>): README, submission and agent templates, crop and manifest code, audit implementation, evaluator, metric implementation, dependency requirements, and relevant tests.
- The pool index has **76,250 unique IDs**. All are present across **22 ZIP shards**. This was an inventory check, not a full image-integrity scan.
- The development archive contains **904 images: 452 cats and 452 dogs**, with shape `[904,32,32,3]`, uint8 pixels. A deterministic sample was visually inspected. The heads vary in breed, pose, lighting, and background; curation must preserve that diversity.
- Available local hardware: **RTX 5060 Ti, approximately 16 GB VRAM; approximately 32 GB RAM; 16 logical CPU processors**. Microsoft PowerPoint is installed. No Git repository existed when inspected. A FigGenie environment has since been added; preserve it.
- No project model has been implemented, trained, or evaluated. Numerical targets below are planned acceptance criteria, not experimental results.

### Decisions already agreed with the student

1. Use the local GPU for long and overnight runs; no paid compute is assumed.
2. Reserve **30–45 minutes daily** for the student's data judgments, independent checks, design choices, and rehearsal.
3. Use **FigGenie as the first reference for figure work**. Its instructions explicitly exclude data charts; supplement it with scientific plotting skills for those charts.
4. Present two architecture design candidates and use the student's selected version consistently.
5. Every plotted graph gets **SVG and PNG** versions. Also retain vector PDF for LaTeX.
6. Every diagram additionally gets readable, editable **draw.io XML source** in a `.drawio` file, plus a plain `.xml` copy. This includes architecture, pipeline, mechanism, flow, and other schematic diagrams. For completeness, also provide draw.io versions of chart panels used in the presentation.
7. PPTX diagrams and plots must contain editable text and vector objects. Photographic samples remain **individual replaceable, movable, and croppable image objects**, as the student selected.
8. The student reports an **in-person instructor exemption** permitting AI slide generation and will personally revise the decks. Record that provenance accurately; do not describe it as a written exemption. The supplied handout otherwise prohibits AI-generated slides.
9. Default language: English for code documentation, report, figures, and presentations.

### Rules with precedence

The handout overrides conflicting starter-kit or briefing statements. In particular, briefing slide 7 calls audit labels optional, but the handout requires **all 200 verdicts and a reason for every bad crop**. Treat them as mandatory.

The pool is course-only material. Do not redistribute it. Do not use external images, generatively edited training images, pretrained generator weights, pretrained autoencoders, or the evaluation classifier/Inception network in curation, training, or generation. Development images and their per-image features must never become training examples, auxiliary-model training data, or pool-selection references. Aggregate official development metrics may guide pipeline, model, and checkpoint choices.

## 2. Rubric coverage and release gates

| Component | Marks | Required evidence for the full-mark target |
|---|---:|---|
| Data | 20 | Valid reproducible manifest; source-to-crop provenance; complete student-labeled 200-image audit; one-page data card with stage counts, precision and uncertainty, failure types, and next steps |
| Official evaluation | 15 | Accuracy ≥ 0.90, FID ≤ 50, recall ≥ 0.40, with every hard condition satisfied; each metric independently earns up to 5 marks |
| Code quality and reproducibility | 10 | Required project tests, additional risk-focused tests, pinned environment, exact reproduction commands, frequent meaningful Git commits |
| Report | 20 | At most six main-text pages; every required section; actual methods and equations; complete official result table; sample grid; honest failed attempts; citations |
| Working with coding agents | 10 | Maintained root AGENTS.md and three contemporaneous independent student checks, including at least one data check |
| Presentation | 25 | Four-minute presentation with live demo, followed by one to two minutes of questions; clear personal understanding and ability to defend decisions |

The materials do not provide finer numerical subrubrics for the qualitative components. Do not invent official submark allocations.

### Automatic disqualification prevention

Before accepting any candidate, require all of the following:

- The evaluator rebuilds the complete training set from the submitted manifest and available pool, with no manifest errors.
- All generation parameters are owned by the single module returned by `load_model`; total parameters ≤ **20,000,000**.
- Submission directory size meets the official **100 MB** limit. Use an internal ceiling of **90,000,000 bytes** to leave room for code, manifest, and serialization overhead; do not depend on differences between MB and MiB.
- Official memorization rate ≤ **5%**. The internal target is ≤ **2%**, assessed across multiple generation seeds.
- Repeated generation with the same inputs is deterministic; different seeds change the outputs.
- Generation produces finite float tensors of shape `[N,3,32,32]`, bounded by `[0,1]`.
- Single-class batch accuracy is no more than **0.20** below mixed-batch accuracy. Also test direct independence from other examples' labels.
- Loading and generation work offline, using files within the submission and declared installed libraries.

Use `--submission ... --grading`, with the pool present, for final local checks. The evaluator's `--samples` route skips submission checks, and a normal run without the pool can omit memorization; neither is release evidence.

### Performance margin

The official excellent thresholds remain 0.90 / 50 / 0.40. Aim for development accuracy ≥ **0.93**, FID ≤ **45**, recall ≥ **0.45**, and memorization ≤ **0.02** across five predeclared generation seeds. These are engineering margins, not official requirements or a statistical guarantee about the hidden set.

The evaluator's source comments report short reference cGAN/DDPM runs below all-three-excellent performance and describe higher tiers as extrapolated. Therefore, allocate substantial effort to curation, controlled tuning, and robustness rather than assuming a reference-size model will earn 15/15.

## 3. Workspace, interfaces, and reproducibility

### Setup

1. Initialize Git and establish exclusions before producing data or checkpoints. Keep supplied archives unchanged and record source checksums.
2. Extract the starter kit to `starter_kit/` without modifying its contents. Extract development data to `data/dev/`.
3. Copy the existing pool index and compressed shards into `data/student/`; keep the original inputs intact. **Do not expand the 22 image shards**: the official pool reader accesses them directly.
4. Create a separate project training environment. Start with Python 3.11 and a matching stable PyTorch/torchvision CUDA build that actually executes convolution and backward passes on this Blackwell GPU. Pin the successful versions and record driver/CUDA details. Preserve the existing FigGenie environment.
5. Run the unmodified starter-kit tests. Cache evaluator weights during setup; generation itself must never download anything.
6. Benchmark 500 training updates and a complete small evaluator run before estimating overnight schedules. Measure preprocessing, generation, Inception, and memorization separately; the evaluator's training-image deduplication can be expensive.

### Logical project layout

| Area | Responsibility |
|---|---|
| `starter_kit/` | Immutable official code, verified against supplied archive |
| `src/curation/`, `src/training/`, `src/evaluation/` | Pool-only curation, training, and orchestration of the untouched evaluator |
| `configs/`, `manifests/`, `annotations/` | Versioned experiment settings, immutable candidate manifests, and human decisions |
| `submission/` | Final inference code, one graded weight file, inference configuration, and exact training manifest |
| `tests/`, `results/`, `logs/` | Project verification, original evaluator JSON, run records, and machine-readable evidence |
| `figures/`, `report/`, `presentation/` | Figure sources/exports, report sources, and both presentation formats |
| `data/`, `checkpoints/`, private build folders | Local-only data, caches, intermediate models, and rendering products |

### Public inference contract

```python
def load_model(device: torch.device) -> torch.nn.Module:
    ...

def generate(model: torch.nn.Module,
             labels: torch.Tensor,
             seed: int) -> torch.Tensor:
    ...  # [N, 3, 32, 32], floating point, RGB, [0, 1]
```

- Labels are 0 for cat and 1 for dog. Load weights relative to `generate.py`, use `weights_only=True`, and return the model in evaluation mode.
- Use a private, explicitly seeded CPU `torch.Generator` for initial noise, then transfer noise to the model device. Do not consume or reset the caller's global random state.
- Follow the starter template's seeded per-position random stream: each fixed output position uses its own noise draw and requested label; changing other labels cannot change that output. Preserve the stream across internal microbatches. Do not interpret the wording as requiring every repeated label in a batch to produce an identical image.
- Use GroupNorm rather than training-time batch statistics. Dropout is disabled during generation. Test repeatability on each supported device; do not claim CPU/GPU bitwise equivalence without measuring it.
- Implement internal inference microbatching, initially 32 images, so the evaluator's default batch of 500 does not cause GPU memory failure.
- No evaluator imports, pretrained networks, training-image retrieval, classifier-based rejection sampling, or checkpoint selection inside `generate`.

### Provenance and commands

Every run records configuration, code commit, manifest SHA-256, data count by class, training seed, environment, start/end times, checkpoint hashes, and its parent checkpoint if resumed. A final model must never have trained on rows absent from its submitted manifest. If a revision removes previously used rows, retrain the final candidate from scratch.

Implement these CLI entry points; the following commands describe intended interfaces and do not yet exist:

```text
python -m src.curation.build --config configs/curation.yaml
python -m src.curation.review --candidates results/curation/candidates.csv
python -m src.curation.freeze --config configs/curation.yaml --out manifests/final.csv
python -m src.training.train --config configs/ddpm.yaml --manifest manifests/final.csv --seed 17
python -m src.evaluation.sweep --config configs/evaluation.yaml
python -m src.release.export --run <selected-run> --out submission
python starter_kit/audit.py --manifest submission/manifest.csv --student-id <actual-student-id>
python starter_kit/evaluate.py --submission submission --grading --n-per-class 1000 --seed 0 --out results/final/seed_0.json
python -m src.figures.build --registry figures/registry.json
python -m src.release.check --submission submission
python -m src.release.package
```

The README maps **each reported number and figure to one command**, its input hashes, and the original result file. Manual review is reproduced by replaying the saved student decisions; it must not be described as an automatic judgment. Capture frequent meaningful Git commits and retain readable history evidence in the submission.

## 4. Data pipeline and audit

### Curation algorithm

1. **Integrity and inventory.** Verify shard CRCs and image decoding, IDs, indexed dimensions, and RGB conversion. Save exact counts and reasons for failures. Use pool image coordinates as delivered, without undocumented EXIF reorientation.
2. **Head candidates.** Use a pinned local [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) checkpoint solely for curation. Start with explicit cat-head and dog-head prompts. Benchmark on 2,000 deterministically sampled pool images. Inspect whole-animal detections and missed heads before committing to the full pass; detector boxes are proposals, not ground truth.
3. **Calibration.** The student reviews 200 proposed crops and 200 randomly selected rejected pool images. Use the latter to diagnose missed categories, not as an unweighted estimate of overall accuracy. Search box thresholds `{0.20,0.30,0.40}` and text thresholds `{0.20,0.25,0.30}` on this same pool-only calibration set. Select the highest-yield setting whose reviewed candidate precision is at least 95%; if none qualifies, retain the highest-precision setting and route all accepted candidates through human review. Record the uncertainty and sample counts.
4. **Crop construction.** Correct proposals to include ears and the full head. Let `h` be the larger extent of that corrected head box; initialize square side `ceil(h / 0.70)`, centered on the head. Keep only source heads at least 32 pixels wide. Use integer crop coordinates and side, `size ≥ 40`, angle within ±45°, and the official rotated-square boundary check. Reject or manually correct boundary failures; never pad or fabricate content.
5. **Orientation.** Use angle zero by default. Save only explicitly justified corrections for tilted heads. Apply exactly the official crop order: rotation counter-clockwise around the crop center with bicubic resampling, square crop, then 32 × 32 Lanczos resize.
6. **Specification review.** Review source and final crop together. Check real photograph, correct species, exactly one head in the crop, visible ears, centered placement, roughly upright orientation, and approximately 60–80% occupancy. Flag cartoons, toys, prints, heavy occlusion, severe blur, multiple heads, and incorrect crops. Do not remove difficult breeds, dark fur, or varied backgrounds simply to simplify generation.
7. **Duplicate control.** Group identical source content and use perceptual-hash matches as duplicate candidates, confirmed visually or by aligned pixel comparison. Keep one appropriate crop per near-duplicate head group. Avoid repeated/strongly overlapping rows and contradictory labels. Save duplicate decisions; do not add duplicates to manipulate the memorization metric.
8. **Freeze and rebuild.** Emit exactly `pool_id,x0,y0,size,angle,label`, run the official manifest checker, rebuild the dataset, and verify the saved training cache against that rebuild. Aim for **2,000–4,000 distinct heads per class**, with at least 500 per class as the handout's stated aim. If the pool yields fewer suitable heads, preserve quality and report the actual count.

Keep per-image detector outputs and review records separate from the six-column manifest. Store source IDs, proposed and accepted geometry, detector revision, scores, decision/reason, duplicate group, and review timestamp. No dev image, dev embedding, or dev nearest-neighbor search enters this pipeline.

### Development policy

- Preserve a fixed 5% pool-only validation split, grouped by duplicate family and stratified by class. Keep it out of the final training manifest; use it for training diagnostics.
- Use the official dev set only for allowed visual specification understanding and aggregate official evaluations.
- Keep the primary training pipeline free of additional image augmentation: the clean training targets are the exact manifest rebuild. Diffusion noise is part of the learning objective. Any future augmentation must first be reconciled with the course provenance requirement and logged explicitly.
- Changes in dataset size, crop occupancy, or filtering create new manifest versions. Never silently change a training run's dataset.

### Final audit and data card

Run the official audit only after the final training manifest is fixed, using the student's actual ID. Its row selection depends on both ID and manifest rows; reordering the manifest does not create a new audit sample.

The student labels all 200 crops personally. A crop is good only if it satisfies the complete specification; every bad verdict includes a concrete reason. Inspect the original source when head resolution or authenticity is uncertain. The score depends on agreement with the instructors' relabeling of 50 crops, **not on declaring a high fraction good**.

Report overall audit precision `good / 200` with a **95% Wilson confidence interval**, and per-class estimates with their actual denominators. Treat these as estimated specification compliance, not detector accuracy on the entire pool. Explain the interval and remaining errors in the one-page DATA_CARD.md, alongside counts after every pipeline stage and one practical next improvement.

Use separate exploratory quality samples during development. Do not manipulate the final manifest to obtain favorable audit rows. If a genuine later correction changes the manifest, retrain as needed and regenerate and relabel the corresponding official audit.

## 5. Model, training, and experiment selection

### Primary method

Use a **pixel-space class-conditional DDPM**, trained entirely from random initialization, with classifier-free guidance from the same network. This avoids a pretrained latent encoder and permits a direct fidelity/diversity tradeoff. The method references are [DDPM](https://arxiv.org/abs/2006.11239), [Improved DDPM](https://arxiv.org/abs/2102.09672), [classifier-free guidance](https://arxiv.org/abs/2207.12598), and [DDIM](https://arxiv.org/abs/2010.02502). Implement and cite only the parts actually used.

| Item | Initial fixed configuration |
|---|---|
| Input/output | RGB 32 × 32, normalized to `[-1,1]` for training; three-channel noise prediction |
| U-Net resolutions | 32, 16, 8, 4 |
| Channels | 64, 128, 192, 256 |
| Blocks | Two residual blocks per encoder/decoder level, mirrored skip connections; two middle residual blocks |
| Normalization/activation | GroupNorm with 8 groups, SiLU |
| Conditioning | 256-dimensional timestep/class conditioning, injected as scale/shift in residual blocks |
| Attention | Four-head spatial self-attention at 8 × 8 and the bottleneck |
| Regularization | Residual dropout 0.10; null-label probability 0.10 |
| Forward diffusion | 1,000 timesteps; cosine cumulative noise schedule with offset 0.008; beta clipped below 1 |
| Objective | Uniform-timestep, unweighted noise-prediction MSE |
| Optimizer | AdamW, learning rate `2e-4`, betas `(0.9,0.999)`, weight decay `1e-4` |
| Batch | Effective 128; initial microbatch 64 and accumulation 2; reduce microbatch if needed without changing effective batch |
| Stabilization | Gradient norm clipping 1.0; EMA decay 0.9999; 2,000-update linear learning-rate warm-up, then constant learning rate |
| Precision | GPU BF16 if the smoke test passes; otherwise FP32; evaluator-facing inference initially FP32 |
| Initial budget | 100,000 optimizer updates; checkpoint every 10,000 updates |
| Sampling | Deterministic DDIM, eta 0; initially 100 steps and guidance scale 1.5 |

Count actual parameters before any long run. The engineering cap is 18 million parameters. If the specified implementation exceeds it, reduce channels uniformly to `(48,96,144,192)` and record that change before the baseline is frozen. The official maximum remains 20 million.

For clean image `x0`, sampled timestep `t`, and Gaussian noise `epsilon`, implement:

```text
xt = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * epsilon
loss = mean((epsilon - epsilon_theta(xt, t, dropped_or_real_label)) ** 2)
epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

For deterministic DDIM steps from `t` to `t_previous`, reconstruct `x0_hat` from the predicted noise, apply the documented clipping convention, and combine it with the corresponding previous-time noise coefficient. Test the timestep indexing, endpoint, and clipping behavior directly. The report must reproduce the implemented equations, including the actual guidance convention: **s = 1 means ordinary conditional prediction**.

### Staged experiment matrix

| Stage | Runs | Decision and purpose |
|---|---|---|
| Correctness | Tiny configuration and one training minibatch | Prove data, forward diffusion, loss, gradients, sampling, and export work before long runs |
| Baseline | Main architecture, seed 17, 50k updates; guidance 1.0 | Establish a reproducible conditional diffusion baseline with all official metrics |
| Learning-rate comparison | Same architecture/data/seed/budget; `1e-4`, `2e-4`, `3e-4` | Isolate optimizer choice; reuse the corresponding baseline run |
| Main training | Extend best learning rate to 100k; up to 200k if metrics still improve | Increase compute only when validation evidence supports it |
| Sampling sweep | EMA/raw weights; DDIM steps `{50,100,200}`; guidance `{1.0,1.5,2.0,2.5}` | Find the accuracy–FID–recall–speed tradeoff without retraining |
| Data ablation | Balanced 1,000/class subset versus the full curated set, otherwise matched 50k budget | Measure the effect of data quantity under equal optimizer updates; also report images seen |
| Crop ablation | Occupancy 0.65 versus 0.75 on the same eligible source-head set | Test alignment choices while keeping both versions within the specification |
| Repeatability | Winning training configuration with seeds 29 and 43, alongside 17 | Assess training variability; select one checkpoint, not a weight ensemble |

The data/crop ablations use their own manifests and fresh initialization. If the full dataset has fewer than 1,000 heads in either class, use a balanced 500-per-class subset for the quantity ablation; if this leaves no meaningful size contrast, report that limitation and omit the contrast rather than duplicating data. EMA and sampling comparisons reuse the same checkpoint. Record failed runs and negative results as they occur, including the reason a configuration was rejected. Do not invent an unsuccessful method to satisfy the report section.

### Evaluation protocol and selection rule

1. Use 250 generated images per class for bounded screening, always comparing like sample counts. Inspect every output grid. Small-sample FID is not a full-mark certificate.
2. Promote at most three checkpoint/sampler combinations to the official 1,000-per-class protocol at default evaluator batch size 500, using generation seeds **0, 101, and 202**.
3. Reject candidates with any violation, missing memorization, or unrebuildable data. Rank valid candidates by the lowest official tier total across those seeds. Within the highest tied tier total, prefer the larger minimum normalized margin across accuracy, FID, and recall; then lower memorization; then faster sampling.
4. Freeze the selected checkpoint and inference configuration before running confirmation seeds **303 and 404**. Report all five runs, not only the best seed. If confirmation fails a required threshold, reopen model selection and record that those seeds are no longer untouched confirmation evidence.
5. Use the official seed-0 result as the exact headline table row; also present five-seed ranges/means and the training-seed comparison. Do not substitute an average for an official run or call seed variation a confidence interval for hidden-test performance.
6. Benchmark final loading and generation on CPU as well as GPU. Report actual parameters, submission bytes, wall-clock training time, hardware, and seconds per 1,000 images.

For the normalized margin in step 3 use the minimum of `(accuracy - 0.90)/0.05`, `(50 - FID)/10`, and `(recall - 0.40)/0.10` across the three selection seeds. This ranking balances all three graded metrics rather than optimizing one at the expense of another.

### Recovery rules

- **Low accuracy:** inspect species labels and crops; compare moderate guidance settings. Never invoke the evaluation classifier during generation or filtering.
- **Poor FID:** inspect crop scale and photographic fidelity; compare EMA, sampling steps, learning rate, and genuine training progress. Do not select individual training images using evaluator features.
- **Low recall:** reduce excessive guidance and restore genuine breed/pose/background diversity; evaluate the larger clean dataset.
- **High memorization:** inspect generated-to-training neighbors, duplicate families, training duration, and capacity; improve genuine diversity or select an earlier checkpoint. Never disguise copies or manipulate the metric's reference set.
- **No improvement:** after three comparable full evaluations without an improved selection criterion, stop extending that run and use the remaining budget on the diagnosed issue.

## 6. Tests and the student's independent verification

### Automated tests

| Test family | Required scenarios and acceptance |
|---|---|
| Crop equivalence | Compare project preprocessing with `catdog_eval.data.crop` on real pool images and synthetic coordinate fixtures; cover angle 0, positive/negative rotation, boundary crops, integer rounding, RGB conversion, and Lanczos output |
| Manifest/provenance | Missing IDs, duplicate geometry, wrong labels, undersized crops, out-of-bounds rotated squares; exact cache/rebuild agreement; checkpoint manifest hash mismatch must fail |
| Forward process | Analytic noise injection/reconstruction, timestep endpoints, finite coefficients, correct normalization, no off-by-one sampler indices |
| Single-batch overfit | Repeatedly train on one fixed image minibatch; fixed denoising probes must show a substantial loss reduction and reconstructed examples must improve; save the before/after evidence and diagnose failure before full training |
| Generation contract | N = 1, 2, 20, 50, 500; both labels, mixed and homogeneous batches; exact shape, finite values, bounds, and seed sensitivity |
| Determinism/isolation | Repeat calls and fresh-process reloads; change other labels while holding a target position fixed; alter internal microbatch size; preserve the caller's RNG state |
| Offline portability | Import submission from a clean working directory; disable downloads; load on CPU and GPU; no dependency on training folders or absolute paths |
| Resources/release | Count all module parameters and directory bytes; only the graded weight file; no raw images, alternate checkpoints, environments, or secrets in the course ZIP |
| Figure formats | SVG/PNG/PDF dimensions, draw.io XML validity and IDs, native PPTX object inventory, labels/values/geometry parity, successful open/save/reopen |

Give the single-batch check a fixed diagnostic budget of 2,000 updates and require at least an 80% reduction on its fixed probe MSE relative to initialization. If that fails, investigate the implementation rather than weakening the test after seeing the result. This is an integration sanity check, not evidence of generalization.

### Three authentic AGENT_LOG entries

The agent may prepare artifacts and commands, but the student must perform and explain the independent check. Mark entries pending until that happens; do not fabricate dates, judgments, failures, or first-person claims.

1. **Data check:** inspect 30 source/crop pairs independently, including difficult species, near-boundary crops, and a rotated case. Measure source head width and crop occupancy on selected examples; compare with official reconstruction. Record actual disagreements or agreement and the rule/test added.
2. **Model/interface check:** manually work through one forward-noise/reconstruction example and independently run fixed-seed, changed-label, and mixed/single-batch probes. Explain why class conditioning and noise indexing cannot depend on other labels.
3. **Results/reproducibility check:** rerun the untouched official evaluator from a fresh process, confirm manifest and memorization checks ran, verify all reported numbers against JSON, and inspect nearest-neighbor examples and sample grids. Check an experiment conclusion against its underlying runs.

Each contemporaneous entry contains context, the agent's output/claim, the student's check, raw evidence, outcome, and any resulting change to AGENTS.md/tests. A sound check can earn full marks even when no mistake is found.

## 7. Figures, draw.io, Beamer, and editable PPTX

### Skill routing

1. **FigGenie first:** use the installed [FigGenie skill](<C:/Users/csmen/Documents/Codex/Pattern-Recognition-Project/.agents/skills/figgenie-paper-diagram/SKILL.md>) for architecture, mechanism, and model-overview design, specifications, text measurement, SVG validation, and editable exports.
2. Use `scientific-visualization` and `scientific-figure-making` for actual data plots. FigGenie's own scope says “Not for data plots or charts”; consulting it first must not force charts through an unsuitable schematic workflow.
3. Use Vivid's architecture guidance for the requested independent A/B candidates and for export cases it supports better. Preserve FigGenie's design priority rather than mixing unrelated template styles.
4. Use the presentation skill for deck assembly and checks, and the LaTeX workflow for report/Beamer compilation. Cite materially used software, skills, model checkpoints, and sources; retain asset/font licenses.

No additional training framework is required. A PyTorch Lightning skill was located during discovery, but plain PyTorch is the selected implementation to keep the training and inference code easy to explain and inspect.

### Figure inventory

| ID | Figure | Evidence or purpose |
|---|---|---|
| F1 | Curation and provenance pipeline | Actual stage counts, permitted curation models, human review, immutable manifest; clearly separate dev evaluation from training |
| F2 | Conditional diffusion architecture and sampling mechanism | U-Net resolutions/channels, skip connections, timestep/class inputs, training loss, EMA, and conditional/unconditional guidance from one model |
| F3 | Data quality and audit | Class counts, source head width/crop occupancy distributions, observed error types, audit precision and 95% intervals |
| F4 | Training and evaluation trajectories | Comparable checkpoints and actual measured metrics; distinguish training seeds from generation seeds |
| F5 | Ablations and sampling tradeoffs | Effects of data quantity, crop occupancy, EMA, guidance, and sampling steps; show accuracy/FID/recall together |
| F6 | Reproducible generated sample grid | Predeclared seed and sample indices, both classes, no best-image selection or enhancement |
| F7 | Memorization diagnostic | Actual generated samples with nearest training neighbors and distances/ratios; distinguish diagnosis from official aggregate metric |

Use a subset in the six-page report and main talk; retain the rest in the appendix or backup slides. Do not put every available plot into the main presentation.

### Journal-style design and selection

Use Nature-style restraint for typography, color, line work, and editable artwork, with FigGenie's AI-figure examples informing the diffusion drawing. This is a course project, not a claim of journal acceptance. Follow the [Nature figure guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) for legibility and editable vectors.

For F1 and F2, prepare a concise factual brief and a single reference sheet. Bundle the student's brief confirmation and exemplar preferences into one review, then produce A/B alternatives from the same facts. A preserves the selected reference's reading structure; B reorganizes the structure. Show both at equal width and use the student's selection. This implements FigGenie's brief/exemplar review and the already requested two-candidate process without unnecessary repeated permission requests.

Show only implemented components and real measured values. Draw no pretrained encoder/decoder in this pixel-space model. Use one consistent color identity for data, trainable components, frozen curation tools, and evaluation; visibly distinguish their roles. Keep labels readable at final report size and projected slide size.

### Common source and output contract

Maintain one versioned figure specification containing semantic IDs, text, geometry, colors, routes, source-data references, and font choices. A selected SVG is the visual reference; update the specification and all exports together after an accepted edit.

Each plot/diagram delivery includes:

```text
figure.svg              editable vector reference
figure.png              matching raster export, normally 300 dpi at intended size
figure.pdf              vector PDF for LaTeX/Beamer
figure.drawio           uncompressed, readable draw.io XML
figure.xml              identical plain XML source for inspection/editing
figure.pptx             native editable figure objects for deck reuse
figure.spec.json        semantic and geometric specification
source data + generator configuration + export/QA record
```

Photographic grids also retain their source arrays and individual image objects. Their SVG may embed the unchanged raster samples; do not claim that photographs become vector information. Preserve the actual 32 × 32 resolution and disclose enlarged display; do not use generative enhancement or smoothing to imply additional detail.

### draw.io implementation and tests

- Produce an uncompressed `<mxfile>` containing `<diagram>` and `<mxGraphModel>` structures. Use stable IDs shared with the SVG specification.
- Use native editable cells for boxes, labels, and diagram elements, and native edges with source/target IDs and explicit waypoints. Represent custom vector components with editable vector stencils where necessary. Never make the entire diagram a single embedded SVG or PNG.
- Preserve grouping/layers, canvas dimensions, colors, line widths, arrowheads, and typography. Express text as text, not outlined glyph images.
- Include native chart marks and labels in draw.io copies of presentation chart panels, with the corresponding CSV/JSON data. A draw.io chart is an editable drawing, not a spreadsheet chart with automatic recalculation.
- Validate XML, unique IDs, edge references, object inventory, and label/value parity. Open in draw.io, edit a label and move a node in a test copy, verify connected edges, save/reopen, and compare its rendered output with the selected SVG. This follows draw.io's [documented editable XML model](https://www.drawio.com/docs/manual/advanced/diagram-source-edit/).

### PPTX editability and visual parity

Use FigGenie's SVG-to-native-object route for supported diagrams. Resolve unsupported SVG constructs into supported vector primitives or use the verified Vivid converter where appropriate; never silently rasterize. Preserve native objects when assembling the presentation.

Acceptance requires native text boxes, vector paths/shapes, connectors, and meaningful groups. A single embedded SVG is insufficient. Keep each photograph as a separate picture object. Retain underlying chart data alongside editable vector plots; document whether editing data requires rerunning the plotting script.

Compare object inventories, all text/numbers, geometry, colors, and line styles. Render from **Microsoft PowerPoint**, align with the reference SVG rendering, and inspect overlays/differences. Allow only renderer antialiasing differences; missing objects, shifted geometry, font substitutions, changed labels, or altered data fail. Do not promise byte-identical pixels across different renderers.

In a test copy, edit a label, recolor a data mark, move a grouped module, and replace/crop one photo. Save and reopen in PowerPoint. Confirm that the delivered final copy has the original approved design. Check equation appearance in PowerPoint if native equation conversion changes fonts. Shared simple labels should use installed fonts available in SVG, draw.io, and Office.

PPTX edits do not automatically update SVG or draw.io. After the student's final changes, reconcile the canonical specification and regenerate/check matching exports; retain the student's edited deck without destructive overwrite.

### Presentation deliverables and timing

Deliver editable `presentation.pptx`, complete Beamer `.tex` source and required build inputs, and compiled `presentation.pdf`. Use 16:9 slides and a restrained academic theme, English wording, matching figure versions, and identical claims/results across both decks. The decks need equivalent content and figure fidelity; their native text layout need not be pixel-identical.

| Main segment | Time | Content |
|---|---:|---|
| Task and outcome | 20 s | What was built; actual headline results |
| Data | 55 s | Pipeline, a concrete crop decision, audit precision/uncertainty |
| Method | 45 s | Conditioning, diffusion objective, guidance and its tradeoff |
| Live demo | 45 s | Load the graded model, generate both classes, repeat/change a seed |
| Results | 45 s | Official table and one informative ablation |
| Agent verification | 30 s | One authentic independent check and resulting AGENTS rule |
| Questions | 1–2 min | Student explains and defends the work |

Use the exact submitted checkpoint and sampling configuration in the demo. Preload the model before speaking; aim for a 16-image demo that completes within 10 seconds on the presentation hardware. Keep a clearly labeled recorded fallback, but do not substitute it for demonstrating the working model when live execution is available.

## 8. Report and packaging

### Six-page report allocation

| Content | Approximate space |
|---|---:|
| Summary and problem | 0.5 page |
| Data pipeline and audit | 1.25 pages |
| Implemented methods and equations | 1.25 pages |
| Results, complete metric table, sample grid, ablations | 1.75 pages |
| What did not work and limitations | 0.5 page |
| Working with coding agents and conclusion | 0.75 page |

The main text must remain ≤6 pages, excluding references and appendix. Put the student's actual name and ID on `report.pdf`. Include accuracy, FID, KID with its reported variability, precision, recall, memorization, parameter count, hardware, training time, and sampling speed. Identify the dev reference and generation protocol explicitly. Trace every reported value to stored evaluator output; never invent hidden-test results.

Explain choices causally: what changed, what evidence supports the improvement, what failed, and what remains uncertain. Separate equal-budget ablations from longer-training comparisons. Attribute the detector, diffusion methods, borrowed code, visual references, and materially used skills. Include the [Scientific Agent Skills reference](https://arxiv.org/abs/2609.00065) when its installed skills materially contribute.

### Two delivery packages

**Complete local project package:** retain all requested SVG/PNG/PDF/PPTX/draw.io/XML assets, figure data and generators, report/Beamer sources, original experiment evidence, and final model. Raw course data remains local and is excluded from any shareable package.

**UMMoodle submission:** upload `report.pdf` separately and a clean project ZIP containing:

- Final `submission/`: inference code and dependencies authored for it, one graded weight file, inference config, and the matching manifest.
- `audit/audit.csv`, source code, tests, configs, text/JSON/CSV experiment logs, and reproducibility evidence.
- README.md, DATA_CARD.md, AGENTS.md, AGENT_LOG.md, citations/licenses, and Git-history evidence.
- Presentation PPTX, Beamer source, compiled presentation PDF, and the scripts/data needed to regenerate supporting figures.

The handout excludes images, the pool/data directory, virtual environments, and alternate checkpoints from the ZIP. Therefore, keep standalone rendered image assets and audit contact sheets in the local deliverable package; embed necessary visuals in the required report/slides and regenerate supporting graphics from submitted code/data. Retain draw.io/XML schematic sources in the ZIP only if they contain no embedded course photographs; otherwise keep them in the local package and provide regeneration inputs. Build the ZIP from an explicit allowlist. Exclude caches, model-download folders, `.venvs`, machine credentials, `.git` internals, and the skill asset corpora; include a readable Git-history export while preserving the complete repository locally.

Test the actual archive by extracting it into a clean directory, supplying only the separately installed course data/environment, and following the README. Confirm the final report and both decks agree with the exported model, manifest, audit, and metric JSON. Keep file hashes for the exact uploaded artifacts.

## 9. Schedule, human ownership, and completion checklist

All dates and times below use Asia/Shanghai. The schedule assumes the agreed overnight GPU access and 30–45 minutes of student involvement each day.

| Dates | Machine/agent work | Student work and exit condition |
|---|---|---|
| Oct 30–Nov 1 | Environment, immutable inputs, Git, official tests, detector pilot, throughput benchmarks | Inspect dev examples; calibrate judgments; confirm dataset specification; first independent data check |
| Nov 2–5 | Full-pool proposals, crop tooling, review replay, duplicate control, first valid manifest; model sanity tests | Daily crop review; approve calibration decisions; at least 500 valid heads/class before baseline training |
| Nov 6–10 | Baseline, learning-rate comparison, improved data pass, preliminary full evaluations | Review errors and sample grids; perform model/interface check; confirm figure briefs and references |
| Nov 11–16 | Main training, sampling sweep, data/crop ablations, repeat training seeds | Resolve borderline data cases; select A/B figures; explain actual method and results aloud |
| Nov 17–20 | Freeze final manifest/model candidate, official audit, five-seed evaluation, CPU/offline tests | Personally label all 200 audit rows; perform independent result check; sign off factual conclusions |
| Nov 21–23 | Six-page report; full figure package; Beamer/PPTX; draw.io and Office verification | Revise report/slides; reconcile personal deck edits; rehearse four-minute talk |
| Nov 24–25 | Clean extraction test, final evaluation and cross-document checks, archive hashes | Two timed rehearsals with live demo and questions; target upload by **Nov 25, 18:00** |
| Nov 26 | Preserve stable final artifacts; address only verified critical issues | Present in assigned lab; verify upload receipt before **23:55** |

The final model/manifest freeze is a real dependency: changing either can invalidate the audit, metrics, report numbers, or demo. Do not make a late “small improvement” without rerunning the affected checks.

If excellent-tier margins are missing by November 16, prioritize the diagnosed data/model problem over optional aesthetics and additional experiments. If the target is still unmet near the deadline, report the shortfall explicitly and submit the strongest valid work on time; never fabricate success or miss the deadline to preserve an unsupported full-mark claim.

### Final acceptance checklist

- [ ] Every handout requirement has an evidence link and responsible owner.
- [ ] Starter-kit files remain unchanged; data and environment setup are reproducible.
- [ ] Final manifest reconstructs the complete training set; all training provenance agrees.
- [ ] The student completed the exact official 200-row audit; all bad rows have reasons.
- [ ] DATA_CARD.md reports stage counts, audit precision/uncertainty, errors, and next steps.
- [ ] Official full-protocol evaluations include all checks and all six quality metrics.
- [ ] Excellent-tier accuracy/FID/recall and every hard limit have been assessed across the declared seeds; any unresolved shortfall is explicit.
- [ ] Required project tests, offline loading, CPU generation, and resource checks pass.
- [ ] Three genuine independent student checks and updated AGENTS.md are complete.
- [ ] Report main text is ≤6 pages and every number/citation is verified.
- [ ] FigGenie led diagram design; chosen A/B versions are used consistently.
- [ ] Every plot has matching SVG/PNG; every diagram has editable draw.io/XML; vector PDF and source data are retained.
- [ ] PPTX contains native editable plot/diagram objects and separate photographic objects; actual PowerPoint edits and rendering were checked.
- [ ] Beamer source compiles; both presentations agree with the report and final model.
- [ ] The student can explain the pipeline, equations, evaluation limitations, and agent checks without relying on “the agent did it.”
- [ ] A timed live demo works with the submitted checkpoint; backup material is clearly labeled.
- [ ] The submission ZIP passes clean-room verification and excludes prohibited data/images/checkpoints/environments.
- [ ] Exact upload artifacts and receipt are preserved before the deadline.

Student name and ID remain runtime inputs for the report and official audit. No additional design preference is needed to begin implementation; the scheduled figure brief/reference and A/B reviews are explicit human tasks within this plan.

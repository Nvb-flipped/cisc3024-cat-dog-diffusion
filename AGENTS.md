# Project rules

Execute PROJECT_PLAN.md; the course handout has precedence. This is an individual CISC3024 project. Keep raw course material local.

- Train the generator from random initialization using only exact crops in its versioned pool manifest. No dev images, dev per-image features, external training images, pretrained generator/autoencoder, or generatively edited targets.
- Pretrained detectors are permitted for pool curation only. Never use the evaluator classifier or Inception for curation, training, or generation.
- Keep starter_kit/ byte-identical to its supplied archive. Run it unchanged for official evidence.
- Required data: real photographs, exactly one centered roughly upright cat/dog head including ears, 60-80% occupancy, original head width >=32 pixels; labels cat=0, dog=1. Use official crop geometry and no padding.
- Human decisions must be replayable. Do not invent student review, audit labels, independent checks, or measured results. Final audit and AGENT_LOG checks remain pending until personally completed by the student.
- Student review availability: daily 12:00–13:00 GMT+8 (Asia/Shanghai). Prepare one concise review queue for that window and continue independent work outside it.
- All inference code/weights/config/manifest belong in submission/. Return one module containing all parameters, <=20M parameters, <=100 MB. Engineering ceilings: 18M and 90,000,000 bytes.
- Generation is offline, uses a private CPU RNG, preserves caller RNG, supports internal microbatching, and isolates each output from other labels. Test GroupNorm/eval behavior and deterministic DDIM endpoints.
- Any manifest removal after training requires fresh final training. Record manifest/code/checkpoint hashes, seeds, timing, versions and actual commands.
- No report success claim without full official evaluation including pool reconstruction and memorization. Student-reported slide-generation exemption was in person, not written.
- Figure work starts with FigGenie; human brief/exemplar and A/B selection remain pending. Keep requested editable formats and source data.
- Approved-only duplicate review runs separately at http://127.0.0.1:8767. Hash similarity is only a comparison aid. Every family/keep choice needs a personal action; bind it to geometry, species and preview. Exact repeated crop pixels cannot both be kept. Preserve manual CSV edits and reconcile them rather than overwriting.
- Baseline training requires frozen personal-review provenance and its pool-content fingerprint. Reuse the exact 2e-4/seed-17/50k baseline in LR comparisons. Quantity ablations use exact parent rows, the same holdout, fresh initialization and a matched update budget; omit a meaningless size contrast.
- Synthetic UI fixtures stay under private_build and use software_fixture origins/diagnostic flags. They are never student judgments, final audit labels or training approvals.
- Every new priority shortlist batch includes a visually inspected agent draft for each exact preview: suggested species, estimated source head width, checklist, and a concrete reason wherever a criterion needs attention. Calculate occupancy from width and crop side. Bind drafts to preview hashes and require the student's explicit check before a verdict is saved. Record confirmed agent estimates as such. Publish matching findings for each batch.

Layout: src/{curation,training,evaluation,release}, configs/, manifests/, annotations/, tests/, results/, logs/, submission/, figures/, report/, presentation/. Local-only: data/, models/, checkpoints/, .venvs/.

Commands and current state: README.md and EXECUTION_STATUS.md. Before substantial work and periodically during a run, check account usage. Stop at 90% five-hour usage, save a handoff/reset time, and resume after reset as authorized. Do not consume reset credits.

Usage reserve: at 80% used (20% remaining), warn once and begin checkpointing/wrapping up. At 85% used, start no new substantial stage and check usage frequently. At 90% used (10% remaining), stop project work cleanly. Do not spend the reserve until an interruption occurs. Actual usage checks drive these thresholds; no native quota-event capability is assumed.

Verification: official tests; crop parity; provenance/cache parity; diffusion math; fixed single-batch 2,000-update diagnostic with >=80% probe-MSE reduction; inference isolation/shape/range/RNG; clean export/offline load; actual archive allowlist check.

Known Windows pitfall: sandbox default temporary folders can reject pytest fixtures. pytest.ini directs fixtures to workspace-local .pytest_tmp. Never change official tests to bypass this environment issue.

Windows process pitfall: the PID returned by a virtual-environment Python launcher can differ from the listening runtime PID. Read the actual localhost port owner and verify its command line before stopping a service. Sandboxed process inspection may not see the elevated host service; do not infer that a live browser-accessible service has exited from that alone.

Review UI pitfall: overlapping saves can otherwise misattribute a late response or skip a crop; serialize saves and attach each result to its original candidate. Discard stale previews and measurements when geometry changes. For a drafted crop, keep the student's current checklist choices on geometry or species edits and reset the separate confirmation; the student updates the individual checks. For a crop without a draft, clear old criteria on geometry edits. Browser checks must not submit real student verdicts.

Measured inference pitfall: on the actual CUDA U-Net, changing physical batch dimensions amplified floating-point differences near the noisy cosine endpoint (an untrained stress test observed max output difference 0.968). Sampling now pads logical chunks to one serialized execution_batch, keeping arithmetic shape fixed. Test changed logical microbatch sizes and N=1 versus mixed prefixes. Treat a change of execution_batch as a sampler configuration change requiring reevaluation; it is not merely a performance switch.

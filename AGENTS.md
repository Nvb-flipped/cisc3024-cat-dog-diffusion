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

Layout: src/{curation,training,evaluation,release}, configs/, manifests/, annotations/, tests/, results/, logs/, submission/, figures/, report/, presentation/. Local-only: data/, models/, checkpoints/, .venvs/.

Commands and current state: README.md and EXECUTION_STATUS.md. Before substantial work and periodically during a run, check account usage. Stop at 90% five-hour usage, save a handoff/reset time, and resume after reset as authorized. Do not consume reset credits.

Verification: official tests; crop parity; provenance/cache parity; diffusion math; fixed single-batch 2,000-update diagnostic with >=80% probe-MSE reduction; inference isolation/shape/range/RNG; clean export/offline load; actual archive allowlist check.

Known Windows pitfall: sandbox default temporary folders can reject pytest fixtures. pytest.ini directs fixtures to workspace-local .pytest_tmp. Never change official tests to bypass this environment issue.

Measured inference pitfall: on the actual CUDA U-Net, changing physical batch dimensions amplified floating-point differences near the noisy cosine endpoint (an untrained stress test observed max output difference 0.968). Sampling now pads logical chunks to one serialized execution_batch, keeping arithmetic shape fixed. Test changed logical microbatch sizes and N=1 versus mixed prefixes. Treat a change of execution_batch as a sampler configuration change requiring reevaluation; it is not merely a performance switch.

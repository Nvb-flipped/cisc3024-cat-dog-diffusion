# Independent model/interface check — pending student work

This worksheet does not count as a completed check. Use pencil/paper or a script you write and understand yourself, and keep the actual output.

- Work through one scalar forward-noise/reconstruction example with clean value 0.4, cumulative alpha 0.81 and sampled noise −0.2. Compute the noisy value, then recover the clean value using the same noise. Derive why the square roots appear. Compare your calculation with `submission/diffusion.py`; do not use an agent's calculated answer as the check.
- Trace a class label and a timestep through `submission/model.py`. Identify the three embedding entries, how null class 2 is used, and where scale/shift reaches a residual block. Explain why GroupNorm and within-image attention cannot read another example's label.
- Read the original interface requirements in the course template. Independently construct repeated-seed, changed-seed, changed-other-label, single-versus-mixed-prefix and logical-microbatch probes. Use the same fixed output position when checking other-label changes. Inspect the caller RNG before and after loading/generation.
- Explain the measured implementation pitfall in AGENTS.md: changing physical CUDA batch shape produced different arithmetic near the noisy cosine endpoint. Distinguish the serialized physical execution size from the logical microbatch, and check that the final export uses the frozen sampler settings.
- Use the actual final/candidate export once it exists; record its weights/config/manifest hashes and your device. A diagnostic or fake-network test does not establish the final model's behavior.

When performed, write the actual date, procedure, raw evidence and outcome in AGENT_LOG.md. Describe what you checked without relying on an agent's claim. No mistake needs to be invented if the implementation is correct.

# Independent results check — pending final experiments

Wait for a frozen candidate and real official results. The current placeholder smoke is environment evidence and must not be used as the project result table.

- Start a fresh process and independently invoke the unchanged `starter_kit/evaluate.py` on the exact exported submission, with course pool present, grading enabled, 1,000 generated images per class and evaluator batch size 500. Record the generation seed and input hashes. Preserve the original JSON rather than copying rounded numbers by hand.
- Confirm that the complete manifest rebuilt, memorization ran, and all hard checks passed. Identify accuracy, FID, KID and its reported variability, precision, recall, memorization, parameters, disk size and sampling time in the original result.
- Match every headline table value to seed 0. Inspect all five declared generation seeds; distinguish a range or mean across seeds from a confidence interval for hidden-test performance.
- Inspect the predeclared sample grid and actual nearest-training-neighbor diagnostics. Explain one failure mode and its evidence, or explicitly state when a suspected failure was not observed.
- Choose one experimental conclusion and independently compare the underlying runs: exact data/manifest, training budget, training seed, sampler and metric protocol. Separate equal-budget comparisons from longer training. Check that a failed confirmation was not silently reused as untouched confirmation evidence.
- Check report, data card, Beamer/PPTX and live demo against the same final artifact hashes. Preserve your actual command/output and explain any discrepancy or confirmed agreement in AGENT_LOG.md.

The student must supply this account when the check actually happens. This file is a procedure, not a fabricated completed entry.

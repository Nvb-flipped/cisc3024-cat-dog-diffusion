# Daily student review queue

Your stated review window is **12:00–13:00 GMT+8**. Work at a comfortable pace; saved judgments are replayable, and you can resume later. These are pending human tasks, not completed checks.

1. Open [crop calibration](http://127.0.0.1:8766/). Before starting, reload an old tab to get the current interface. Review the fixed **200 proposed crops** and **200 rejected sources**, enter your name, and save your actual decisions. Use “Next unreviewed” when returning. Good crops need the full specification, actual species, source width and occupancy; correct geometry must be previewed before saving. A detector/CLIP suggestion is not ground truth.
2. After calibration, the agent can summarize the nine settings without counting corrections as original successes. Continue the full-pool review in later sessions. The eventual baseline needs at least **500 kept training heads per class after duplicate review and the holdout**; proposals and clicks on “bad” do not count toward that threshold.
3. When you have approved crops, use [duplicate families](http://127.0.0.1:8767/). Compare originals/crops, confirm shared or distinct family IDs, and keep one representative. Check across suggested sets too. No family decisions are made automatically.
4. For the first independent check, use [the data worksheet](student_checks/01_data_check.md). The local study pack is `private_build/student_data_check_v001/`: contact sheet, full sources, exact crops and blank measurements. It is deliberately difficult and is **not the final audit**. Independently inspect the original course crop code and perform the check; then record what actually happened in AGENT_LOG.md. The agent's pixel-parity result alone is not your check.

Later, review [F1/F2 factual briefs](figures/review/briefs.md) and `figures/review/model_exemplar_candidates.png`, choose 2–4 numbered references, then select between the resulting A/B designs. This is the already agreed design review; no architecture drawing is approved yet.

The model/interface and results worksheets in `student_checks/` are for later candidate/final-artifact checks. Final audit labels, your actual name/ID, source/conclusion verification, slide revision and rehearsal remain your tasks. None have been filled in on your behalf.

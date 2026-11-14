# Student review and submission checklist

Calibration, Priority shortlist Batches 1–19, and their duplicate reviews are complete. All 13 approved Batch 19 crops are in the frozen v007 training set. Your usual review window is 12:00–13:00 GMT+8.

The 30 verdicts for Task 1 are saved. The remaining steps are selected source measurements, the official 30-row rebuild, and your account in `AGENT_LOG.md`.

The working submission deadline is **November 22, Asia/Shanghai**. Complete the independent data check and figure choices while the final model is being selected. The official audit and final results rerun depend on the frozen submission.

## Ready now

| Task | Open | What to save or send |
| --- | --- | --- |
| **1. Independent data check** | [Instructions](student_checks/01_data_check.md) · [30-pair contact sheet](private_build/student_data_check_v001/contact_sheet.png) · [Full-size sources and crops](private_build/student_data_check_v001/images) · [Measurement CSV](private_build/student_data_check_v001/personal_measurements.csv) | Your verdict and species for all 30 pairs, measurements on selected difficult examples, the official rebuild output, and your account in [AGENT_LOG.md](AGENT_LOG.md). |
| **2. Independent model/interface check** | [Worksheet](student_checks/02_model_interface.md) · [Current candidate's diffusion code](private_build/v007_32k_pag015/diffusion.py) · [Model code](private_build/v007_32k_pag015/model.py) · [Sampler settings](private_build/v007_32k_pag015/inference.json) | You can do the scalar example and code walkthrough now. Save your interface probes and their output. If the export changes, repeat the probes on the final export and record the hashes and device in [AGENT_LOG.md](AGENT_LOG.md). |
| **3. Figure brief and reference choices** | [F1/F2 briefs](figures/review/briefs.md) · [Numbered reference sheet](figures/review/model_exemplar_candidates.png) | Reply with any factual corrections and 2–4 reference numbers, with a short note about what you like in each. F1 uses historical v001 counts. The latest candidate to show in F2 uses v007, 32k updates, DDIM50, CFG 3.25 and PAG 0.15. |
| **4. Name and ID** | [Course project plan](PROJECT_PLAN.md) | Use your official name and actual student ID for the report and the official audit. You can enter the ID locally when you run the commands. |
| **5. Read and practise with the current drafts** | [Development report](private_build/report_20261114_development/report.pdf) · [Editable PowerPoint draft](private_build/presentation_review/development.pptx) · [Beamer draft](private_build/presentation_development/presentation.pdf) | Mark wording you want to revise and practise explaining the data, model, and evaluation. Save any slide edits as a separate copy so they can be carried into the final decks. Figures and scores in these files are development material. |

## Open when the corresponding artifact is ready

| Task | Entry | What to save or send |
| --- | --- | --- |
| **6. Select F1/F2 designs** | Return here for the A/B previews once the figure brief and references are confirmed. | Your A/B choice for each figure and any requested corrections. |
| **7. Official 200-crop audit** | Use the command below once the final training manifest is frozen. | In `audit/audit.csv`, give all 200 rows `good` or `bad` and give every `bad` row a concrete reason. Use the numbered `audit/sheet_*.png` images and inspect the original source when needed. This is a separate review from the 30-pair check. |
| **8. Independent results check** | [Results worksheet](student_checks/03_results_reproduction.md) and the evaluation command below, after the final export and receipts are ready. | Your evaluator JSON, command and observations in [AGENT_LOG.md](AGENT_LOG.md). Check the final model, results table and evidence together. |
| **9. Final review, rehearsal and submission** | Use the final report, Beamer/PPTX decks, demo, and verified project ZIP when available. [Presentation and release plan](PROJECT_PLAN.md) · [README with live demo command](README.md) | Personally revise the final decks and check the name, ID, figures and conclusions. Do two timed rehearsals of the four-minute talk with live demo, then practise 1–2 minutes of questions. Upload `report.pdf` separately and the project ZIP to UMMoodle, and retain the receipt. |

Run these commands from the project root once the final artifact is ready. The audit wrapper runs the unchanged official tool, preserves the exact sample receipt and refuses to overwrite an existing audit directory.

```powershell
$studentId = Read-Host 'Actual student ID'
& .venvs\training\Scripts\python.exe -m src.evaluation.audit prepare --manifest submission\manifest.csv --student-id $studentId --data-root data --out audit
```

```powershell
$result = 'private_build/student_final_seed0.json'
if (Test-Path $result) { throw 'Choose a new result filename before repeating the evaluation.' }
& .venvs\training\Scripts\python.exe starter_kit\evaluate.py --submission submission --data-root data --device cuda --grading --n-per-class 1000 --batch-size 500 --seed 0 --out $result
```

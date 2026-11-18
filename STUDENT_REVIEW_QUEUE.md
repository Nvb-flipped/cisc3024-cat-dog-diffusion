# Student review and submission checklist

Calibration, Priority shortlist Batches 1–22, and their duplicate reviews are complete. The 60 approved additions from Batches 20–21 are in v008. The 30 approved additions from Batch 22 are in v009. Your usual review window is 12:00–13:00 GMT+8.

The 30 verdicts for Task 1 are saved. The remaining steps are selected source measurements, the official 30-row rebuild, and your account in `AGENT_LOG.md`.

The internal completion date is **November 23, Asia/Shanghai**, and the formal deadline is **November 26**. The selected development candidate is the constant-rate v009 attention model at 34k with FreeU strength 2.0, three official development scores of 12/15, and completed CPU/CUDA offline verification. The v007 export is preserved as a fallback. Complete the independent data check and figure choices while the final model is being selected. The official audit and final results rerun depend on the frozen submission.

Batch 22 crop and duplicate decisions are saved. The next pool expansion has 206 visually inspected agent drafts across [Batch 23, 74 crops](http://127.0.0.1:8766/?mode=shortlist&batch=23), [Batch 24, 63 crops](http://127.0.0.1:8766/?mode=shortlist&batch=24), and [Batch 25, 69 crops](http://127.0.0.1:8766/?mode=shortlist&batch=25). Check each exact preview, estimated width, occupancy and checklist, confirm the draft, correct geometry where feasible and save your own verdict. The findings are available for [Batch 23](http://127.0.0.1:8769/priority_shortlist_v023_findings.html), [Batch 24](http://127.0.0.1:8769/priority_shortlist_v024_findings.html) and [Batch 25](http://127.0.0.1:8769/priority_shortlist_v025_findings.html). Then use [Approved-only duplicate review](http://127.0.0.1:8767/) for the new approvals. These drafts have no training approval. The official 200-crop audit opens after the final manifest is frozen.

## Ready now

All **50 batches are now published**. Batches **26–50 add 420 visually inspected drafts (200 suggested cats, 220 suggested dogs)**, with estimated head widths, automatically calculated occupancy, checklists, attention reasons and matching findings. Start at [Head Study, Batch 26](http://127.0.0.1:8766/?mode=shortlist&batch=26) and use the Batch field through 50. [All batch findings and links](http://127.0.0.1:8769/priority_shortlist_expansion_findings.html) list each batch separately. Existing Batches 23–25 remain ready above. Training use will be decided after personal crop and family review; no new draft is a training approval.

At [Duplicate Families, Batch 26 preparation](http://127.0.0.1:8767/?batch=26#draft-batches), the **Prepared batch** selector lists 1–50. This read-only panel shows the exact drafts and suggested close matches, at most eight crops per page (four per row on desktop). Similarity does not establish duplicate or distinct families. Save crop verdicts in Head Study first, then click **Reload approved crops** in Duplicate Families and personally save family/keep choices in the approved section below. Edited geometry is compared again after approval. No family choices are saved by the preparation panel.

All scheduled follow-ups remain cancelled. Local API checks confirm 50 batches in both services and preserve the existing 1,462 personally approved crop records. Computer Use could not confidently determine the browser URL and stopped; browser visual verification remains uncompleted. Read-only endpoint, preview parity and DOM tests passed; no student verdict was submitted during verification.

| Task | Open | What to save or send |
| --- | --- | --- |
| **1. Independent data check** | [Instructions](student_checks/01_data_check.md) · [30-pair contact sheet](private_build/student_data_check_v001/contact_sheet.png) · [Full-size sources and crops](private_build/student_data_check_v001/images) · [Measurement CSV](private_build/student_data_check_v001/personal_measurements.csv) | Your verdict and species for all 30 pairs, measurements on selected difficult examples, the official rebuild output, and your account in [AGENT_LOG.md](AGENT_LOG.md). |
| **2. Independent model/interface check** | [Worksheet](student_checks/02_model_interface.md) · [Current candidate's diffusion code](private_build/v007_class_annealing010/diffusion.py) · [Model code](private_build/v007_class_annealing010/model.py) · [Sampler settings](private_build/v007_class_annealing010/inference.json) | You can do the scalar example and code walkthrough now. Save your interface probes and their output. If the export changes, repeat the probes on the final export and record the hashes and device in [AGENT_LOG.md](AGENT_LOG.md). |
| **3. Figure brief and reference choices** | [F1/F2 briefs](figures/review/briefs.md) · [Numbered reference sheet](figures/review/model_exemplar_candidates.png) | Reply with any factual corrections and 2–4 reference numbers, with a short note about what you like in each. F1 uses historical v001 counts. The current development candidate for F2 uses v007, 32k updates, DDIM50, CFG 3.25, PAG 0.15 and temporal sharpening 1.03. |
| **4. Name and ID** | [Course project plan](PROJECT_PLAN.md) | Use your official name and actual student ID for the report and the official audit. You can enter the ID locally when you run the commands. |
| **5. Read and practise with the current drafts** | [Development report](private_build/report_20261118_selection/report.pdf) · [Earlier PowerPoint draft](private_build/presentation_review/development.pptx) · [Earlier Beamer draft](private_build/presentation_development/presentation.pdf) | The report includes the November 18 model selection. The earlier slide drafts need the student's own update for the current model and results. Practise explaining the data, model and evaluation. Figures and scores are development material. |

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

# Independent data check — pending student work

This is a worksheet, not an AGENT_LOG entry. No student judgments, dates, measurements or outcomes have been filled in. The prepared 30 pairs deliberately include difficult proposals; do not treat them as an unbiased audit or as training acceptances.

1. Run `python scripts/prepare_student_data_check.py` once if the local pack has not been prepared. It refuses an existing directory so your measurements cannot be overwritten. Inspect `private_build/student_data_check_v001/contact_sheet.png`, then open each full-resolution `images/*_source.png` alongside its official/project 32px crops. The sheet enlarges 32px crops three times with nearest-neighbor display; the full source file retains original pixel dimensions.
2. Independently judge each pair against the handout. Do not trust detector labels or CLIP hints. Record your species/verdict and concrete reasons in `personal_measurements.csv`. Measure source head width and estimate occupancy on selected difficult examples using the full-resolution source. The proposed head width is deliberately omitted from the worksheet.
3. Trace one boundary crop and the 10-degree rotation probe through the original `starter_kit/catdog_eval/data.py`. Explain the coordinate origin, square center, rotation direction and interpolation. The rotation probe is a geometry exercise, not a proposed final training acceptance.
4. Independently run the untouched official reconstruction for these diagnostic rows in a new local folder. This is a 30-row study rebuild, not the official 200-row final audit:

```powershell
& .venvs\training\Scripts\python.exe starter_kit\audit.py --manifest private_build\student_data_check_v001\diagnostic_manifest.csv --student-id YOUR_ACTUAL_ID --n 30 --data-root data --out private_build\student_data_check_v001\your_official_rebuild
```

The official tool sorts its rows, so match source IDs and geometry rather than assuming its image index matches the worksheet case number. Check the original crop implementation and actual pixels yourself; an agent-produced equality result alone is not an independent check.

Record the actual context, your independent procedure, evidence files and outcome in AGENT_LOG.md when you perform the check. Agreement is a valid result. If you find a disagreement, describe the concrete example, the correction and the rule/test added. Do not invent a mistake or copy a first-person account written by an agent.

The final audit remains a separate task on the final manifest, with exactly 200 rows and your actual student ID.

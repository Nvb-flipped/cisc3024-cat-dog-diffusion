# Note on the evidence files in this folder

`evidence_outline.md`, `claims.csv`, `evidence_snapshot.json` and `source_manifest.json` are the **development-stage** evidence
records of the first report drafts (experiments up to about v009). They are kept unchanged, with their hashes, as a historical
record, and they are **not** the source of the final report.

The final report (`report.tex`, `report.pdf`) takes every number from the stored evaluator outputs through
`scripts/build_report_assets.py`, which writes `numbers.tex` and the figures (`python -m scripts.build_report_assets`).
The final personal audit, the three independent checks and the final evaluation that the old outline lists as pending are
complete: see `DATA_CARD.md`, `AGENT_LOG.md`, `results/final/` and `README.md`.

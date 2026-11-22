# CISC3024: conditional cat/dog head generation

For the November 19 private PC transfer, start with [MIGRATION_HANDOFF.md](MIGRATION_HANDOFF.md). Git contains development history and current personal review records; the private `migration-2026-11-19` release contains the large, hash-verified project snapshot. A clone alone omits data/checkpoints/private build files. Use the restore tool before rebuilding the environment. The historical introduction below describes v002; current model/data status is in the handoff and EXECUTION_STATUS.md.

This repository implements the pipeline in [PROJECT_PLAN.md](PROJECT_PLAN.md). The generator is a pixel-space conditional diffusion U-Net trained from scratch. The corrected v002 manifest contains 1,197 personally reviewed training crops and a separate 63-crop holdout. The original v001 manifest and its official results are retained for development comparisons. See [EXECUTION_STATUS.md](EXECUTION_STATUS.md) for the current model comparison and [STUDENT_REVIEW_QUEUE.md](STUDENT_REVIEW_QUEUE.md) for the pending personal checks.

The course pool is restricted to this course. Keep it local; do not push it or redistribute project materials. The supplied handout takes precedence over the starter kit. Labels are cat=0 and dog=1.

## Supported platforms and how to run the demo

**Reproduction environment.** Every number in the report, every command in the table below and the whole test suite were produced and checked on **Ubuntu Linux** (kernel 7.0, Python 3.11.16, PyTorch 2.9.1 with CUDA 12.8, one NVIDIA RTX 5090). Reproducing the project (data curation, training, evaluation) is supported on Linux only; training needs an NVIDIA GPU, curation additionally needs a vision-language model server and several GB of local models, and the course pool and starter kit must be placed locally (not part of the ZIP). The older PowerShell sections below come from the first, Windows-based phase of the project and have not been re-run since.

**Generation and the demo (`submission/`, `demo.py`)** need only Python, `torch`, `numpy` and `pillow` and no network or dataset. They run on the CPU or an NVIDIA GPU. They have been tested on Linux (CPU-only and GPU); Windows and macOS are expected to work but have **not been tested by the author**. Apple's `mps` backend is deliberately not used.

| | Linux | Windows | macOS |
|---|---|---|---|
| `demo.py`, CPU | tested | not tested | not tested |
| `demo.py`, NVIDIA GPU | tested | not tested | not applicable |
| full project (curation, training, evaluation) | tested | not supported | not supported |

Speed matters: one 16-image grid takes about half a second on the RTX 5090, but on a CPU it takes minutes (the sampler runs 150 network passes per image batch). A live demonstration therefore needs a GPU; a CPU run gives the same images, only much more slowly.

Run the demo from the unzipped project folder (Python 3.10 to 3.12; `requirements-demo.txt` lists the packages):

```text
# Linux / macOS
python3 -m venv demo-env && source demo-env/bin/activate
python -m pip install numpy pillow
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # macOS: python -m pip install torch
python demo.py --check          # interface self-check (many minutes on a CPU, seconds on a GPU)
python demo.py --seed 0         # one 16-image grid (8 cats, 8 dogs) saved under demo_output/
python demo.py                  # interactive: type a seed per grid, q to quit

# Windows (PowerShell)
py -3 -m venv demo-env
demo-env\Scripts\activate
python -m pip install numpy pillow
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python demo.py --check
```

With a GPU, install the CUDA build of torch from the PyTorch website instead of the CPU build; `demo.py` then uses it automatically.

**Windows with WSL2 and an NVIDIA GPU (the setup planned for the live demo, a laptop with an RTX 3060).** WSL2 is Linux, so the Linux commands apply; the GPU is passed through by the Windows NVIDIA driver. Not yet tested by the author (only Linux with another GPU was).

```text
# In Windows: install a current NVIDIA driver (it provides CUDA to WSL2) and WSL2 with Ubuntu:  wsl --install
# In the Ubuntu (WSL) terminal. Copy the unzipped project into the Linux file system, not /mnt/c (much faster), then:
cd ~/project
python3 -m venv demo-env && source demo-env/bin/activate     # if venv is missing:  sudo apt install python3-venv
python -m pip install numpy pillow torch                      # the default Linux torch wheel already contains CUDA (a large download)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"    # must print True and the GPU name
python demo.py --check          # a few seconds on the GPU
python demo.py                  # type 0, 0, 1 (a grid takes about a second), q to quit
```

If the last check prints `False`, torch cannot see the GPU: update the NVIDIA driver on Windows and run `wsl --shutdown`, then reopen Ubuntu. The model needs well under 2 GB of GPU memory. The same seed gives the same image on the same computer. A different CPU, GPU or torch version can change a few pixels by one grey level (measured on the CPU between torch 2.9.1 and 2.15.0: 384 of 352,160 pixels, each by 1 of 255), so the printed image hash can differ between computers while the pictures look identical; `--check` therefore tests behaviour, not hashes. `weights.pt` is checked against the hash in `submission/provenance.json` before anything is generated.

## Final model (v014): one command per report number

Current state: the submitted model is **v014, EMA 0.9995, 80,000 updates, sampler g2.75** (`submission/`, `manifests/v014.csv`, 8,029 rows). Decisions and evidence: `results/fullpool_20261119/DECISION_LOG.md` (entries 53-78), `EXPERIMENTS.md` (all 94 experiment folders), `results/final/` (official runs from `submission/`). Commands below run from the repository root on Linux (`python` = `.venvs/training/bin/python`; the older PowerShell sections further down document the first Windows-era setup). GPU jobs ran in restricted containers through `scripts/gpu_job.sh`, but each command also runs directly.

| Report item | Command | Output |
|---|---|---|
| Inputs unchanged (starter kit, pool) | `python scripts/setup_inputs.py --verify-only` | `results/setup/source_hashes.json` |
| 76,250 pool images decode | `python -m src.curation.integrity` | `results/curation/integrity.json` |
| Candidate registry and geometry rescue (63,115 candidates) | `python -m src.curation.fullpool --out results/fullpool_20261119`, then `python -m src.curation.rescue clip` and `python -m src.curation.rescue dino` | `results/fullpool_20261119/queue_stage4_*.csv` |
| Qwen3.5-9B screen (6,718 passed) | start the model server with `scripts/vllm_server.sh ~/models/qwen/Qwen3.5-9B qwen3.5-9b`, then `python -m src.curation.vlm_screen --queue QUEUE.csv --ledger LEDGER.jsonl --model qwen3.5-9b --model-revision c2022362` | `results/fullpool_20261119/vlm/*.jsonl` |
| Review ordering and re-crops | `python -m src.curation.rank_classifier embed`, `... rank`, `python -m src.curation.recrop --reviews LEDGERS --out results/fullpool_20261119/recrop_batchN.csv` | `recovery_ranked.csv`, `recrop_batch1..6.csv` |
| Agent visual review (17,174 crops) | contact sheets from `python -m src.curation.contact_sheet`, verdicts appended to ledgers by the agent | ledgers under `private_build/fullpool_qa/` (not in the ZIP; their SHA-256 are in `manifests/v014.provenance.json`) |
| Manifest (8,029 rows, 3,096 cat / 4,933 dog) | `python -m src.curation.freeze_reviewed --out manifests/v014.csv --reviews LEDGERS... --recrops results/fullpool_20261119/recrop_batch{1,2,3,4,5,6}.csv` | `manifests/v014.csv`, `.validation.csv`, `.provenance.json`, `results/fullpool_20261119/v014.candidate_status.csv` |
| Cache equals the official rebuild | `python -m scripts.run_v014 parity` | `results/curation/v014_cache_parity.json` |
| Single-batch overfit diagnostic (2,000 updates) | `python -m scripts.run_v010 diagnostic v010_attn16_flip` | `results/diagnostics/` |
| Training (160k updates, 1k checkpoints; the 80k checkpoint is submitted) | `python -m scripts.run_v014_variants train v014_ema9995_flip` | `checkpoints/v014_ema9995_flip_seed17/` |
| Export, sampler and official evaluation of the checkpoint | `python -m scripts.run_v014_variants screen v014_ema9995_flip --updates 80000 --sampler g275 --seeds 0 101 202 303 404` | `results/experiments/v014_ema9995_flip/u080000_g275/` |
| **Table 2 of the report** (official runs from `submission/`) | `python -m src.evaluation.run --submission submission --seed S --out results/final/seed_S.json` for S in 0 101 202 303 404 | `results/final/seed_*.json` |
| Same numbers with the course tool directly | `python starter_kit/evaluate.py --submission submission --grading --seed 0 --n-per-class 1000 --batch-size 500 --out my_seed0.json` | evaluator summary and JSON |
| 61-seed statistics, recall figure | `python -m scripts.build_report_assets` | `report/numbers.tex`, `report/fig_*.pdf` |
| Audit sample and result (160/200) | `python -m src.evaluation.audit prepare --manifest submission/manifest.csv --student-id ID --data-root data --out audit`, then label `audit/audit.csv`, then `... audit summarize ...`; breakdown: `python -m src.evaluation.audit_breakdown` | `audit/audit.csv`, `results/audit/breakdown.json` |
| Offline CPU and CUDA interface checks | `python -m src.release.check --submission submission --device cpu --out results/release/offline_cpu.json` (and `--device cuda`) | `results/release/offline_*.json` |
| Tests (216 pass, 1 skipped) | `python -m pytest -q tests starter_kit/tests` | terminal |
| Experiment table | `python -m scripts.build_experiments_index && python -m scripts.build_experiments_md` | `EXPERIMENTS.md` |
| Report PDF | `tectonic -X compile report/report.tex` | `report/report.pdf` |

The agent visual review is a judgement step, so the AI-found rows cannot be re-derived by rerunning code: they are reproducible from the recorded verdict ledgers and candidate tables. The 1,929 personally reviewed rows come from `annotations/` through `manifests/v010.csv`. Intermediate checkpoints that no result cites were pruned on 21 Nov 2026 (`results/pruning/ledger_*.json`); the final model and every cited checkpoint remain.

## Environment and original inputs

Commands below run from the repository root in PowerShell. The working environment uses Python 3.11.18, PyTorch 2.9.1+cu128 and an RTX 5060 Ti. FP32/BF16 convolution, attention and backward execution have been measured. Exact package versions are in `requirements-lock.txt`; PyTorch wheels come from the CUDA 12.8 index.

```powershell
# With an installed Python 3.11:
python -m venv .venvs\training
& .venvs\training\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& .venvs\training\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The first setup used a local uv bootstrap and a Python runtime inside `.venvs/runtimes/`; those are local tooling, not course submission files. Preserve the separate `.venvs/figgenie/` environment.

Place the supplied archives in `Course_Project_Resources/` and the existing pool index/22 compressed shards in `student/`. Unpack `starter_kit.zip` and `dev_set.zip` **at the root**, yielding `starter_kit/` and `data/dev/dev.npz`. Do not add an extra nesting directory. Then:

```powershell
& .venvs\training\Scripts\python.exe scripts\setup_inputs.py
& .venvs\training\Scripts\python.exe scripts\setup_inputs.py --verify-only
& .venvs\training\Scripts\python.exe scripts\smoke_environment.py
& .venvs\training\Scripts\python.exe -m pytest -q starter_kit\tests
```

`setup_inputs.py` hashes the originals, copies and verifies the compressed pool under `data/student/`, and verifies every supplied starter file against its archive. It does not expand image shards. Source hashes: `results/setup/source_hashes.json`. `.gitattributes` preserves exact manifest/JSON/ledger bytes because their hashes are provenance identifiers.

## Pool integrity and curation

```powershell
& .venvs\training\Scripts\python.exe -m src.curation.integrity
& .venvs\training\Scripts\python.exe scripts\setup_detector.py
& .venvs\training\Scripts\python.exe -m src.curation.build --config configs\curation.yaml
& .venvs\training\Scripts\python.exe -m src.curation.duplicates
& .venvs\training\Scripts\python.exe -m src.curation.review --port 8766
```

`setup_detector.py` is the only detector-download step. It records the immutable Hugging Face revision and hashes under `models/grounding-dino-tiny/provenance.json`; proposal generation checks those hashes and loads offline. A pilot spec fixes shuffled pool IDs, prompts and all nine threshold combinations. Use `--resume` only with that identical spec. Use a fresh `--out` for a changed experiment.

The completed pilot covers 2,000 images. Its union across all nine settings contains 3,352 proposals, of which 1,269 have valid crop geometry, across 697 sources. These are **not** accepted training heads. The diagnostic sheets show non-photographs and wrong species proposals. Detector confidence alone is insufficient.

Open `http://127.0.0.1:8766`. The student personally reviews the fixed 200 proposed crops and 200 rejected sources. Reviewers enter their own name; no judgments are prefilled. Good crops require the complete specification, confirmed source head width, occupancy and preview of the exact geometry. Decisions append to `annotations/decisions.jsonl`. Crop previews are generated by the same project crop function verified against `catdog_eval.data.crop`.

`duplicate_candidates.csv` contains raw proposal comparisons, not duplicate verdicts. After personally approving crops, use the approved-only family review:

```powershell
& .venvs\training\Scripts\python.exe -m src.curation.duplicate_review --port 8767
```

Open `http://127.0.0.1:8767`. Compare originals and exact crops, confirm family IDs, and explicitly choose keep=yes/no. Source, crop and mirrored-crop dHash only suggest comparisons; also check for duplicate families across comparison sets. One kept representative per family and per source is the conservative baseline. Exact duplicate crop pixels cannot both be kept, even under different family IDs.

Personal family events append to `annotations/duplicate_decisions.jsonl`; their materialized `annotations/duplicate_groups.csv` binds every decision to the current crop geometry, species and preview hash. Changing those invalidates the family decision. Existing manual CSV edits are preserved and require reconciliation before browser export. The old raw-proposal group template is diagnostic material, not the final reviewed-group file.

After calibration and a reviewed full-pool pass, freeze a new immutable version:

```powershell
& .venvs\training\Scripts\python.exe -m src.curation.freeze --config configs\curation.yaml --out manifests\v001.csv
```

This checks reviewed pixels, duplicate decisions and official geometry, creates a fixed class-stratified/group-disjoint 5% pool holdout, and records decision/manifest hashes. Baseline training requires at least 500 **training** heads per class after holdout. No dev image/feature is used for training or selection of pool images. No image augmentation is applied to the exact training targets.

To evaluate an expansion, freeze to a new output path and pass `--preserve-split-from manifests\v002.csv`. The prior training rows must still be personally approved and kept at the exact geometry. The prior holdout is copied byte for byte and the newly kept rows join the training split. This produced v003 with 1,205 training crops and the same 63-crop holdout.

After training builds its caches, independently compare their labels and pixels with a fresh official reconstruction. Use a fresh output path to preserve prior receipts.

```powershell
& .venvs\training\Scripts\python.exe -m scripts.verify_manifest_cache manifests\v002.csv manifests\v002.validation.csv --out results\curation\v002_cache_parity.json
```

## Correctness and training

```powershell
& .venvs\training\Scripts\python.exe -m pytest -q -m "not integration"
# Optional UI race checks with an installed Node.js runtime:
node --test tests/review_ui.test.js
# Run once in a fresh results directory; the diagnostic has a fixed 2,000-update budget:
& .venvs\training\Scripts\python.exe -m pytest -q -s --run-training tests\test_single_batch.py
& .venvs\training\Scripts\python.exe -m src.training.diagnostics benchmark
```

`manifests/diagnostic_two_heads.csv` contains two agent-inspected course-pool crops used **only** for integration/compute checks. The corrected labels and provenance are explicit. These are not student audit evidence, baseline data or a trained model suitable for submission. Diagnostics start from scratch, never transfer weights to a candidate, and retain no generator checkpoint. The probe must reduce fixed-probe noise MSE by at least 80% in exactly 2,000 updates and improve reconstruction. Failure triggers investigation; do not weaken the criterion after observing it.

The 500-update benchmark uses the actual selected architecture, BF16, effective batch 128, microbatch 64, AdamW and EMA, but only the diagnostic rows. Its speed estimates concern compute, not sample quality.

The completed 2,000-update diagnostic reduced fixed-probe noise MSE by 98.51%; reconstruction MSE fell from 0.5410 to 0.02934. The 500-update compute benchmark measured 0.1058 seconds/update after initialization, projecting about 1.47 hours for 50k updates on this device. That projection excludes evaluation, data preparation, interruptions and any full-dataset overhead; it is not an actual baseline training time.

```powershell
# Once the student-reviewed manifest exists:
& .venvs\training\Scripts\python.exe -m src.training.train --config configs\ddpm.yaml --manifest manifests\v001.csv --seed 17
# Resume only the same architecture, objective, seed and exact manifest:
& .venvs\training\Scripts\python.exe -m src.training.train --config configs\ddpm.yaml --manifest manifests\v001.csv --seed 17 --resume checkpoints\RUN\update_050000.pt --updates 100000
```

The baseline uses channels 48/96/144/192 and 10,664,371 parameters. The leading development candidate uses channels 32/64/96/128 and 5,082,147 parameters, with seed 17, 30,000 updates, EMA weights, 50 DDIM steps and guidance 2.5. The U-Net uses GroupNorm, timestep/class scale-shift conditioning, two residual blocks per level and low-resolution attention. Null conditioning is label 2. The cosine schedule has 1,000 steps and the loss predicts Gaussian noise. Guidance is `uncond + s*(cond-uncond)`, so s=1 is conditional prediction. DDIM uses eta=0 and recomputes noise after clipping reconstructed x0.

Reproduce the corrected v002 noise-prediction reference in a new checkpoint directory:

```powershell
& .venvs\training\Scripts\python.exe -m src.training.train --config configs\runs\86c3c843eb\capacity_32_seed17_u030000.yaml --manifest manifests\v002.csv --validation-manifest manifests\v002.validation.csv --seed 17 --updates 30000 --out checkpoints\capacity_32_v002_reproduction
```

The predeclared v002 velocity comparison changes the denoising target while holding the data, architecture, training seed, update budget and sampling protocol fixed. Its exact training command and input hashes are in `results/plans/velocity_32_seed17_63bc3fce35.json`. The matching diagnostic can be reproduced with `python -m src.training.diagnostics overfit --prediction-type velocity --out results/diagnostics/velocity_reproduction`.

The [v002 EDM plan](results/plans/edm_32_seed17_63bc3fce35.json) compares EDM preconditioning and deterministic Heun sampling with the 30,000-update epsilon/DDIM reference. It fixes the reviewed crops, holdout, U-Net, training seed, optimizer, EMA and update cap. Reproduce the two-crop integration check in a fresh directory:

```powershell
& .venvs\training\Scripts\python.exe -m src.training.edm_diagnostic --manifest manifests\v002.csv --out results\diagnostics\edm_reproduction
```

Run `python -m scripts.run_edm_comparison` to execute the frozen comparison or resume its verified checkpoints. The official results and decision receipt are under `results/experiments/edm_32_seed17_v002/`.

The [patch comparison](results/plans/patch_grid_v002.json) evaluates 4×4 and 8×8 packing with the same pixel-resolution refinement branch. It keeps the full-image epsilon objective, exact v002 targets, holdout and DDIM sampler. Run `python -m scripts.run_patch_comparison` to execute the frozen protocol or resume from its verified receipts. The diagnostic command is `python -m src.training.diagnostics overfit --patch-size 4 --patch-detail --out results/diagnostics/patch4_detail_reproduction`. Change `4` to `8` and use a fresh output folder for the other variant.

The [v006 coverage comparison](results/plans/v006_coverage.json) adds 99 personally approved, distinct crops to the exact v002 manifest and preserves the 63-row holdout. It reuses the seed-17, 30,000-update U-Net recipe and EMA/DDIM50/CFG 2.5. Reproduce the matched fit with the training command in that plan and a fresh `--out` directory. Export the 28k and 30k EMA checkpoints with `--steps 50 --guidance 2.5` and run the official wrapper with seed 0. The official results are under `results/experiments/v006_coverage/`. The 30,000-update result is the primary matched comparison. The original run directory's latest completion is now at 32k, with the matched 30k completion retained as its own segment receipt.

Rebuild the v002 development report grid from the first 16 official seed-0 outputs per class:

```powershell
& .venvs\training\Scripts\python.exe -m src.figures.photo_grid --result results/experiments/capacity_32_seed17_v002/seed_0.json --submission private_build/capacity_32_seed17_u030000_63bc3fce35 --out private_build/report_v002_reference_grid
```

The six-slide development talk has editable PowerPoint source in `presentation/build_development.mjs` and matching Beamer source in `presentation/presentation.tex`. The PowerPoint builder takes the bundled Node runtime directory containing `node_modules`, prepares exact seed-0 tiles locally, and exports into `private_build/presentation_development`. Compile the Beamer source from `presentation/` after building those tiles. Use the final model and verified personal checks when updating both versions for submission.

Compile the report with its bibliography and keep the auxiliary file from the same build. Record the physical page count and the six-page main-text check with `python -m scripts.verify_report_pages --pdf report/report.pdf --aux PATH_TO_BUILD/report.aux --out report/page_count.json`.

The follow-up Min-SNR experiment keeps the epsilon target and applies bounded timestep weights with gamma 5. Use the exact training command in `results/plans/minsnr5_32_seed17_63bc3fce35.json` and a fresh output directory. The 2,000-update correctness probe is `python -m src.training.diagnostics overfit --min-snr-gamma 5 --out results/diagnostics/minsnr_reproduction`. This probe records unweighted epsilon MSE on fixed noise and never exports weights.

Run metadata records the actual environment, code commit/dirty state, configuration, seed, manifest hash, counts, timing and parent checkpoint hash. Losses, exact optimizer/RNG state and EMA are retained locally. A manifest change requires a fresh final training lineage. Changing a dataset must not silently reuse a cache or checkpoint.

Each attempt has a separate `segment_*.json`, `loss_*.csv` and completion record. Checkpoints cannot be overwritten. Resuming a checkpoint older than an existing later checkpoint requires a fresh `--out`, preserving both branches. The segment records exact training-source hashes as well as the commit. If the matching `.validation.csv` exists, each checkpoint records fixed-noise raw/EMA losses on those disjoint pool sources; the holdout hash is fixed across resume. These are denoising diagnostics, not official quality scores.

Baseline training requires the frozen personal-review provenance, including its exact manifest and pool-content hashes. The training checkpoint retains that provenance hash; changed pool files cannot silently become a fresh cache for an old reviewed manifest.

The plan's three equal-budget learning-rate runs share seed 17 and 50,000 optimizer updates. The 2e-4 recipe uses the same output folder as the baseline above, so a verified completed baseline is reused. Previewing before the reviewed manifest exists reports a pending state and creates no fake checkpoints:

```powershell
& .venvs\training\Scripts\python.exe -m src.training.lr_comparison --manifest manifests\v001.csv
# Once data gates pass, run serially; interrupted jobs require an explicit checkpoint resume:
& .venvs\training\Scripts\python.exe -m src.training.lr_comparison --manifest manifests\v001.csv --run
# Later, create a fresh balanced quantity-ablation manifest from exact reviewed parent rows:
& .venvs\training\Scripts\python.exe -m src.curation.subset --parent manifests\v001.csv --out manifests\quantity_v001.csv
```

The quantity subset uses 1,000/class when available, otherwise 500/class, keeps the exact parent holdout bytes, and refuses a comparison with no size contrast. It requires fresh initialization and the matched 50k-update budget. Report the full-data class ratio and actual images seen; do not call a changed class ratio a pure quantity effect.

## Export and official evaluation

```powershell
& .venvs\training\Scripts\python.exe scripts\cache_evaluator.py
& .venvs\training\Scripts\python.exe -m src.release.export --run checkpoints\RUN\update_050000.pt --manifest manifests\v001.csv --out private_build\candidate_001 --guidance 1.0
& .venvs\training\Scripts\python.exe -m src.release.check --submission private_build\candidate_001 --device cpu
& .venvs\training\Scripts\python.exe -m src.release.check --submission private_build\candidate_001 --device cuda
$env:TORCH_HOME = Join-Path (Get-Location).Path 'models\evaluator'
& .venvs\training\Scripts\python.exe -m src.evaluation.run --submission private_build\candidate_001 --seed 0 --out results\final\seed_0.json
```

`--run` names an exact checkpoint file; it never silently chooses the latest. Export contains one module's parameters, one weight file, exact manifest, config and inference source. It rejects checkpoint/manifest mismatches and diagnostic weights. Candidate exports are immutable. The offline checker executes from a clean working directory with network disabled and project source removed from the inference import path. Its default counts are 1,2,20,50,500. CPU and GPU bitwise equivalence is not asserted.

The initial conditional baseline uses guidance 1.0 as specified in the staged plan. Later sampling comparisons vary guidance explicitly and preserve their own evaluations.

Sampling pads logical chunks to a fixed serialized `execution_batch` (initially 32) and keeps each output in the same physical slot when logical chunks change. Varying physical batch dimensions amplified differences in a CUDA stress test near the noisy cosine endpoint. A trained CPU probe found a maximum difference of 1.58e-5 when regrouping moved an output between slots. The corrected sampler exactly reproduced the trained CPU and CUDA probes with logical chunk sizes 32 and 7. Changing `execution_batch` requires reevaluation.

`configs/evaluation.yaml` has no invented checkpoint paths. Add actual candidates after training. `python -m src.evaluation.sweep --config configs/evaluation.yaml --plan` writes exact commands. Screening uses 250/class; promote at most three frozen sampler combinations to 1,000/class selection seeds 0/101/202. Confirm one frozen winner on 303/404. Keep every original official JSON/grid. Per-image dev features remain exclusively inside the unchanged evaluator.

Only `--submission --grading` with the pool present establishes manifest/memorization evidence. `--samples`, missing-pool runs or small-screening FID cannot certify release. Official result values must be traced to their JSON, and seed-0 is the headline row. No hidden-test performance can be promised.

The pixel-neighbor replay regenerates the exact official seed-0 batch sequence, checks the recorded memorization statistics, and saves the first four examples by requested class and flag status. Keep its training-crop previews local.

```powershell
& .venvs\training\Scripts\python.exe -m src.evaluation.neighbors --submission private_build\refinement_v001_seed17\u021000_ema_s50_g2.5 --result results\experiments\refinement_v001_seed17\u021000_ema_s50_g2.5\seed_0.json --out private_build\memorization_replay_v002\21k_cfg2.5
```

For the live demo, start the command before presenting and wait for `Model ready`. Enter seed `0` twice, then `1`, to generate the same class layout with repeated and changed seeds. The console prints elapsed generation time and the PNG hash.

```powershell
& .venvs\training\Scripts\python.exe -m src.release.demo --submission submission --out private_build\demo
```

## Student ownership and remaining deliverables

The student's own work is recorded in `AGENT_LOG.md` (three independent checks of the agent's data, results and interface work, with their evidence under `student_checks/evidence/`), `audit/audit.csv` (the personal 200-crop audit: 160 of 200 good) and `annotations/` (the 1,929 personally reviewed crops). `student_checks/` keeps the original check procedures.

The audit is generated only after the final manifest and the actual student ID are known:

```powershell
& .venvs\training\Scripts\python.exe -m src.evaluation.audit prepare --manifest submission\manifest.csv --student-id ACTUAL_ID --data-root data --out audit
# After the student personally completes every verdict and bad-row reason:
& .venvs\training\Scripts\python.exe -m src.evaluation.audit summarize --manifest submission\manifest.csv --student-id ACTUAL_ID --out audit
```

The student labels all 200 rows personally and gives a reason for every bad row; the result (80.0%, Wilson 95% interval 73.9-85.0%) is in `DATA_CARD.md`. The report, the slides, final packaging and rehearsal follow PROJECT_PLAN.md.

The audit wrapper invokes the unchanged official script, refuses an existing audit directory, records the manifest/tool hashes and student ID, and recomputes the exact official 200-row sample before summarizing. A manifest change invalidates this audit. Audit intervals are Wilson 95% intervals, with class-specific denominators retained. The wrapper checks file identity/completeness; it cannot certify who supplied the judgments.

`src.evaluation.run` preserves the unchanged evaluator's JSON and records the exact inference-file hashes before and after evaluation. Final release checks require matching sidecars, all five declared generation seeds, complete memorization, CPU/GPU offline checks, and matching report-page evidence. `src.release.package --list` previews the explicit course ZIP allowlist; building the ZIP refuses incomplete structural evidence. Excellent-tier shortfalls are recorded separately so they cannot be hidden by a structural pass.

After final packaging, test the actual ZIP with a new extraction directory. This uses its archived inference and checking code, the separately installed Python environment, and—when requested—the unchanged supplied starter kit, course data and evaluator cache. It does not copy raw course data into the ZIP:

```powershell
& .venvs\training\Scripts\python.exe -m src.release.package --list
# Only after all structural release gates pass:
& .venvs\training\Scripts\python.exe -m src.release.package --out private_build\project_submission.zip
& .venvs\training\Scripts\python.exe -m src.release.verify_archive private_build\project_submission.zip --extract-to private_build\clean_room_v001 --official-seed 0
```

The clean-room checker validates the archive member allowlist and hashes before extraction, rejects escaping paths or embedded photographs in text sources, and checks CPU/GPU inference from the extracted files. Its receipt binds the exact ZIP hash. The tiny synthetic archive test is software validation only; a final course archive has not yet been built.

The F4 checkpoint curves and F5 sampler sweep can be rebuilt directly from the retained data and official JSON:

```powershell
& .venvs\training\Scripts\python.exe -m src.figures.build --figure F4 --from-source-summary --out private_build\F4_rebuilt
& .venvs\training\Scripts\python.exe -m src.figures.build --figure F5 --from-source-summary --out private_build\F5_rebuilt
```

The F6 generated grid and F7 nearest-neighbor figure use the official seed-0 PNG and the corresponding local `src.evaluation.neighbors` replay. Their source tables and summaries record the fixed tile positions and individual image hashes. Use `python -m src.figures.photo_grid --help` to rebuild them from the matching evaluation and export. Photo grids, image tiles, and drawings with embedded course images are local artifacts.

No upload, cloud publication or course-data redistribution is performed by these commands.

# Sources, software and local assets

This course project does not redistribute the supplied pool, pretrained curation weights, metric weights, virtual environments, or installed skill corpora. Version pins are in requirements-lock.txt. Actual source/checkpoint hashes are retained in run metadata and the local models/*/provenance.json records.

## Methods and checkpoints

- DDPM noise prediction: Ho, Jain and Abbeel (2020), https://arxiv.org/abs/2006.11239.
- Deterministic DDIM sampling: Song, Meng and Ermon (2020), https://arxiv.org/abs/2010.02502. The implementation uses eta=0 and explicitly recomputes noise after clipping reconstructed clean pixels.
- Cosine schedule: Nichol and Dhariwal (2021), https://proceedings.mlr.press/v139/nichol21a.html.
- Classifier-free guidance: Ho and Salimans (2022), https://arxiv.org/abs/2207.12598. Our guidance s corresponds to 1+w in that paper's convention.
- EDM preconditioning and deterministic Heun sampling: Karras, Aittala, Aila and Laine (2022), https://arxiv.org/abs/2206.00364. The comparison implements the paper's equations using the existing U-Net and exact course crops.
- Grounding DINO: https://arxiv.org/abs/2303.05499. The pinned IDEA-Research/grounding-dino-tiny checkpoint revision is a2bb814dd30d776dcf7e30523b00659f4f141c71. Its locally retained model card declares Apache-2.0. Used for pool proposals only.
- CLIP: Radford et al. (2021), https://proceedings.mlr.press/v139/radford21a.html. The pinned openai/clip-vit-base-patch32 revision is 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. Used only as an exploratory pool photo/species review-ranking aid; not an automatic acceptance rule. The upstream CLIP repository license is MIT: https://github.com/openai/CLIP/blob/main/LICENSE.
- Official classifier, Inception/FID features, audit and evaluator belong to the supplied course toolkit and its upstream dependencies. They are used unchanged for evaluation/audit only. Metric-weight checksums are in results/setup/evaluator_weights.json. No metric weights are included in the submission ZIP.

## Runtime packages

Installed package metadata declares: PyTorch 2.9.1+cu128 (BSD-3-Clause); torchvision 0.24.1+cu128 (BSD); Pillow 11.3.0 (MIT-CMU); transformers 4.57.3 (Apache 2.0); PyYAML 6.0.3 (MIT); pytest 8.4.2 (MIT); huggingface-hub 0.36.0 (Apache). NumPy 2.2.6, SciPy 1.16.3, Matplotlib 3.10.7, pytorch-fid 0.3.0 and safetensors 0.7.0 retain their detailed installed license notices. The environment and package binaries are installed separately; this list does not replace their own notices. Figure conversion also uses the existing FigGenie environment's python-pptx/Pillow/lxml/font tooling.

## Skills, artwork and fonts

- FigGenie is the first design reference and supplies the local SVG-to-native-PPTX converter. The installed helper and its corpora are excluded from the ZIP. Its bundled AI figure references belong to their paper authors/publishers; the gallery's MIT license covers its index, not arbitrary reuse of the figures. The ten candidate references in figures/review/exemplars.json are for personal design study only. No reference artwork is copied into F0.
- Scientific Writing and Scientific Visualization procedures from Scientific Agent Skills materially informed evidence handling and plotting. Citation: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, https://doi.org/10.48550/arXiv.2609.00065. The citation is retained in report/references.bib. Named student source verification remains pending.
- Other materially consulted local workflows: scientific-figure-making, frontend-design, simplify, PowerShell expertise, computer use, PDF reading and presentation guidance. Installed SKILL.md files remain local; no skill corpus is redistributed.
- F0 plot text uses the system-installed Arial font. Font files are not embedded or redistributed. Recreating the exact typography requires a lawful installed copy. Do not silently substitute a font and claim unchanged geometry.
- Report/figure code and project-specific native-vector mappings were authored for this project with coding-agent assistance. The full final disclosure and authentic student checks remain in progress.

Detailed native-format regeneration currently depends on the locally installed FigGenie helper at `.agents/skills/figgenie-paper-diagram/scripts/svg2pptx.py`; it is an external authoring dependency. Keep its hash/version with export QA. Do not copy the helper into a distributable package without confirming its applicable license. The generated editable PPTX itself is retained as a project deliverable.

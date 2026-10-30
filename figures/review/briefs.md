# F1 and F2: factual design briefs for student review

Status: proposed briefs; student confirmation and exemplar choices are pending. No A/B architecture drawing is approved. These briefs follow the FigGenie-first workflow in PROJECT_PLAN.md. Reference images are for studying structure, not for redistribution or copying artwork.

## F1 — From the supplied pool to a replayable training manifest

The figure should explain how the project turns the supplied image pool into a defensible, reproducible set of training heads. Its main reading direction follows the data: immutable compressed course pool → integrity checks → frozen pretrained curation tools → exact crop preview → personal student judgments and duplicate-family decisions → immutable training/holdout manifests. Put the actual completed counts at their corresponding stages: 76,250 sources; zero integrity failures; 125,941 Grounding DINO proposals; 47,855 valid square proposals from 26,925 sources. These last numbers describe proposals, not accepted training images. Keep accepted counts visibly pending until personal review has produced evidence.

Grounding DINO and the optional CLIP photo/species ranking aid are frozen curation tools. CLIP scores rank review cards; they are neither calibrated probabilities nor automatic acceptance decisions. The student checks real photography, species, one centered upright head including ears, source head width, occupancy, and crop geometry. Personal duplicate decisions keep one crop per family. The exact official crop operation produces the 32 × 32 RGB targets; the manifest records source ID, geometry, rotation, and label. A grouped 5% pool holdout stays separate from optimizer training.

Show development evaluation in a distinct lane. The untouched evaluator receives generated samples and the development reference; its classifier and Inception features never flow into training or pool curation. Use one consistent identity for course data, frozen tools, student decisions, and trainable model inputs. Avoid miniature screenshots or invented stage yields. The caption must say that curation and acceptance are still in progress.

## F2 — Conditional pixel diffusion, training and sampling

The figure should connect the implemented learning objective to the actual generation path. Show a 32 × 32 RGB training crop receiving Gaussian noise at a sampled timestep, followed by one trainable conditional U-Net that predicts the added noise. A timestep embedding and a cat/dog embedding condition residual blocks through scale and shift. Null class 2 supports classifier-free guidance; training independently drops class conditioning with probability 0.10. The objective is mean squared error between predicted and sampled Gaussian noise. The generator starts from random initialization. There is no pretrained image encoder, latent representation, or decoder in this model.

The U-Net has resolutions 32, 16, 8, and 4, channels 48, 96, 144, and 192, two residual blocks per level, skip connections, GroupNorm and SiLU, and attention at low resolution. It contains 10,664,371 parameters. Show EMA as a maintained copy of weights, not another jointly trained generator. The default configuration uses a 1,000-step cosine training schedule; 100 deterministic DDIM sampling steps and guidance 1.5 are initial settings, not a selected winning sampler.

In a separate sampling strip, private seeded Gaussian noise passes through repeated denoising with cat/dog or null predictions from the same selected weight set. Guidance is ε_uncond + s(ε_cond − ε_uncond), so s = 1 is ordinary conditional prediction. DDIM uses eta = 0, clips reconstructed clean pixels, and recomputes noise after clipping. The output is RGB in [0,1]. A short annotation can explain fixed physical execution batches for label isolation and reproducibility. Avoid claims about sample quality before full official evaluation.

## Reference choices to make

Review `model_exemplar_candidates.png` at a readable zoom. Choose 2–4 numbered examples whose reading structure, grouping, or visual hierarchy you prefer. Choices guide design; they do not imply copying text, graphics, or results. The ten candidates are indexed in `exemplars.json`. After brief confirmation and these choices, A/B alternatives will use the same implemented facts and equal display width, and the student's selected version will become the canonical design.

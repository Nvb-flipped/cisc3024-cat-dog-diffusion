# F1 and F2: factual design briefs for student review

These proposed briefs describe the frozen v001 development data and the current leading model. Confirm the briefs and pick exemplars before we prepare A/B layouts. Reference images are for studying structure.

## F1 — From the supplied pool to a replayable training manifest

Follow the data from the compressed course pool through integrity checks, detector proposals, exact crop previews, student review, duplicate-family decisions, and the frozen manifests. Label the proposal stages with 76,250 decoded sources, zero integrity failures, 125,941 Grounding DINO proposals, and 47,855 valid square proposals from 26,925 sources. For the reviewed v001 split, show 1,260 retained crops, with 1,197 used for training (583 cats and 614 dogs) and 63 held out (31 cats and 32 dogs).

Grounding DINO and the optional CLIP photo/species ranking aid are frozen curation tools. CLIP scores rank review cards; they are neither calibrated probabilities nor automatic acceptance decisions. The student checks real photography, species, one centered upright head including ears, source head width, occupancy, and crop geometry. Personal duplicate decisions keep one crop per family. The exact official crop operation produces the 32 × 32 RGB targets; the manifest records source ID, geometry, rotation, and label. A grouped 5% pool holdout stays separate from optimizer training.

Show development evaluation in a distinct lane. The unchanged evaluator receives generated samples, the development reference, and reconstructed training crops for memorization checks. Use consistent colors or labels for course data, frozen tools, student decisions, and model inputs. The caption should identify the counts as v001 development data and state that the crop recheck and official audit are pending.

## F2 — Conditional pixel diffusion, training and sampling

The figure should connect the implemented learning objective to the actual generation path. Show a 32 × 32 RGB training crop receiving Gaussian noise at a sampled timestep, followed by one trainable conditional U-Net that predicts the added noise. A timestep embedding and a cat/dog embedding condition residual blocks through scale and shift. Null class 2 supports classifier-free guidance; training independently drops class conditioning with probability 0.10. The objective is mean squared error between predicted and sampled Gaussian noise. The generator starts from random initialization. There is no pretrained image encoder, latent representation, or decoder in this model.

Show the leading development U-Net with resolutions 32, 16, 8, and 4, channel widths 32, 64, 96, and 128, two residual blocks per level, skip connections, GroupNorm and SiLU. Spatial attention appears at 8 × 8, 4 × 4, and the bottleneck. The model has 5,082,147 parameters and uses a 1,000-step cosine training schedule. Show EMA as a maintained copy of the U-Net weights. The leading checkpoint used training seed 17 and 30,000 updates. Its sampler uses 50 deterministic DDIM steps and guidance 2.5.

In a separate sampling strip, private seeded Gaussian noise passes through repeated denoising using cat/dog and null predictions from the same EMA weights. Guidance is ε_uncond + s(ε_cond − ε_uncond), so s = 1 is ordinary conditional prediction. DDIM uses eta = 0, clips reconstructed clean pixels, and recomputes noise after clipping. The output is RGB in [0,1]. A short annotation can explain how fixed execution slots preserve reproducibility across batch compositions. Identify this configuration as the leading development candidate until the final export is evaluated.

## Reference choices to make

Review `model_exemplar_candidates.png` at a readable zoom. Choose 2–4 numbered examples whose reading structure, grouping, or visual hierarchy you prefer. Choices guide design; they do not imply copying text, graphics, or results. The ten candidates are indexed in `exemplars.json`. After brief confirmation and these choices, A/B alternatives will use the same implemented facts and equal display width, and the student's selected version will become the canonical design.

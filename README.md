# Representational vs. behavioral shape bias in vision-language models

Code for the paper's experiments. It measures how strongly three vision-language models
(LLaVA-1.5-7B, PaliGemma2-10B-mix, Qwen3-VL-8B) *behave* shape-biased on cue-conflict
images, and compares that with how much shape and texture information linear probes can
read from the models' language-side hidden states.

The repository contains code only. Every stimulus, hidden state, table and figure is
regenerated from two input folders.

## Inputs

| Variable | What it is |
|---|---|
| `IMAGENET_VAL_DIR` | ImageNet (ILSVRC2012) validation images, one subfolder per WNID (`n01440764/`, ...). |
| `GEIRHOS_STIMULI_DIR` | `stimuli/style-transfer-preprocessed-512` from [rgeirhos/texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape). Used only for the random-vs-grouped split comparison. |
| `OUTPUT_DIR` | Where all generated artifacts go (default `outputs/`). |
| `MODELS_ROOT_PATH` | Optional directory of local model snapshots laid out as `<org>/<name>`; if empty, weights are downloaded from the Hugging Face Hub. |
| `HF_TOKEN` | Optional Hugging Face token; needed for the gated `google/paligemma2-10b-mix-224`. |

## Setup

```bash
conda env create -f environment.yaml
conda activate rbb
cp configs/paths.env.example configs/paths.env   # then edit the paths
```

## Reproducing

```bash
scripts/reproduce.sh            # everything
scripts/reproduce.sh probe      # a single stage
python -m rbb.pipeline --list all   # print every command without running it
```

Stages run in this order. Each one skips outputs that already exist, so an interrupted
run resumes where it stopped.

| Stage | Module | Output (under `$OUTPUT_DIR`) |
|---|---|---|
| `imagenet16` | `rbb.stimuli.imagenet16` | `imagenet16/`: 1450 validation images of the 16 classes |
| `stimuli` | `rbb.stimuli.cue_conflict` | `stimuli/main_l<λ>/` (1200 stimuli per λ), `stimuli/crop_l<λ>/` (300, crop control), `stimuli/style_crops/` |
| `ceilings` | `rbb.stimuli.ceilings` | `ceiling_stimuli/{shape_photos,texture_photos,white_canvas_l<λ>}/` |
| `extract` | `rbb.extract`, `rbb.analysis.recognizability` | `hidden_states/<condition>/<model>/`: per-image `.npy` (layers × dim), `answers.csv`, `shape_bias.json` |
| `probe` | `rbb.probing` | `probes/folds/*.json`, `probes/<condition>/<model>/f1_metrics.json` |
| `analyses` | `rbb.analysis.*` | `results/*.csv`, `results/*.json` |
| `figures` | `rbb.figures.*` | `figures/*.pdf`, `figures/*.png` |
| `validate` (optional) | `rbb.analysis.validate_probe` | `results/torch_vs_sklearn_validation.json` |

**Compute.** Tested on one 16 GB GPU with 24 GB of CPU RAM for offloading. Stimulus
generation (Gatys style transfer, 500 LBFGS iterations per image) and hidden-state
extraction take about 15–20 GPU-hours in total. The later stages take minutes.

**Determinism.** Given the same images, hidden states, answers, probes and statistics are
reproduced exactly. Stimulus generation is not bit-exact on GPU: cuDNN nondeterminism is
amplified over 500 LBFGS iterations, so two runs of the generator differ at the pixel
level (the shape/texture pairs and all settings are identical). Numbers computed on
regenerated stimuli therefore match the paper up to this sampling noise.

### Where each paper result comes from

| Paper item | File | Produced by |
|---|---|---|
| Stimulus examples | `figures/stimuli.pdf` | `rbb.figures.stimuli` |
| Appendix stimuli (white canvases, crop control) | `figures/appendix_stimuli.pdf` | `rbb.figures.appendix` |
| Shape bias and probe F1 vs. λ | `figures/scale_sweep_combined.pdf`, `results/correlation_stats.json` | `rbb.figures.curves`, `rbb.analysis.correlation` |
| Answer breakdown | `figures/answer_breakdown.pdf`, `results/answer_breakdown.csv` | `rbb.analysis.behavior` |
| Probe ceilings | `results/ceilings.csv` | `rbb.analysis.ceilings` |
| Within-condition AUROC | `results/within_condition_auroc.csv`, `figures/margin_distribution_example.pdf` | `rbb.analysis.within_condition last-layer` |
| AUROC across depth | `results/within_condition_by_layer.csv`, `figures/within_condition_by_layer.pdf` | `rbb.analysis.within_condition depth` |
| Crop control | `results/crop_control_*.csv` | `rbb.analysis.crop_control` |
| Style recognizability | `results/style_recognizability.csv` | `rbb.analysis.recognizability` |
| Stimulus distances | `results/stimulus_distances.csv` | `rbb.analysis.stimulus_distances` |
| Measurement protocol (random vs. grouped split) | `results/protocol_split_comparison.csv` | `rbb.analysis.protocol` |

## Method details that are fixed in code

- **Stimuli.** Gatys et al. style transfer with torchvision's VGG19. Style layers are
  relu1_1 to relu5_1 (Gram matrices), the content layer is relu4_2, images are 256 px,
  and λ is the style-weight scale. Shape/texture pairs are listed in
  `src/rbb/resources/cue_conflict_pairs.csv`. The generator checks that it reproduces
  that list.
- **ImageNet subset.** The exact 1450 files are listed in
  `src/rbb/resources/imagenet16_manifest.csv`. `sample_manifest()` in
  `rbb/stimuli/imagenet16.py` documents how the list was drawn (up to 100 per class, seed 0).
- **Models.** Weights are loaded in bfloat16, and image inputs are cast to float16. The
  answer is the argmax next token of the language-model head at the last position,
  decoded and matched to the answer letters A–P.
- **Probes.** Multinomial logistic regression (L2, C = 1) is fit with batched LBFGS in
  PyTorch on standardized features, one probe per layer. Cross-validation is 3-fold
  StratifiedGroupKFold, grouped by source image (seed 42). One fold assignment per
  stimulus set is shared across models, λ values and layers.

## Attribution

The WNID lists of the 16 categories (`src/rbb/resources/imagenet16_wnids.json`) are taken
from Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing shape
bias improves accuracy and robustness* (ICLR 2019),
[rgeirhos/texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape). Their
cue-conflict stimulus set is used as an input, not redistributed.

## License

MIT, see [LICENSE](LICENSE).

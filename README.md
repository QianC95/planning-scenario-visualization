# Planning scenario visualization benchmark

This repository accompanies [*Generative Artificial Intelligence for Planning Scenario Visualization: A Framework and Benchmark for Controllable Street-Level Image Synthesis*](https://doi.org/10.1111/tgis.70431). It contains metadata for the 7,120 paired images (5,105 train; 2,015 test), the published evaluation prompts and results, and code to evaluate model outputs. **It contains no Street View images or image-derived condition maps.** The paper's Data Availability Statement says that street-level images are subject to third-party licensing restrictions and are not redistributed. Without lawful access to the images, the public files allow inspection of the study design and published results, but cannot recompute the image-based metrics in Tables 3 and 4.

## Contents

| Path | Purpose |
| --- | --- |
| `metadata/train_pairs.csv`, `metadata/test_pairs.csv` | Final split, image references, text annotations, and panorama metadata; no images |
| `metadata/schema.md` | Field definitions and provenance |
| `src/evaluate.py` | Evaluates generated images against the full test split by `sample_id` |
| `config/prompts.json` | Model-specific and four planning-oriented evaluation prompts from Tables 2 and 4 |
| `results/published_table3.csv`, `results/published_table4.csv` | Values transcribed from the paper, not recomputed by this release |

The metadata files are the public release artifacts. The authors' private data preparation code and restricted source imagery are outside this repository. See [schema and provenance](metadata/schema.md).

## Evaluate generated images

Install an appropriate PyTorch build for your hardware, then the packages in `requirements-eval.txt`. The evaluator requires the 2,015 before and after images named by `metadata/test_pairs.csv`, plus one generated image for every test `sample_id`. It does **not** download or reconstruct the restricted images. The `--image-root` directory should contain the relative paths listed in the metadata, such as `before_sq/...` and `after_sq/...`. It defaults to CPU and a batch size of four, and asks torch-fidelity to conserve CPU memory; opt in to GPU use with `--device cuda` when resources permit.

Create a prediction CSV with columns `sample_id,generated_path` (see `metadata/predictions_template.csv`). `generated_path` is relative to `--predictions-root`. It may follow any naming convention; the ID is the join key. Paths must remain below their specified roots. Predictions must cover exactly the test IDs, with no duplicates.

```bash
python src/evaluate.py \
  --model controlnet_canny_finetuned \
  --metadata metadata/test_pairs.csv \
  --predictions /path/to/predictions.csv \
  --image-root /path/to/restricted-images \
  --predictions-root /path/to/generated-images \
  --output /path/to/evaluation.json
```

Use a key from `config/prompts.json` for `--model`. The output JSON contains LPIPS, FID, KID, model-specific CLIP/ΔCLIP, and the four planning-oriented CLIP/ΔCLIP evaluations. ΔCLIP is omitted for PITI and T2I-Adapter guide because those workflows have no prior-state image input. The evaluator resizes images to 512 × 512 for CLIP and for most LPIPS evaluations; PITI LPIPS uses 256 × 256 to match its original 256-pixel outputs. It uses VGG LPIPS and `openai/clip-vit-base-patch16`, and computes FID/KID from the original generated and target files. It reports mean and sample standard deviation for LPIPS and mean and standard deviation for KID. All expected files are checked before any metric is computed.

For a close numerical comparison with the paper, use the same image set, generated outputs, software versions, and preprocessing. The paper reports one output per input with fixed seeds; those outputs and all model weights are outside this release. Commercial model outputs can also change across service versions. See [reproduction limits](DATA_AVAILABILITY.md).

## Published experiment

The paper evaluates 11 settings across PITI, T2I-Adapter (guide and preserve), InstructPix2Pix, DiffEdit, Nano Banana, and five ControlNet conditions. Appendix A describes the 4 × NVIDIA RTX A5500 workstation, 512 × 512 evaluation resolution, and fine-tuned ControlNet's LoRA training (Adam, learning rate 2 × 10⁻⁶, 6,000 steps). The prompts and reported metrics are in this repository; upstream model implementations and pretrained weights should be obtained from their original providers under their respective terms.

## Citation

Cao, Q., Kurtz, H., Calabria, J., Vick, R. A., Yao, X. A., & Mai, G. (2026). Generative Artificial Intelligence for Planning Scenario Visualization: A Framework and Benchmark for Controllable Street-Level Image Synthesis. *Transactions in GIS, 30*(7), e70431. [https://doi.org/10.1111/tgis.70431](https://doi.org/10.1111/tgis.70431)

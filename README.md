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

Install `requirements-eval.txt` and supply the restricted before/after images and one generated image per test pair. List the generated images in a CSV with `sample_id,generated_path` columns ([template](metadata/predictions_template.csv)).

```bash
python src/evaluate.py \
  --model controlnet_canny_finetuned \
  --metadata metadata/test_pairs.csv \
  --predictions /path/to/predictions.csv \
  --image-root /path/to/restricted-images \
  --predictions-root /path/to/generated-images \
  --output /path/to/evaluation.json
```

Choose `--model` from `config/prompts.json`. The command writes CLIP, ΔCLIP, LPIPS, FID, and KID results to JSON. See [data availability](DATA_AVAILABILITY.md) for the image access limits.

## Published experiment

The paper evaluates 11 settings across PITI, T2I-Adapter (guide and preserve), InstructPix2Pix, DiffEdit, Nano Banana, and five ControlNet conditions. Appendix A describes the 4 × NVIDIA RTX A5500 workstation, 512 × 512 evaluation resolution, and fine-tuned ControlNet's LoRA training (Adam, learning rate 2 × 10⁻⁶, 6,000 steps). The prompts and reported metrics are in this repository; upstream model implementations and pretrained weights should be obtained from their original providers under their respective terms.

## Citation

Cao, Q., Kurtz, H., Calabria, J., Vick, R. A., Yao, X. A., & Mai, G. (2026). Generative Artificial Intelligence for Planning Scenario Visualization: A Framework and Benchmark for Controllable Street-Level Image Synthesis. *Transactions in GIS, 30*(7), e70431. [https://doi.org/10.1111/tgis.70431](https://doi.org/10.1111/tgis.70431)

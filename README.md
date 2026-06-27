# Malaria Cell Classification with Vision Transformers

Fine-tuning three Vision Transformers (ViT-B/16, Swin-T and DeiT-S) to classify parasitised versus
uninfected red blood cells from the NIH/LHNCBC malaria dataset, evaluated under a leakage-resistant
patient-level protocol. The project reports both predictive performance and computational efficiency.

## Results (held-out test set, patient-level split)

| Model | Accuracy | F1 | ROC-AUC | Params (M) | Latency (ms/img) | GFLOPs |
|---|---|---|---|---|---|---|
| DeiT-S | 0.966 | 0.966 | 0.994 | 21.7 | 7.4 | 4.25 |
| Swin-T | 0.971 | 0.971 | 0.996 | 27.5 | 14.1 | 4.51 |
| ViT-B/16 | 0.973 | 0.973 | 0.996 | 85.8 | 19.3 | 16.87 |

All three models exceed the roughly 0.89 accuracy reported for CNNs under comparable patient-level
evaluation, while differing by around four times in computational cost. See `docs/results.md` for the full
experiment record and `docs/figures/` for the figures.

## Dataset

The NIH/LHNCBC malaria dataset (27,558 segmented single-cell images, balanced between parasitised and
uninfected). Download the archive from the NIH public repository
(`https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip`) and unzip it so that the following
folders exist:

```
data/raw/Parasitized/*.png
data/raw/Uninfected/*.png
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.12.

```bash
uv sync
```

## Usage

```bash
# 1. Build the patient-level train/validation/test split (writes data/split.csv)
uv run python scripts/build_split.py

# 2. Train and evaluate the models (all three, or pass keys: deit_s swin_t vit_b16)
uv run python scripts/train_eval.py

# 3. Aggregate the per-model results into a comparison table and figures
uv run python scripts/build_results.py
```

The notebooks in `notebooks/` mirror this workflow: data audit, training, evaluation and results.

## Method notes

- **Patient-level split.** A 70/15/15 split grouped by patient, so no patient appears in more than one
  partition. This prevents the data leakage that inflates accuracy when cells from the same patient are
  split at random.
- **Transfer learning.** ImageNet-pretrained backbones from `timm`, fine-tuned with PyTorch Lightning
  using `bf16` mixed precision and gradient accumulation to fit a 6 GB GPU.
- **Metrics.** Accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix for performance;
  parameter count, FLOPs and inference latency for efficiency.

## Layout

```
src/malaria_vit/   importable package (config, data, models, lit_module, train, metrics, evaluate, efficiency)
scripts/           build_split.py, train_eval.py, build_results.py
tests/             pytest suite (incl. the no-leakage split test)
notebooks/         01 data audit, 02 train, 03 evaluate, 04 results
docs/              results.md, comparison.csv, figures/
```

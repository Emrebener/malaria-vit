# Experiment record

> Factual record of the experiments: dataset audit, split, training configuration and results.

## Environment
- GPU: NVIDIA RTX 3060 Laptop (6 GB VRAM). Python 3.12 (uv). PyTorch 2.6.0+cu124, `timm`, PyTorch Lightning.
- Mixed precision: `bf16-mixed`; small per-model batch + gradient accumulation to fit 6 GB; fixed seed (42).

## Dataset
- NIH/LHNCBC malaria cell dataset (downloaded from the NIH public link), 27,558 segmented single-cell images, exactly balanced (13,779 parasitised / 13,779 uninfected).

## Dataset audit (key findings)
- **200 distinct patients**, recovered from the leading `C<number>` in each filename — matching the known cohort of **150 *P. falciparum*-infected + 50 healthy** patients.
- **150 patients contribute cells to *both* classes** (infected patients provide both parasitised and normal-looking cells); the 50 healthy patients provide only uninfected cells.
- *Why this matters:* a random image-level split would place the **same patient** in train and test simultaneously — in both classes — inflating accuracy. This is the concrete justification for patient-level splitting.
- Filename parsing note: the patient identifier is the leading `C<number>` (e.g. `C39P4thinF…`→`C39`, `C225ThinF…`→`C225`, `C51AP12…`→`C51`); the `P` sub-code is not a reliable patient key (3,552 uninfected files have no `P` part).

## Patient-level split
- Patient-grouped, class-stratified, **70/15/15**, seed 42, saved as `data/split.csv`.
- **Verified on real data: 0 patients span splits** (no leakage). 200 patients → **140 / 30 / 30** (train/val/test).
- Cell counts per split: train 19,897 · val 3,587 · test 4,074 (vary by patient cell-count; test set well balanced: 2,049 uninfected / 2,025 parasitised).

## Protocol
- Input 224×224, ImageNet normalisation. Train-time augmentation: H/V flips, small rotation, mild colour jitter (stronger for DeiT-S).
- Transfer learning from ImageNet weights, binary head; AdamW, early stopping on validation loss, best-validation checkpoint used for the single held-out test evaluation.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix (performance) + parameters, latency, FLOPs (efficiency).

## Training configuration and convergence
Shared across all three: AdamW, lr 3e-5 (constant, no schedule), PyTorch default weight decay (0.01), cross-entropy loss,
`bf16-mixed`, max 15 epochs, early stopping on validation loss (patience 3), best-validation-loss checkpoint, seed 42.

| Model | Real batch | Accum | Effective batch | Aug | Epochs run | Best epoch (1-idx) | Val acc @ best |
|---|---|---|---|---|---|---|---|
| DeiT-S | 32 | 2 | 64 | strong jitter | 4 | 1 | 0.965 |
| Swin-T | 24 | 3 | 72 | standard | 7 | 4 | 0.967 |
| ViT-B/16 | 12 | 5 | 60 | standard | 10 | 7 | 0.967 |

Two caveats:
- **Effective batch is not identical (60 to 72).** The real batch was sized to fit 6 GB of VRAM, so the three models were
  not trained under perfectly matched conditions.
- **DeiT-S hit its best validation loss after a single epoch.** A well-pretrained backbone converges almost immediately on
  this dataset; no model used its 15-epoch budget.

## Model results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params (M) | Latency (ms) | GFLOPs |
|---|---|---|---|---|---|---|---|---|
| DeiT-S | 0.966 | 0.978 | 0.954 | 0.966 | 0.994 | 21.7 | 7.4 | 4.25 |
| Swin-T | 0.971 | 0.965 | 0.977 | 0.971 | 0.996 | 27.5 | 14.1 | 4.51 |
| ViT-B/16 | 0.973 | 0.980 | 0.966 | 0.973 | 0.996 | 85.8 | 19.3 | 16.87 |

Benchmark for context (published, patient-level): CNNs ~0.89 (Laghari et al., 2025).

## Key takeaways
- **All three ViTs reach 0.966–0.973 accuracy / 0.994–0.996 ROC-AUC** on held-out (unseen) patients under the verified leakage-free protocol — above the ~0.89 CNN patient-level benchmark (Laghari et al., 2025) and in line with Rajaraman et al. (2018) single-model patient-level (0.959 / AUC 0.991). Under clean evaluation, ViTs match/beat the CNN baselines — the gap this project set out to test.
- **Accuracy ranking is tight:** ViT-B/16 (0.973) ≳ Swin-T (0.971) > DeiT-S (0.966) — a spread of only ~0.7 points.
- **Efficiency differs sharply:** DeiT-S is lightest (21.7 M params, 7.4 ms, 4.25 GFLOPs); ViT-B/16 costs ~4× the parameters, ~2.6× the latency and ~4× the FLOPs for +0.7 points of accuracy. **Swin-T is the best balance** — near-top accuracy at ~1/3 of ViT-B/16's parameters and ~1/4 of its FLOPs.
- **Deployment implication:** the lightweight transformers (DeiT-S, Swin-T) deliver almost all the accuracy at a fraction of the compute — directly relevant to low-resource, point-of-care settings.
- Patient-level evaluation established as leakage-free on the real data (0 patients spanning splits); the 150-patients-in-both-classes finding makes leakage-resistant evaluation essential.
- Sanity check passed: AUC 0.994–0.996 (not 1.00), balanced confusion matrices — credible classifiers, not leakage artefacts.

## Figures & artefacts
- `comparison.csv` — the full comparison table.
- `figures/performance.png` — accuracy / F1 / ROC-AUC per model, with the ~0.89 CNN baseline line.
- `figures/accuracy_vs_params.png`, `figures/accuracy_vs_latency.png` — the accuracy–efficiency trade-off.
- `figures/cm_<model>.png` — per-model confusion matrices.
- Raw per-model metrics + checkpoints live under `outputs/<model>/` (gitignored).

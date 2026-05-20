# Screw Detection & Classification — Report

> The run-results section at the bottom is appended automatically by the
> last cell of `screw_detection.ipynb`. Methodology and design rationale
> are documented here up front.

## 1 · Problem

Inspect images of industrial fasteners and assign each to one of six
classes:

```
Ecrou OK / Ecrou NOK        (nuts — good / bad)
Rondelle OK / Rondelle NOK  (washers — good / bad)
Vis OK / Vis NOK            (screws — good / bad)
```

The classification has two orthogonal axes:

- **Family** — Ecrou / Rondelle / Vis (decidable from geometry alone).
- **Condition** — OK / NOK (requires texture / surface evidence).

## 2 · Data

Six class folders under `data/raw/`. Splits are stratified 70 / 15 / 15
into train / val / test and written to `data/splits/*.csv` with a
`class_mapping.json` so the label indices are stable across runs.

## 3 · Two pipelines, side by side

### 3.1 Classical OpenCV pipeline

| Stage          | Operator                          | Rationale                                |
|----------------|-----------------------------------|------------------------------------------|
| Pre-processing | CLAHE + Gaussian blur             | normalise illumination, suppress noise   |
| Segmentation   | Otsu (inverted) + open + close    | foreground darker than the tray          |
| Detection      | 8-connected components            | one label per physical object            |
| Description    | contour features + 7 Hu moments   | scale/rotation-robust shape descriptors  |
| Classifier     | shape rules                       | transparent, tunable baseline            |

Family rules:

- `aspect_ratio > 2.2` → **Vis** (elongated)
- `circularity > 0.78` → **Rondelle** (round disc)
- otherwise → **Ecrou** (nut)

Because the dataset has no ground-truth pixel masks, the classical
pipeline is presented as *pseudo-segmentation* — it cannot be scored as a
segmentation model, but it gives an interpretable family baseline.

### 3.2 Deep-learning classifier

| Setting        | Value                                            |
|----------------|--------------------------------------------------|
| Backbone       | **ConvNeXt-Tiny**, ImageNet-pretrained (timm)    |
| Head           | linear, `num_classes=6`                          |
| Input          | 224 × 224, ImageNet normalisation                |
| Augmentation   | RandAugment(2, 9), RandomHorizontalFlip, RandomErasing(p=0.25) |
| Optimizer      | AdamW, lr=3e-4, weight_decay=1e-4                |
| Schedule       | linear warm-up 3 epochs → cosine annealing       |
| Loss           | cross-entropy, label_smoothing=0.05              |
| Precision      | mixed (`torch.cuda.amp`) on GPU                  |
| Epochs (full)  | 25                                               |
| TTA            | horizontal-flip logit averaging at test time     |

**Why ConvNeXt-Tiny?** Modern ConvNet that consistently outperforms
ResNet-50 / EfficientNet-B0 on small fine-grained datasets at comparable
parameter count (~28M). The 6-class screw task is fine-grained: the
OK/NOK distinction within each family hinges on subtle surface defects,
which ConvNeXt's larger receptive field captures better than a ResNet-18
baseline.

## 4 · Evaluation protocol

- **Headline metric:** macro-F1 on the held-out test split. Macro
  averaging weights all six classes equally, so a model that nails Vis
  but flounders on Rondelle NOK cannot hide behind accuracy.
- **Secondary:** per-class precision / recall / F1, confusion matrix,
  failure-case grid.
- **Explainability:** Grad-CAM heatmaps over the last ConvNeXt stage on
  six randomly sampled test images — both correct and incorrect.

## 5 · Reproducibility

```bash
jupyter notebook screw_detection.ipynb   # Run All
```

All RNGs are seeded (`SEED=42`). Outputs are written to `results/` and
overwritten on re-run. Best validation checkpoint is saved to
`results/checkpoints/convnext_tiny_best.pth`.

## 6 · Limitations

- Otsu assumes a roughly bimodal histogram; very cluttered scenes will
  need adaptive thresholding.
- The shape-rule family classifier is intentionally minimal — washers
  viewed obliquely may be misread as nuts.
- Touching fasteners merge into one connected component; a watershed
  step is the natural extension.
- No ground-truth pixel masks, so Task 2 is reported as
  pseudo-segmentation only.
  
## Run results (2026-05-20 05:16)

- Model: **convnext_tiny** (27.8M params), epochs=25, mode=full
- Splits: train=2100, val=450, test=450
- **Test macro-F1: 0.9889**, accuracy: 0.9889
- Classical family-classifier accuracy: 0.4100

### Per-class metrics

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| Ecrou NOK    |      1      |   1      |     1      |        75 |
| Ecrou OK     |      1      |   1      |     1      |        75 |
| Rondelle NOK |      1      |   0.9733 |     0.9865 |        75 |
| Rondelle OK  |      0.974  |   1      |     0.9868 |        75 |
| Vis NOK      |      1      |   0.96   |     0.9796 |        75 |
| Vis OK       |      0.9615 |   1      |     0.9804 |        75 |

Artifacts: `results/metrics.csv`, `results/run_summary.json`,
`results/figures/{confusion_matrix,training_curves,failure_cases}.png`,
`results/gradcam/gradcam_grid.png`.  
  

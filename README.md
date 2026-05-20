# Screw Detection & Classification

End-to-end industrial fastener inspection in **a single notebook**:
[`screw_detection.ipynb`](./screw_detection.ipynb).

Combines a **classical OpenCV pipeline** (CLAHE → Otsu → morphology →
contour features + Hu moments → rule-based family classifier) with a
**deep-learning classifier** (ConvNeXt-Tiny, ImageNet-pretrained,
fine-tuned for 25 epochs with AdamW + cosine warm-up, AMP, RandAugment,
RandomErasing, horizontal-flip TTA). Includes Grad-CAM explanations.

Runs identically on **Google Colab** (auto-mounts Drive and locates the
project) and **locally**.

## Repository layout

```
.
├── screw_detection.ipynb   # the entire pipeline
├── README.md               # this file
├── report.md               # methodology + results write-up
└── data/raw/
    ├── Ecrou NOK/    ├── Ecrou OK/
    ├── Rondelle NOK/ ├── Rondelle OK/
    └── Vis NOK/      └── Vis OK/
```

## Running on Google Colab

1. Upload this repository to **Google Drive** as
   `MyDrive/screw_project/` (the notebook also accepts
   `MyDrive/screw_detection/` or any folder containing `data/raw/`).
2. Open `screw_detection.ipynb` in Colab.
3. Runtime → Change runtime type → **T4 GPU** (recommended for 25 epochs).
4. **Run All**. Cell 1 auto-detects Colab, mounts Drive, and locates the
   project root. Cell 2 installs any missing dependency (`timm`,
   `opencv-python`, etc.). Everything downstream is automatic.

## Running locally

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-image \
            scikit-learn timm tqdm Pillow torch torchvision jupyter
jupyter notebook screw_detection.ipynb
```

Then **Run All**.

## Modes

Edit `MODE` at the top of the notebook:

| Mode    | Epochs | Use case                              |
|---------|--------|---------------------------------------|
| debug   |   1    | smoke-test the pipeline               |
| mini    |   5    | fast verification on a laptop         |
| medium  |  15    | local GPU                             |
| **full**|  25    | best results on Colab GPU (default)   |

## What the notebook produces

```
results/
├── metrics.csv                 # per-class precision / recall / F1
├── run_summary.json            # machine-readable run summary
├── checkpoints/
│   └── convnext_tiny_best.pth
├── classical/
│   └── features.csv            # contour + Hu features for sampled images
├── figures/
│   ├── class_distribution.png  ├── sample_grid.png
│   ├── classical_stages.png    ├── training_curves.png
│   ├── confusion_matrix.png    └── failure_cases.png
└── gradcam/
    └── gradcam_grid.png
data/splits/
├── train.csv  ├── val.csv  ├── test.csv  └── class_mapping.json
report.md      # a results section is appended after each run
```

## Headline numbers

- **Macro-F1** on the test split is the headline metric (accuracy alone
  is not reported as a primary number — class balance can be uneven).
- Classical family-classifier accuracy is reported alongside, as a
  transparent baseline for the *family* (Ecrou / Rondelle / Vis) decision.

## Design decisions

| Decision              | Choice                       | Why                                           |
|-----------------------|------------------------------|-----------------------------------------------|
| Backbone (6-class)    | ConvNeXt-Tiny                | best 6-class screw discriminator, ~28M params |
| Optimizer             | AdamW + cosine warm-up       | reliable, AMP-friendly                        |
| Mixed precision       | `torch.cuda.amp`             | ~2× speed-up on T4                            |
| Augmentation          | RandAugment(2, 9) + RandomErasing | strong but consistent                    |
| TTA                   | horizontal-flip averaging    | small free boost at test time                 |
| Loss                  | cross-entropy, label-smoothing 0.05 | calibration                            |
| Classical baseline    | OpenCV-only, no GT masks     | dataset has no pixel labels — kept as *family* baseline |

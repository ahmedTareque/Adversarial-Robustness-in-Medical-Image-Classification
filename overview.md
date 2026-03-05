
# Adversarial Robustness in Medical Image Classification — 3-Day Overview

## Day 01 — Baseline Model Development

**Goal:** Train a pneumonia classifier and establish clean performance.

**Dataset Structure:**

    data/
    ├── train/NORMAL, PNEUMONIA
    ├── val/NORMAL, PNEUMONIA
    └── test/NORMAL, PNEUMONIA

**Model:** MobileNetV2 (pretrained, frozen) + GlobalAveragePooling + Dropout(0.2) + Dense(1, sigmoid)

**Loss & Metrics:** binary_crossentropy, accuracy, precision (~77%), recall (~99%)

**Training:** 10 epochs, early stopping, model checkpoint

**Outcome:** Strong screening baseline ready for adversarial testing

---

## Day 02 — FGSM Attack & Robustness Evaluation

**Goal:** Evaluate model vulnerability to adversarial perturbations using FGSM

**FGSM Formula:** `x_adv = x + ε * sign(∇_x J(θ, x, y))`

**Observations:**

- Accuracy drops as epsilon increases
- Recall drops
- Model overconfident on wrong predictions
- Clean accuracy ≠ robust model

**Robustness Gap:** Clean Accuracy − Adversarial Accuracy

**Outcome:** Vulnerability confirmed, high recall does not imply robustness

---

## Day 03 — Grad-CAM Visualization & Interpretation

**Goal:** Visualize model attention on clean vs adversarial images

**Implementation:**

- Used last conv layer `Conv_1`
- Gradients → global pooling → weighted feature maps → ReLU → heatmap
- Resize heatmap to 224×224 before overlay

**Observations:**

- Clean images: attention on infection areas
- Adversarial images: attention shifts, feature maps distorted
- Explains robustness gap visually

**Outcome:** Model is interpretable; vulnerability is due to feature-level perturbations

---

## 3-Day Project Status

| Component | Status |
|-----------|--------|
| Baseline model | ✅ |
| FGSM attack | ✅ |
| Robustness gap measured | ✅ |
| Grad-CAM visualization | ✅ |
| Interpretability analysis | ✅ |

**Next Steps:** Adversarial training, defensive distillation, gradient regularization

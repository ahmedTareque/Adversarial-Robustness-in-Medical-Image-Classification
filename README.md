# 🩺 Adversarial Robustness in Medical Image Classification

------------------------------------------------------------------------

# 📝 Day 01 -- Baseline Model Development

## 1️⃣ Project Goal (Big Picture)

**Objective:**\
To study how vulnerable medical image classification models are to
adversarial attacks.

### Specifically

-   Train a pneumonia classifier\
-   Measure clean (normal) performance\
-   Later evaluate robustness under adversarial perturbations

------------------------------------------------------------------------

## Dataset Structure

data/ ├── train/ │ ├── NORMAL/ │ └── PNEUMONIA/ ├── val/ │ ├── NORMAL/ │
└── PNEUMONIA/ └── test/ ├── NORMAL/ └── PNEUMONIA/

We used `image_dataset_from_directory()` for automatic labeling and
batching.

------------------------------------------------------------------------

## Model Strategy

Transfer Learning using MobileNetV2 (ImageNet pretrained).\
Backbone frozen → Train only classifier head.

Architecture:

Input → Preprocessing → MobileNetV2 → GlobalAveragePooling\
→ Dropout(0.2) → Dense(1, sigmoid)

------------------------------------------------------------------------

## Metrics Used

-   Accuracy\
-   Precision\
-   Recall (Sensitivity)\
-   F1 Score\
-   AUC-ROC

High recall (\~99%) is critical for medical screening.

------------------------------------------------------------------------

## Day 01 Result

Clean Test Accuracy ≈ 81.6%\
High recall → Good screening model

Baseline established.

------------------------------------------------------------------------

# 🧨 Day 02 -- Adversarial Attack & Defense

## FGSM Attack

Formula:

x_adv = x + epsilon \* sign(gradient)

We evaluated model across: epsilon = \[0, 0.01, 0.03, 0.05, 0.1\]

Generated robustness curve (epsilon vs accuracy).

Observation: Accuracy drops significantly as epsilon increases.

------------------------------------------------------------------------

## Adversarial Training

During training: - Generate adversarial batch - Combine clean +
adversarial - Train model on combined data

Tradeoff observed:

  Aspect            Effect
  ----------------- -------------------------
  Clean Accuracy    Slight decrease
  Robust Accuracy   Significant improvement

Robust model shows improved stability under attack.

------------------------------------------------------------------------

# 🔍 Day 03 -- Interpretability & Research Framing

## Grad-CAM Analysis

Generated heatmaps for: - Clean image - Adversarial image

Key Observation: Adversarial attacks shift model attention away from
meaningful lung regions.

This shows adversarial noise alters internal reasoning patterns.

------------------------------------------------------------------------

## Key Findings

-   Baseline model is vulnerable to small perturbations\
-   Accuracy decreases as epsilon increases\
-   Adversarial training improves robustness\
-   Grad-CAM reveals attention shift\
-   Clear accuracy--robustness tradeoff

------------------------------------------------------------------------

## Final Project Status

Baseline Model ✅\
FGSM Attack ✅\
Robustness Evaluation ✅\
Adversarial Training ✅\
Grad-CAM Analysis ✅

------------------------------------------------------------------------

This project demonstrates a complete mini research pipeline in robust
medical AI.

# Adversarial Robustness in Medical Image Classification

## Day 03 --- Grad-CAM Visualization & Robustness Interpretation

------------------------------------------------------------------------

## 1️⃣ Objective of Day 03

After:

-   ✅ Building a strong baseline (Day 01)
-   ✅ Attacking the model using FGSM (Day 02)

Today's focus:

**Understand *why* the model fails under adversarial attack using
Grad-CAM visualization.**

This moves our project from pure metrics → to interpretability.

------------------------------------------------------------------------

## 2️⃣ Why Interpretability Matters in Medical AI

In healthcare:

-   Accuracy alone is not enough.
-   We must understand *what the model is looking at*.
-   Doctors require visual justification.

If adversarial noise changes the model decision, we need to see:

-   Did attention shift?
-   Did noise hijack feature extraction?
-   Is the model focusing on irrelevant regions?

------------------------------------------------------------------------

## 3️⃣ Grad-CAM Implementation

We used the final convolution layer:

    Conv_1

Grad-CAM Steps:

1.  Forward pass → Get prediction
2.  Compute gradient of predicted class w.r.t. feature maps
3.  Global average pooling of gradients
4.  Weight feature maps
5.  Apply ReLU
6.  Normalize heatmap
7.  Resize to (224×224)
8.  Overlay on original image

------------------------------------------------------------------------

## 4️⃣ Important Bug Fix (Critical Insight)

Initial Error:

ValueError: operands could not be broadcast together with shapes (7,7,3)
(224,224,3)

### Root Cause:

-   Grad-CAM heatmap was 7×7 (from final conv layer)
-   Original image was 224×224
-   They must match before overlay

### Fix:

Resize heatmap before blending:

    heatmap = cv2.resize(heatmap, (224, 224))

This ensures correct broadcasting.

------------------------------------------------------------------------

## 5️⃣ Clean vs Adversarial Visualization

### Clean Image

-   Model focused on lung region
-   Activation strong in infection areas
-   Decision confident

### Adversarial Image

-   Attention shifted
-   Activation diffused or misplaced
-   Noise disrupted feature representation

This explains the robustness gap observed in Day 02.

------------------------------------------------------------------------

## 6️⃣ Research Insight

Adversarial perturbations:

-   Do NOT look visible to humans
-   But significantly alter gradient flow
-   Distort intermediate feature maps
-   Redirect model attention

This proves the vulnerability is internal representation-level --- not
pixel-level perception.

------------------------------------------------------------------------

## 7️⃣ Technical Outputs

Files Generated:

    results/clean_heatmap.png
    results/adv_heatmap.png

Console Output Confirmed:

-   ✅ Heatmaps generated
-   ✅ Saved successfully
-   ⚠️ TensorFlow OUT_OF_RANGE warning (harmless end-of-dataset signal)

------------------------------------------------------------------------

## 8️⃣ Key Research Contribution

We now have:

-   Clean accuracy benchmark
-   Adversarial degradation measurement
-   Visual interpretability proof

This strengthens the academic depth of the project beyond simple attack
metrics.

------------------------------------------------------------------------

## 9️⃣ Project Status After 3 Days

  Component                   Status
  --------------------------- --------
  Baseline model              ✅
  FGSM attack                 ✅
  Robustness gap measured     ✅
  Grad-CAM visualization      ✅
  Interpretability analysis   ✅

------------------------------------------------------------------------

## 🔬 Next Logical Step (Day 04)

Potential directions:

-   Adversarial Training
-   Defensive Distillation
-   Input Gradient Regularization
-   Evaluate Robust Accuracy vs Clean Accuracy trade-off

------------------------------------------------------------------------

We have now transitioned from:

Model Training → Attack → Interpretation

This is now a complete mini research pipeline in adversarial medical AI.

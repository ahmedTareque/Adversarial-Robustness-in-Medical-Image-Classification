# Adversarial Robustness in Medical Image Classification

## Day 01 -- Baseline Model Development

------------------------------------------------------------------------

## 1️⃣ Project Goal (Big Picture)

**Objective:**\
To study how vulnerable medical image classification models are to
adversarial attacks.

### Specifically

-   Train a pneumonia classifier
-   Measure clean (normal) performance
-   Later evaluate robustness under adversarial perturbations

------------------------------------------------------------------------

## ✅ Day 01 Scope

-   Building a strong baseline model
-   Training it correctly
-   Evaluating performance on clean test data

------------------------------------------------------------------------

## 2️⃣ Dataset Structure

    data/
    │
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

### Why this structure?

We use:

``` python
tf.keras.utils.image_dataset_from_directory()
```

This automatically: - Reads folder names as labels
- Assigns binary targets
- Handles batching
- Reduces manual labeling errors

------------------------------------------------------------------------

## 3️⃣ Data Loading Strategy "The Sorting Hat"

### Key Parameters

| Parameter | Why We Use It 
|----------|-------------|
| `image_size=(224,224)` | Standard size for MobileNetV2
| `batch_size=32` | Efficient processing
| `label_mode='binary'` | Because we have 2 classes
| `Optimization` | Added caching and prefetching

### Pipeline Optimization

``` python
dataset.cache().prefetch(tf.data.AUTOTUNE)
```

-   `cache()` → Stores dataset in RAM for faster training
-   `prefetch()` → Prepares next batch while training

Keyword: **Data Pipeline Optimization**

| Keywords | Why We Use It 
|----------|-------------|
| `image_dataset_from_directory` | this function simplifies data loading and preprocessing by automatically labeling images based on their folder structure. So, If an image is in the PNEUMONIA folder, the code automatically tells the AI: "This is a 1."
| `Normalization` | Raw X-rays have pixel values from 0 to 255. AI math works best with small numbers. We scaled them to -1 to 1 using the preprocess_input function.

------------------------------------------------------------------------

## 4️⃣ Model Architecture "The Pre-trained Expert"

Transfer Learning using:

``` python
MobileNetV2(weights='imagenet', include_top=False)
```

### Why MobileNetV2?

-   Lightweight
-   Pretrained on ImageNet
-   Strong feature extractor
-   Fast convergence

### Freezing Backbone

``` python
base_model.trainable = False
```

Prevents retraining millions of parameters.
We train only the classifier head.

Keyword: **Feature Extraction Mode, Frozen Layers (Non-Trainable), Transfer Learning**

| Keywords | Why We Use It 
|----------|-------------|
| `Transfer Learning` | this model was already trained on 1.4 million general images and "transferred" its vision skills to our X-ray task. This is much faster and more effective than training from scratch.
| `Frozen Layers (Non-Trainable)` | We locked the "vision" part of the model. We only trained the Dense Layer (the final decision-maker). This is efficient and prevents overfitting on our small dataset.

------------------------------------------------------------------------

## Final Architecture

Input (224x224x3)\
→ Preprocessing (scale to \[-1,1\])\
→ MobileNetV2 (frozen)\
→ GlobalAveragePooling\
→ Dropout(0.2)\
→ Dense(1, sigmoid)

------------------------------------------------------------------------

## 5️⃣ Loss & The Metrics: "Success Beyond Accuracy"

``` python
loss='binary_crossentropy'
metrics=['accuracy', Precision(), Recall()]
```

### Why Binary Crossentropy?

Standard loss for probability-based binary classification.

### Why Precision & Recall?

In medical AI:

#### Keyword: 
`Recall (Sensitivity)` — Your Score: ~99%

##### What it means:
Out of 100 sick people, your model caught 99 of them.

##### Why it's vital: 
In medicine, missing a sick person (False Negative) is much more dangerous than accidentally flagging a healthy person.

#### Keyword: 
`Precision` — Your Score: ~77%

##### What it means:
When the model says "Pneumonia," it is right about 77% of the time. (Some healthy lungs were flagged as sick as well).

| Metric | Why We Use It 
|----------|-------------|
| `Precision` | Measures the accuracy of positive predictions
| `Recall (Sensitivity)` | Measures the ability to find all positive patients
| `F1-Score` | Harmonic mean of Precision and Recall
| `AUC-ROC` | Evaluates model performance across all thresholds


High recall is critical --- missing pneumonia is dangerous.

------------------------------------------------------------------------

## 6️⃣ Training Details. The Training Guardrails: "Anti-Overfitting"

-   Epochs: 10
-   Trainable Parameters: 1,281
-   Strategy: Efficient transfer learning

| Keyword | Why We Use It 
|----------|-------------|
| `Early Stopping` | Stops training when validation performance degrades
| `Model Checkpointing` | Saves the best model during training

------------------------------------------------------------------------

## 7️⃣ Test Results

  Metric      Value
  ----------- ----------
  Accuracy    0.8157  
  Precision   \~0.7767
  Recall      \~0.9897

| Metric | Value
|----------|-------------|
| `Accuracy` | 0.8157 |
| `Precision` | ~0.7767 |
| `Recall` | ~0.9897 |
| `F1-Score` | ~0.8650 |
| `AUC-ROC` | ~0.9500 |

### Interpretation

-   Strong screening model (very high recall)
-   Some false positives (moderate precision)
-   Acceptable trade-off in healthcare setting

------------------------------------------------------------------------

## 8️⃣ Research Importance

A strong clean baseline is required before evaluating adversarial
robustness.

If baseline is weak → robustness evaluation is meaningless.

------------------------------------------------------------------------

<!-- ## 9️⃣ Next Step (Day 02)

Implement FGSM attack and measure:

`Robustness Gap = Clean Accuracy − Adversarial Accuracy`

------------------------------------------------------------------------

## 📈 Day 01 Status

| Component | Status
|----------|-------------|
| ` Dataset pipeline` | ✅ |
| `Transfer learning model` | ✅ |
| `Training complete` | ✅ |
| `Test evaluation` | ✅ |
| `Baseline established` | ✅ |

------------------------------------------------------------------------ -->

Baseline model successfully established.\
Ready for adversarial robustness evaluation.

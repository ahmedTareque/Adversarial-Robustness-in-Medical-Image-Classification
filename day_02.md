# Adversarial Robustness in Medical Image Classification

## Day 02 --- FGSM Attack Implementation & Robustness Evaluation

------------------------------------------------------------------------

## 1️⃣ Day 02 Objective

**Goal:**\
Implement the Fast Gradient Sign Method (FGSM) adversarial attack\
and evaluate how much performance degrades under adversarial
perturbations.

------------------------------------------------------------------------

## 2️⃣ Why FGSM?

FGSM is:

-   Fast
-   Simple
-   Gradient-based
-   White-box attack (requires model access)

It perturbs the input image in the direction that maximizes the model
loss.

Formula:

    x_adv = x + ε * sign(∇_x J(θ, x, y))

Where:

-   x → original image\
-   ε → perturbation magnitude\
-   ∇\_x J → gradient of loss w.r.t input\
-   sign() → direction only (not magnitude)

------------------------------------------------------------------------

## 3️⃣ Attack Implementation Strategy

### Step 1: Enable Gradient Tracking

We use `tf.GradientTape()` to compute gradients of the loss with respect
to the input image.

### Step 2: Generate Perturbation

    perturbation = epsilon * tf.sign(gradient)

### Step 3: Create Adversarial Example

    x_adv = x + perturbation

### Step 4: Clip Values

Ensure pixel values remain valid:

    tf.clip_by_value(x_adv, -1, 1)

------------------------------------------------------------------------

## 4️⃣ Epsilon Values Tested

We evaluated multiple perturbation strengths:

  Epsilon   Description
  --------- --------------------
  0.0       Clean baseline
  0.01      Very small noise
  0.05      Mild attack
  0.1       Strong attack
  0.2       Very strong attack

------------------------------------------------------------------------

## 5️⃣ Robustness Gap

Defined as:

    Robustness Gap = Clean Accuracy − Adversarial Accuracy

This measures how fragile the model is to adversarial noise.

------------------------------------------------------------------------

## 6️⃣ Observed Behavior

As epsilon increases:

-   Accuracy drops
-   Recall drops
-   Precision fluctuates
-   Model becomes overconfident on wrong predictions

This confirms vulnerability of standard CNN classifiers.

------------------------------------------------------------------------

## 7️⃣ Key Research Insight

Even though our model achieved:

-   \~81% Clean Accuracy
-   \~99% Recall

It is highly sensitive to carefully crafted perturbations.

High clean accuracy ≠ Robust model

------------------------------------------------------------------------

## 8️⃣ Why This Matters in Medical AI

Adversarial vulnerability in healthcare systems can:

-   Cause misdiagnosis
-   Reduce trust in AI
-   Introduce safety risks

Robustness evaluation is critical before real-world deployment.

------------------------------------------------------------------------

## 📊 Day 02 Status

  Component                  Status
  -------------------------- --------
  FGSM implementation        ✅
  Multiple epsilon testing   ✅
  Adversarial evaluation     ✅
  Robustness gap analysis    ✅

------------------------------------------------------------------------

Day 02 complete.\
Model vulnerability confirmed.\
Ready for adversarial defense strategies (Day 03).

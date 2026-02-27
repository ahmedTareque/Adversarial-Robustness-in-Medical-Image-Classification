# import tensorflow as tf
# import numpy as np
# import json
# import os
# import matplotlib.pyplot as plt

# # ==========================================================
# # Configuration & Paths
# # ==========================================================
# # We use .h5 for TensorFlow models
# MODEL_PATH = "models/baseline_best.h5" 
# RESULTS_PATH = "results/metrics.json"
# DATA_PATH = "data/test"
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPSILON = 0.03  # Strength of the adversarial noise
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide spammy warnings
# # ==========================================================
# # Load Model & Data
# # ==========================================================
# def load_project_assets():
#     """Loads the trained model and the test dataset."""
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first!")
        
#     model = tf.keras.models.load_model(MODEL_PATH) # Or load_model(MODEL_PATH)
    
#     # Using the TF data loader we discussed earlier
#     test_ds = tf.keras.utils.image_dataset_from_directory(
#         DATA_PATH,
#         image_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         label_mode='binary' # Important for medical classification
#     )
#     return model, test_ds

# # ==========================================================
# # FGSM Attack Core Logic
# # ==========================================================
# def create_adversarial_pattern(model, images, labels):
#     """
#     The heart of the attack:
#     1. Computes the loss gradient with respect to the input pixels.
#     2. Takes the 'sign' of that gradient to identify which way to nudge pixels.
#     """
#     with tf.GradientTape() as tape:
#         tape.watch(images)
#         prediction = model(images)
#         loss = tf.keras.losses.BinaryCrossentropy()(labels, prediction)

#     # Get the gradients of the loss w.r.t to the input image
#     gradient = tape.gradient(loss, images)
#     # Get the sign: +1 if gradient is positive, -1 if negative
#     signed_grad = tf.sign(gradient)
#     return signed_grad

# # ==========================================================
# # Evaluation Functions
# # ==========================================================
# def run_evaluation(model, test_ds, epsilon):
#     """Runs evaluation on both clean and adversarial data."""
#     total_images = 0
#     clean_correct = 0
#     adv_correct = 0

#     print(f"Running evaluation (Epsilon: {epsilon})...")

#     for images, labels in test_ds:
#         # 1. Evaluate Clean Data
#         clean_preds = model.predict(images, verbose=0)
#         # Convert sigmoid probabilities (>0.5) to binary (0 or 1)
#         clean_classes = (clean_preds > 0.5).astype(np.int32)
#         clean_correct += np.sum(clean_classes == labels.numpy().astype(np.int32))

#         # 2. Generate Adversarial Data
#         # We calculate the 'noise' specifically designed to trick this model
#         perturbations = create_adversarial_pattern(model, images, labels)
#         adv_images = images + epsilon * perturbations
#         # Keep pixels in valid range (MobileNetV2 expects [-1, 1])
#         adv_images = tf.clip_by_value(adv_images, -1, 1)

#         # 3. Evaluate Adversarial Data
#         adv_preds = model.predict(adv_images, verbose=0)
#         adv_classes = (adv_preds > 0.5).astype(np.int32)
#         adv_correct += np.sum(adv_classes == labels.numpy().astype(np.int32))

#         total_images += labels.shape[0]

#     clean_acc = clean_correct / total_images
#     adv_acc = adv_correct / total_images
    
#     return clean_acc, adv_acc

# # ==========================================================
# # Main Execution
# # ==========================================================
# def main():
#     print("Initializing Evaluation...")
#     model, test_ds = load_project_assets()

#     clean_acc, adv_acc = run_evaluation(model, test_ds, EPSILON)

#     # Calculate Robustness Drop
#     drop = clean_acc - adv_acc

#     print("-" * 30)
#     print(f"Clean Accuracy:       {clean_acc:.4%}")
#     print(f"Adversarial Accuracy: {adv_acc:.4%}")
#     print(f"Accuracy Drop:        {drop:.4%}")
#     print("-" * 30)

#     # Save results to JSON
#     results = {
#         "clean_accuracy": clean_acc,
#         "adversarial_accuracy": adv_acc,
#         "epsilon": EPSILON,
#         "vulnerability_index": drop
#     }

#     os.makedirs("results", exist_ok=True)
#     with open(RESULTS_PATH, "w") as f:
#         json.dump(results, f, indent=4)

#     print(f"Success! Metrics saved to {RESULTS_PATH}")

# if __name__ == "__main__":
#     main()



# import tensorflow as tf
# import numpy as np
# import json
# import os
# from attacks.fgsm import fgsm_attack
# from data_loader import load_data

# # ==========================================================
# # Configuration
# # ==========================================================
# MODEL_PATH = "models/baseline_best.h5"
# DATA_PATH = "data"
# RESULTS_PATH = "results/metrics.json"
# EPSILON = 0.03 # The strength of the attack

# def evaluate_robustness():
#     # 1. Load Model and Data
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: {MODEL_PATH} not found. Train the model first!")
#         return

#     model = tf.keras.models.load_model(MODEL_PATH)
#     _, _, test_ds = load_data(DATA_PATH)

#     print("\nStarting Evaluation...")
    
#     total_images = 0
#     clean_correct = 0
#     adv_correct = 0

#     # 2. Loop through test data
#     for images, labels in test_ds:
#         # Ensure labels are the right shape (batch_size, 1)
#         labels = tf.reshape(labels, (-1, 1))
        
#         # --- Clean Evaluation ---
#         preds = model.predict(images, verbose=0)
#         clean_classes = (preds > 0.5).astype(np.int32)
#         clean_correct += np.sum(clean_classes == labels.numpy().astype(np.int32))

#         # --- Adversarial Evaluation ---
#         # Generate adversarial images using your fgsm.py
#         adv_images = fgsm_attack(images, EPSILON, model, labels)
        
#         adv_preds = model.predict(adv_images, verbose=0)
#         adv_classes = (adv_preds > 0.5).astype(np.int32)
#         adv_correct += np.sum(adv_classes == labels.numpy().astype(np.int32))

#         total_images += labels.shape[0]
#         print(f"Processed {total_images} images...", end="\r")

#     # 3. Calculate Final Scores
#     clean_acc = clean_correct / total_images
#     adv_acc = adv_correct / total_images
#     robustness_gap = clean_acc - adv_acc

#     # 4. Display & Save Results
#     print("\n" + "="*30)
#     print(f"RESULTS (Epsilon: {EPSILON})")
#     print("-" * 30)
#     print(f"Clean Accuracy:       {clean_acc:.4f}")
#     print(f"Adversarial Accuracy: {adv_acc:.4f}")
#     print(f"Robustness Gap:       {robustness_gap:.4f}")
#     print("="*30)

#     results = {
#         "epsilon": EPSILON,
#         "clean_accuracy": clean_acc,
#         "adversarial_accuracy": adv_acc,
#         "robustness_gap": robustness_gap
#     }

#     os.makedirs("results", exist_ok=True)
#     with open(RESULTS_PATH, "w") as f:
#         json.dump(results, f, indent=4)
#     print(f"Metrics saved to {RESULTS_PATH}")

# if __name__ == "__main__":
#     evaluate_robustness()


import tensorflow as tf
import numpy as np
import json
import os
from attacks.fgsm import fgsm_attack
from data_loader import load_data

# ==========================================================
# Configuration
# ==========================================================
MODEL_PATH = "models/baseline_best.h5"
DATA_PATH = "data"
RESULTS_PATH = "results/metrics.json"
EPSILON = 0.03 # The strength of the attack

def evaluate_robustness():
    # 1. Load Model and Data
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first!")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_ds = load_data(DATA_PATH)

    print("\nStarting Evaluation...")
    
    total_images = 0
    clean_correct = 0
    adv_correct = 0

    # 2. Loop through test data
    for images, labels in test_ds:
        # Ensure labels are the right shape (batch_size, 1)
        labels = tf.reshape(labels, (-1, 1))
        
        # --- Clean Evaluation ---
        preds = model.predict(images, verbose=0)
        clean_classes = (preds > 0.5).astype(np.int32)
        clean_correct += np.sum(clean_classes == labels.numpy().astype(np.int32))

        # --- Adversarial Evaluation ---
        # Generate adversarial images using your fgsm.py
        adv_images = fgsm_attack(images, EPSILON, model, labels)
        
        adv_preds = model.predict(adv_images, verbose=0)
        adv_classes = (adv_preds > 0.5).astype(np.int32)
        adv_correct += np.sum(adv_classes == labels.numpy().astype(np.int32))

        total_images += labels.shape[0]
        print(f"Processed {total_images} images...", end="\r")

    # 3. Calculate Final Scores
    clean_acc = clean_correct / total_images
    adv_acc = adv_correct / total_images
    robustness_gap = clean_acc - adv_acc

    # 4. Display & Save Results
    print("\n" + "="*30)
    print(f"RESULTS (Epsilon: {EPSILON})")
    print("-" * 30)
    print(f"Clean Accuracy:       {clean_acc:.4f}")
    print(f"Adversarial Accuracy: {adv_acc:.4f}")
    print(f"Robustness Gap:       {robustness_gap:.4f}")
    print("="*30)

    results = {
        "epsilon": EPSILON,
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "robustness_gap": robustness_gap
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_robustness()
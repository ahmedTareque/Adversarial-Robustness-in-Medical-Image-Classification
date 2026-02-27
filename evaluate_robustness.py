import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from attacks.fgsm import fgsm_attack
from data_loader import load_data
from training import build_model

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

model = build_model()
model.load_weights("models/baseline_model.h5")

# ---------------------------------------------------
# Load test data
# ---------------------------------------------------

_, _, test_ds = load_data("data")

# Epsilon values to test
epsilons = [0, 0.01, 0.03, 0.05, 0.1]

accuracies = []

print("Starting Robustness Evaluation...\n")

for epsilon in epsilons:

    correct = 0
    total = 0

    for images, labels in test_ds:

        # Expand labels shape if needed
        labels = tf.reshape(labels, (-1, 1))

        if epsilon == 0:
            # Clean prediction
            predictions = model(images)
        else:
            # Generate adversarial images
            adv_images = fgsm_attack(images, epsilon, model, labels)
            predictions = model(adv_images)

        # Convert probability to binary
        predicted_labels = tf.cast(predictions > 0.5, tf.float32)

        correct += tf.reduce_sum(tf.cast(predicted_labels == labels, tf.float32))
        total += images.shape[0]

    accuracy = correct / total
    accuracies.append(float(accuracy))

    print(f"Epsilon: {epsilon} | Accuracy: {accuracy:.4f}")

# ---------------------------------------------------
# Plot robustness curve
# ---------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(epsilons, accuracies, marker='o')
plt.xlabel("Epsilon (Attack Strength)")
plt.ylabel("Accuracy")
plt.title("Model Robustness Curve (FGSM)")
plt.grid(True)

plt.savefig("results/robustness_curve.png")
plt.show()
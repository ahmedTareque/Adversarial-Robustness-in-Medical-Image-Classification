# --- 1. Imports ---
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
# Import the loader function we wrote earlier
from attacks.fgsm import fgsm_attack
from data_loader import load_data 

# # 1. Now build the model
# model = build_model()
# model.summary()

def build_model():
    # Preprocessing layer to scale pixels from [0,255] to [-1,1] for MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Load Base Model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False, # Removes the final ImageNet layers
        weights='imagenet'
    )
    
    # Freeze the base model (don't train it yet)
    base_model.trainable = False

    # Build the full model
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Lambda(preprocess_input), # Auto-scales your images
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), # Good for preventing overfitting
        layers.Dense(1, activation='sigmoid') # Binary output: Normal vs Pneumonia
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

model = build_model()
model.summary()

# 2. Actually load the datasets
DATA_PATH = "data" 
train_ds, val_ds, test_ds = load_data(DATA_PATH)

# Check if data loaded
# for images, labels in train_ds.take(1):
#     print(f"Batch shape: {images.shape}") # Should be (32, 224, 224, 3)
#     print(f"Label shape: {labels.shape}") # Should be (32, 1)

for images, labels in train_ds:

    labels = tf.reshape(labels, (-1,1))

    # Generate adversarial images
    adv_images = fgsm_attack(images, epsilon=0.03, model=model, label=labels)

    # Combine clean + adversarial
    combined_images = tf.concat([images, adv_images], axis=0)
    combined_labels = tf.concat([labels, labels], axis=0)

    model.train_on_batch(combined_images, combined_labels)

# --- 3. DATA AUGMENTATION (ADVERSARIAL) ---
print("Generating Adversarial Training Data...")

# We create a function to map over the dataset to inject noise
def make_adversarial_batch(image, label):
    # Generate the noise
    adv_image = fgsm_attack(image, 0.03, model, label)
    # We want the model to see both clean and noisy images
    combined_imgs = tf.concat([image, adv_image], axis=0)
    combined_lbls = tf.concat([label, label], axis=0)
    return combined_imgs, combined_lbls

# Apply the attack to the training data
# Note: This makes the training set 2x larger
train_ds_robust = train_ds.map(make_adversarial_batch)

# --- 4. DEFINE CALLBACKS ---
# This goes right before model.fit
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/baseline_best.h5",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=3,
        monitor="val_accuracy",
        restore_best_weights=True,
        verbose=1
    )
]

# --- 5. THE TRAINING BLOCK ---
print("\nStarting Adversarial Training...")
history = model.fit(
    train_ds_robust, # Use the robust dataset!
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# --- 6. THE TRAINING BLOCK ---
print("\nStarting Adversarial Training...")
history = model.fit(
    train_ds_robust, # Use the robust dataset!
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# --- 7. SAVING & PLOTTING HISTORY ---
# This runs only after training is finished
print("\nTraining Complete. Generating Plots...")

def plot_and_save_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    # Create results folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        
    plt.savefig('results/training_history.png')
    print("Plot saved to results/training_history.png")
    plt.show()

plot_and_save_history(history)

# --- 8. FINAL TEST EVALUATION ---
print("\nEvaluating on Test Set...")
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
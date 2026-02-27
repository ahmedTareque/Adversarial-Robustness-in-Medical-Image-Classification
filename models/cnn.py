# ============================================
# Simple CNN for Medical Image Classification
# ============================================

import tensorflow as tf
from tensorflow.keras import layers, models


def build_simple_cnn(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds a simple Convolutional Neural Network.

    Parameters:
    -----------
    input_shape : tuple
        Shape of input image (height, width, channels)

    num_classes : int
        1  -> Binary classification
        >1 -> Multi-class classification

    Returns:
    --------
    model : tf.keras.Model
        Compiled CNN model
    """

    # Create Sequential model (stack layers one after another)
    model = models.Sequential()

    # ============================================
    # 1️⃣ Input Layer
    # ============================================
    # Defines the shape of input images
    model.add(layers.Input(shape=input_shape))

    # ============================================
    # 2️⃣ Convolution Block 1
    # ============================================
    # Conv2D:
    # - 32 filters
    # - 3x3 kernel
    # - ReLU activation
    # - Padding = same (keeps image size same)
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))

    # BatchNormalization:
    # - Stabilizes training
    # - Speeds convergence
    model.add(layers.BatchNormalization())

    # MaxPooling:
    # - Reduces spatial size
    # - Keeps strongest features
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ============================================
    # 3️⃣ Convolution Block 2
    # ============================================
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))

    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ============================================
    # 4️⃣ Convolution Block 3
    # ============================================
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    ))

    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ============================================
    # 5️⃣ Flatten
    # ============================================
    # Converts 3D feature maps into 1D vector
    model.add(layers.Flatten())

    # ============================================
    # 6️⃣ Fully Connected Layer
    # ============================================
    model.add(layers.Dense(128, activation='relu'))

    # Dropout:
    # - Randomly turns off neurons
    # - Helps prevent overfitting
    model.add(layers.Dropout(0.5))

    # ============================================
    # 7️⃣ Output Layer
    # ============================================
    if num_classes == 1:
        # Binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        loss_function = tf.keras.losses.BinaryCrossentropy()
        metrics = ['accuracy']
    else:
        # Multi-class classification
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        metrics = ['accuracy']

    # ============================================
    # 8️⃣ Compile Model
    # ============================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_function,
        metrics=metrics
    )

    return model
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_gradcam_heatmap(img_array, model, pred_index=None):

    # Force model build
    _ = model(tf.zeros((1, 224, 224, 3)))

    # Extract base model
    base_model = model.get_layer("mobilenetv2_1.00_224")

    # Find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found.")

    print(f"✅ Using last conv layer: {last_conv_layer.name}")

    # Model: input → conv feature maps
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Classifier head (everything after base_model)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # Recreate head manually
    x = model.layers[2](x)  # global_average_pooling2d
    x = model.layers[3](x)  # dropout
    x = model.layers[4](x)  # dense

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        preds = classifier_model(conv_outputs)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def save_and_display_gradcam(img_tensor, heatmap, cam_path="results/gradcam.png", alpha=0.4):

    # Convert tensor to numpy
    img = img_tensor.numpy()

    # Normalize image to 0-255
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    # 🔥 Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    superimposed_img = cv2.addWeighted(jet, alpha, img, 1 - alpha, 0)

    # Save
    cv2.imwrite(cam_path, superimposed_img)

    print(f"✅ Heatmap saved to {cam_path}")
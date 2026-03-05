import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from explain import get_gradcam_heatmap, save_and_display_gradcam
from attacks.fgsm import fgsm_attack
from data_loader import load_data
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# 1. Load Model and Data
model = tf.keras.models.load_model("models/baseline_best.h5", custom_objects={'preprocess_input': preprocess_input})
_, _, test_ds = load_data("data")
# last_conv_layer_name = "Conv_1" # Correct layer for MobileNetV2

# 2. Get one image from test set
for images, labels in test_ds.take(1):
    image = images[0]
    label = labels[0]
    
    # 3. Create Adversarial Version
    # Reshape for attack function (batch size 1)
    img_batch = tf.expand_dims(image, 0)
    label_batch = tf.expand_dims(label, 0)
    adv_image_batch = fgsm_attack(img_batch, 0.03, model, label_batch)
    adv_image = adv_image_batch[0]

    # 4. Generate Heatmaps
    # clean_heatmap = get_gradcam_heatmap(img_batch, model, last_conv_layer_name)
    # adv_heatmap = get_gradcam_heatmap(adv_image_batch, model, last_conv_layer_name)
    
    clean_heatmap = get_gradcam_heatmap(img_batch, model)
    adv_heatmap = get_gradcam_heatmap(adv_image_batch, model)

    # 5. Visualize
    os.makedirs("results", exist_ok=True)
    
    # Clean Heatmap
    save_and_display_gradcam(image, clean_heatmap, cam_path="results/clean_heatmap.png")
    # Adv Heatmap
    save_and_display_gradcam(adv_image, adv_heatmap, cam_path="results/adv_heatmap.png")

    print("✅ Heatmaps generated in /results folder.")
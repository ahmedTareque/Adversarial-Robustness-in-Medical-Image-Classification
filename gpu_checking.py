import tensorflow as tf

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
print(len(gpus) > 0)

# Print GPU name if available
if gpus:
    print(gpus[0].name)
else:
    print("No GPU")
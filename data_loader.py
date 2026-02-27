import tensorflow as tf
import matplotlib.pyplot as plt
import os

def load_data(data_dir, batch_size=32, img_size=(224, 224)):
    # Load Training Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    # Load Validation Dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    # Load Test Dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Optimize dataset loading
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    

    return train_ds, val_ds, test_ds

def show_samples(dataset):
    class_names = dataset.class_names
    plt.figure(figsize=(15, 5))
    # Grab one batch (32 images), take 5
    for images, labels in dataset.take(1):
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"{class_names[int(labels[i])]}")
            plt.axis("off")
    plt.show()

# --- RUNNING IT ---
if __name__ == "__main__":
    DATA_PATH = "data" # Update this to your local path
    train, val, test = load_data(DATA_PATH)
    
    # 1. Print Class Names
    print(f"Classes: {train.class_names}")
    
    # 2. Show 5 samples
    show_samples(train)
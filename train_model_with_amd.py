import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
import numpy as np
import random


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"    # Show all logs
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "3"     # Verbose logging
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_MLIR"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Check if GPU is available using the legacy API.
print("Is GPU available:", tf.test.is_gpu_available())


data_directory = "Training/"  

classes = ["0", "1", "2", "3", "4", "5", "6"]
img_size = 224  # Adjust if needed
batch_size = 64

def custom_data_generator(data_dir, classes, img_size, batch_size):
    """
    A generator that yields batches of images and labels.
    """
    # Build a list of (image_path, label) tuples for all classes.
    data_list = []
    for category in classes:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Directory {path} not found!")
            continue
        class_num = classes.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            data_list.append((img_path, class_num))
    
    # Shuffle the list once at the start of each epoch.
    while True:
        random.shuffle(data_list)
        total_samples = len(data_list)
        for i in range(0, total_samples, batch_size):
            batch_data = data_list[i:i+batch_size]
            batch_images = []
            batch_labels = []
            for img_path, label in batch_data:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    resized = cv2.resize(img, (img_size, img_size))
                except Exception as e:
                    print(f"Error resizing {img_path}: {e}")
                    continue
                batch_images.append(resized)
                batch_labels.append(label)
            if len(batch_images) == 0:
                continue
            X = np.array(batch_images, dtype=np.float32).reshape(-1, img_size, img_size, 3) / 255.0
            y = np.array(batch_labels, dtype=np.int32)
            yield X, y

# Determine total number of samples (for steps_per_epoch)
all_samples = 0
for category in classes:
    path = os.path.join(data_directory, category)
    if os.path.exists(path):
        all_samples += len(os.listdir(path))
print(f"Total samples found: {all_samples}")
steps_per_epoch = all_samples // batch_size

# ===================== Model Definition =====================
# Load pre-trained MobileNetV2 (available in TF 1.x)
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(img_size, img_size, 3),
    pooling='avg',
    weights='imagenet'
)
# Freeze the base model during initial training
base_model.trainable = True

# Create custom classification head using your new design:
# Here we use the base model's input and output, then add our custom layers.
base_input = base_model.input
base_output = base_model.output

# New classification head: two sequential Dense layers with ReLU, then final softmax.
x = layers.Dense(128)(base_output)
x = layers.Activation('relu')(x)
x = layers.Dense(64)(x)
x = layers.Activation('relu')(x)
final_output = layers.Dense(7, activation='softmax')(x)

new_model = keras.Model(inputs=base_input, outputs=final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
new_model.summary()


# Train the model using the custom data generator.
initial_epochs = 25
new_model.fit_generator(
    generator=custom_data_generator(data_directory, classes, img_size, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=initial_epochs
)



# Save the Model
new_model.save("my_model.keras", save_format="keras")
new_model.save("my_model.h5", save_format="h5")
print("Model training complete and saved as 'my_model.keras' and 'my_model.h5'")

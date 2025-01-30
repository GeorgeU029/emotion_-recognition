import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import os
import numpy as np
import random

# Data directory and classes
data_directory = "Training/"
classes = ["0", "1", "2", "3", "4", "5", "6"]
img_size = 224

# Load training data
training_data = []
def create_training_data():
    global training_data  # Ensure this modifies the main list
    for category in classes:
        path = os.path.join(data_directory, category)
        if not os.path.exists(path):
            print(f"Directory {path} not found!")
            continue
        print(f"Processing category: {category}")  # Check if it's processing categories

        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                if img_array is None:
                    print(f"Failed to load image: {img_path}")  # Check if images fail to load
                    continue

                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error processing {img}: {e}")  # Capture any other errors

    print(f"Total training samples loaded: {len(training_data)}")  # Final confirmation



create_training_data()
random.shuffle(training_data)

# Prepare features and labels
x, y = [], []
for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3) / 255.0  # Normalize
y = np.array(y)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), pooling='avg')

# Transfer learning
base_input = base_model.input
base_output = base_model.output

final_output = layers.Dense(128, activation='relu')(base_output)
final_output = layers.Dense(64, activation='relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

# Create new model
new_model = keras.Model(inputs=base_input, outputs=final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(f"Total training samples loaded: {len(training_data)}")

# Train the model
new_model.fit(x, y, epochs=15)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Enviromental!
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"    # Show all logs
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "3"     # Verbose logging

# Optional thread settings (if needed)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_MLIR"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Check if GPU is available using the legacy API.
print("Is GPU available:", tf.test.is_gpu_available())


data_directory = "Training/"  
img_size = 224
batch_size = 128
epochs = 25

# ---------------------- Data Pipeline using ImageDataGenerator ----------------------
# Create a data generator with a validation split.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1  # 10% of data used for validation
)

# Training generator: uses the "training" subset.
train_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse',  # Use sparse labels for integer class indices
    subset='training',
    shuffle=True
)

# Validation generator: uses the "validation" subset.
validation_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=True
)


# Load the pre-trained MobileNetV2 base (without its top classifier)
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(img_size, img_size, 3),
    pooling='avg',
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model

# Add custom layers for our 7-class problem.
x = layers.Dense(128, activation='relu')(base_model.output)
x = layers.Dense(64, activation='relu')(x)
final_output = layers.Dense(len(train_generator.class_indices), activation='softmax')(x)

# Define the complete model.
model = models.Model(inputs=base_model.input, outputs=final_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)


model.save("my_model.keras")
print("Model training complete and saved as 'my_model.keras'")

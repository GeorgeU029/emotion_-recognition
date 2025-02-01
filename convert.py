import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform, Zeros

# ✅ Load the original TensorFlow 1.x model
tf.compat.v1.disable_eager_execution()  # Ensure compatibility mode
model = load_model("my_model.h5", compile=False)  # Do not compile yet

# ✅ Fix initializers manually
for layer in model.layers:
    if hasattr(layer, "kernel_initializer"):
        layer.kernel_initializer = GlorotUniform()
    if hasattr(layer, "bias_initializer"):
        layer.bias_initializer = Zeros()

# ✅ Save in a TensorFlow 2.x-compatible format
model.save("my_model_converted.h5")

print("✅ Model successfully converted to TensorFlow 2.x format.")

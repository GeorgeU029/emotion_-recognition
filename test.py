import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your pre-trained model
new_model = tf.keras.models.load_model("my_model.keras")

# Read the image and print its shape
frame = cv2.imread("sadboy.webp")
print("Original frame shape:", frame.shape)

# Face detection algorithm setup
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the gray image
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

face_roi = None  # Initialize face_roi in case no face is found

for (x, y, w, h) in faces:
    # Define the regions of interest in the gray and color images
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    # Draw a rectangle around the detected face in the original frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Optionally, detect features (such as eyes) within the face ROI
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        # No additional features found; you can handle it if needed
        pass
    else:
        # For each detected sub-feature, crop the face ROI.
        # This loop will override face_roi each time, so the last detected sub-region is used.
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey:ey+eh, ex:ex+ew]

# If a face ROI was detected, process it
if face_roi is not None:
    # Resize and preprocess the face ROI for the model
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    prediction = new_model.predict(final_image)
    print("Raw prediction scores:", prediction[0])
    print("Predicted class:", np.argmax(prediction))
else:
    print("No face ROI detected.")

# Instead of showing the image interactively (which can trigger the "agg" backend warning),
# we'll save the images to a file.

# Create a figure with two subplots: one for the full frame and one for the face ROI.
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the full frame with the rectangle(s)
axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axs[0].set_title("Detected Faces")
axs[0].axis("off")

# Display the cropped face ROI (if available)
if face_roi is not None:
    axs[1].imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Cropped Face ROI")
else:
    axs[1].text(0.5, 0.5, "No face ROI detected", horizontalalignment='center', verticalalignment='center')
    axs[1].set_title("Cropped Face ROI")
axs[1].axis("off")

# Save the figure to a file
plt.savefig("output.png")
print("Figure saved to output.png")

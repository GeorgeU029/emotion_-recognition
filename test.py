import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your pre-trained model (make sure the model file "my_model.h5" is in the working directory)
new_model = tf.keras.models.load_model("my_model.h5")

# Read the image and print its shape
frame = cv2.imread("mad.jpeg")
print("Original frame shape:", frame.shape)

# Face detection algorithm setup
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the gray image
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

face_roi = None       # To store the region-of-interest (ROI) for prediction
face_coords = None    # To store coordinates for annotation

# Padding ratio (10% padding on each side)


# Loop over each detected face (process only the first detected face)
for (x, y, w, h) in faces:
    # Calculate padded coordinates
    x_pad = max(0, int(x -  w))
    y_pad = max(0, int(y - h))
    x_end = min(frame.shape[1], int(x + w + w))
    y_end = min(frame.shape[0], int(y + h +  h))
    
    # Use the padded region as the ROI for this face
    face_roi = frame[y_pad:y_end, x_pad:x_end]
    face_coords = (x_pad, y_pad, x_end - x_pad, y_end - y_pad)
    
    # Draw a rectangle around the padded face region on the original frame
    cv2.rectangle(frame, (x_pad, y_pad), (x_end, y_end), (255, 0, 0), 2)
    
    # Process only the first detected face
    break

predicted_emotion = "No prediction"
raw_scores_text = ""
if face_roi is not None:
    # Resize the ROI to the size expected by your model (e.g., 224x224)
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0  # Normalize pixel values

    # Run prediction
    prediction = new_model.predict(final_image)
    print("Raw prediction scores:", prediction[0])
    predicted_class = np.argmax(prediction)
    print("Predicted class (number):", predicted_class)

    # Map the predicted class to an emotion label
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = emotions[predicted_class] if predicted_class < len(emotions) else "Unknown"
    print("Predicted emotion:", predicted_emotion)
    
    # Prepare text for suptitle (you can adjust formatting as needed)
    raw_scores_text = ", ".join([f"{score:.2f}" for score in prediction[0]])
    
    # Annotate the original frame with the predicted emotion.
    if face_coords is not None:
        (fx, fy, fw, fh) = face_coords
        text_position = (fx, max(fy - 10, 20))  # slightly above the detected face
    else:
        text_position = (10, 30)
    
    # Draw a black rectangle for text visibility and put the predicted emotion text.
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, predicted_emotion, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    print("No face ROI detected.")

# ===================== Visualization =====================
# Create a figure with two subplots for a polished visualization.
fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

# Full image with annotations
axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axs[0].set_title("Detected Faces & Emotion", fontsize=16)
axs[0].axis("off")

# Cropped face ROI (if available)
if face_roi is not None:
    axs[1].imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Cropped Face ROI", fontsize=16)
else:
    axs[1].text(0.5, 0.5, "No face ROI detected", horizontalalignment='center', 
                verticalalignment='center', fontsize=16)
    axs[1].set_title("Cropped Face ROI", fontsize=16)
axs[1].axis("off")

# Add a super-title with the predicted emotion and raw scores.
plt.suptitle(f"Predicted Emotion: {predicted_emotion}\nRaw Scores: {raw_scores_text}", fontsize=20, color='darkblue')

plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout so suptitle fits well

# Save the figure to a file
plt.savefig("output.png")
print("Figure saved to output.png")

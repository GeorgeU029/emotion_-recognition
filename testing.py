import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your pre-trained model 
new_model = tf.keras.models.load_model("my_model.h5")

# Read the image and print its shapes
frame = cv2.imread("Neutral.jpg")
print("Original frame shape:", frame.shape)

# Face detection algorithm setup
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the gray image
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

face_roi = None       
face_coords = None    

padding_ratio = 0.1

# Loop over each detected face (only the first face is processed)
for (x, y, w, h) in faces:
    # Calculate padded coordinates (so the box is a bit larger than the detected face)
    x_pad = max(0, int(x - padding_ratio * w))
    y_pad = max(0, int(y - padding_ratio * h))
    x_end = min(frame.shape[1], int(x + w + padding_ratio * w))
    y_end = min(frame.shape[0], int(y + h + padding_ratio * h))
    
    # Use the padded region as the ROI for prediction
    face_roi = frame[y_pad:y_end, x_pad:x_end]
    face_coords = (x_pad, y_pad, x_end - x_pad, y_end - y_pad)
    
    # Draw a rectangle around the padded face region on the original frame
    cv2.rectangle(frame, (x_pad, y_pad), (x_end, y_end), (255, 0, 0), 2)
    
    # Process only the first detected face
    break

predicted_emotion = "No prediction"
raw_scores_text = ""
if face_roi is not None:
    # Prepare the face ROI for prediction: resize and normalize the image
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
    
    # Prepare text for display
    raw_scores_text = ", ".join([f"{score:.2f}" for score in prediction[0]])
    
    # Determine text position (above the face box if possible)
    if face_coords is not None:
        (fx, fy, fw, fh) = face_coords
        text_position = (fx, max(fy - 10, 20))
    else:
        text_position = (10, 30)
    
    # Draw a background rectangle for text visibility
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    # Put the predicted emotion text on the image
    cv2.putText(frame, predicted_emotion, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    print("No face ROI detected.")

# Plot only the original image with the annotations
plt.figure(figsize=(10, 8), dpi=100)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Emotion: {predicted_emotion}\nRaw Scores: {raw_scores_text}", fontsize=16)
plt.axis("off")
plt.tight_layout()

# Save the figure to a file
plt.savefig("output.png")
print("Figure saved to output.png")
plt.show()

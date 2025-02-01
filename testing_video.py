import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np

# Load pre-trained model 
new_model = tf.keras.models.load_model("my_model.h5")

# face detection using Haar Cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup video capture from a file 
video_path = "vid1.mov"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Get video properties for output writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup VideoWriter to save the annotated output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_video.avi", fourcc, fps, (frame_width, frame_height))

# Parameters for processing
padding_ratio = 0.1
input_size = 224  # Model expects 224x224

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    # Create a copy of the frame for annotations
    annotated_frame = frame.copy()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    predicted_emotion = "No prediction"
    raw_scores_text = ""
    face_roi = None
    face_coords = None

    # Process only the first detected face
    for (x, y, w, h) in faces:
       
        x_pad = max(0, int(x - padding_ratio * w))
        y_pad = max(0, int(y - padding_ratio * h))
        x_end = min(frame.shape[1], int(x + w + padding_ratio * w))
        y_end = min(frame.shape[0], int(y + h + padding_ratio * h))
        
        # Extract the face ROI using padded coordinates
        face_roi = frame[y_pad:y_end, x_pad:x_end]
        face_coords = (x_pad, y_pad, x_end - x_pad, y_end - y_pad)
        
        # Draw a rectangle around the padded face region on the annotated frame
        cv2.rectangle(annotated_frame, (x_pad, y_pad), (x_end, y_end), (255, 0, 0), 2)
        
        # Process only the first detected face per frame
        break

    if face_roi is not None:
        # Resize the ROI to the size expected by your model (224x224)
        try:
            resized_roi = cv2.resize(face_roi, (input_size, input_size))
        except Exception as e:
            print(f"Error resizing ROI: {e}")
            continue
        
        final_image = np.expand_dims(resized_roi, axis=0)
        final_image = final_image / 255.0  # Normalize pixel values
        
        # Run prediction
        prediction = new_model.predict(final_image)
        predicted_class = np.argmax(prediction)
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        predicted_emotion = emotions[predicted_class] if predicted_class < len(emotions) else "Unknown"
        raw_scores_text = ", ".join([f"{score:.2f}" for score in prediction[0]])
        
        # Annotate the frame with predicted emotion.
        if face_coords is not None:
            (fx, fy, fw, fh) = face_coords
            text_position = (fx, max(fy - 10, 20))  
        else:
            text_position = (10, 30)
        
        # Draw a black rectangle for text visibility at the top-left corner
        cv2.rectangle(annotated_frame, (0, 0), (250, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, predicted_emotion, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
        cv2.putText(annotated_frame, raw_scores_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        print("No face ROI detected in this frame.")
    
    # Write the annotated frame to the output video file
    out.write(annotated_frame)
    
    

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Output video saved as 'output_video.avi'")

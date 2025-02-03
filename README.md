# Emotion Recognition Model

This project implements an emotion recognition model that classifies facial expressions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. It uses transfer learning with MobileNetV2 and a custom data generator.

## Key Features

- **Custom Data Generator:** Loads and preprocesses images from the `Training/` directory.
- **Transfer Learning:** Uses a pre-trained MobileNetV2 model with custom dense layers.
- **Face Detection & Emotion Prediction:** Uses OpenCV to detect faces and TensorFlow to predict emotions.

## Important Note

This project uses **TensorFlow 1.5**. I trained the model on an AMD GPU using **DirectML** for GPU acceleration. Adjustments may be needed for other setups or newer versions of TensorFlow.

## Installation & Setup

git clone https://github.com/GeorgeU029/emotion_-recognition.git
cd emotion_-recognition
pip install -r requirements.txt

## Training the Model

Run the following script to build and train the model.

python train_model_with_amd.py

## Testing the Model

After training, test the model using these scripts:
python testing.py         # For images
python testing_video.py   # For videos

## Output

The results will be saved as:
output.png



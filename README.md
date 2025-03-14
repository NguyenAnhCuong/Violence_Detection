# Violence Detection Model

This repository contains a deep learning model for detecting violence in video streams. The model is built using a 3D Convolutional Neural Network (CNN) and is trained to classify video clips as either "Violence" or "Non-Violence".

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Requirement](#requirement)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Running the Model](#running-the-model)
- [Acknowledgements](#acknowledgements)

## Introduction

The goal of this project is to develop a real-time violence detection system using a 3D CNN model. The model processes video frames and predicts whether the video contains violent activities.

## Model Architecture

The model architecture consists of multiple 3D convolutional layers followed by max-pooling layers, a flatten layer, and fully connected dense layers. The final layer uses a sigmoid activation function to output a binary classification.

```python
model = Sequential([
    Conv3D(32, kernel_size=(3,3,3), activation="relu", input_shape=(16, 224, 224, 3)),
    MaxPooling3D(pool_size=(1,2,2)),
    Conv3D(64, kernel_size=(3,3,3), activation="relu"),
    MaxPooling3D(pool_size=(1,2,2)),
    Conv3D(128, kernel_size=(3,3,3), activation="relu"),
    MaxPooling3D(pool_size=(2,2,2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
```

## Requirements

pip install tensorflow opencv-python numpy playsound

## Usage

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load dataset
dataset_path = "./dataset"
X, y = load_dataset(dataset_path)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = Sequential([
    Conv3D(32, kernel_size=(3,3,3), activation="relu", input_shape=(16, 224, 224, 3)),
    MaxPooling3D(pool_size=(1,2,2)),
    Conv3D(64, kernel_size=(3,3,3), activation="relu"),
    MaxPooling3D(pool_size=(1,2,2)),
    Conv3D(128, kernel_size=(3,3,3), activation="relu"),
    MaxPooling3D(pool_size=(2,2,2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("violence_detection_model.h5")
```

## Running the Model

```python
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from playsound import playsound
import threading

# Load trained model
model = tf.keras.models.load_model("./violence_detection_model.h5")

# Camera RTSP URL
rtsp_url = "rtsp://admin:RAJVMI@192.168.1.12:554/h264_stream"

# Connect to camera
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 16)
cap.set(cv2.CAP_PROP_FPS, 16)

if not cap.isOpened():
    print("Cannot connect to camera.")
    exit()

save_path = "saved_frames"
if not os.path.exists(save_path):
    os.makedirs(save_path)

frame_size = (96, 96)
frame_count = 30
violence_threshold = 0.95
alarm_sound = "./clock-alarm.mp3"

frames = []
violence_frames = []

def play_alarm():
    playsound(alarm_sound)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from camera.")
        break

    frame_display = cv2.resize(frame, (800, 600))
    cv2.putText(frame_display, "Monitoring", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    frame_resized = cv2.resize(frame, frame_size)
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_resized = frame_resized / 255.0

    frames.append(frame_resized)

    if len(frames) == frame_count:
        video_clip = np.array(frames, dtype=np.float32)
        video_clip = np.expand_dims(video_clip, axis=0)

        prediction = model.predict(video_clip)
        violence_prob = prediction[0, 0]
        label = "Violence" if violence_prob > violence_threshold else "Non-Violence"
        color = (0, 0, 255) if violence_prob > violence_threshold else (0, 255, 0)
        
        cv2.putText(frame_display, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        print(f"Prediction: {label} - Violence: {violence_prob:.2f}")

        if violence_prob > violence_threshold:
            threading.Thread(target=play_alarm, daemon=True).start()

        frames = frames[-4:]

    cv2.imshow("Camera", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Acknowledgements

***TensorFlow***:https://www.tensorflow.org/

***OpenCV***:https://opencv.org/

***NumPy***:https://numpy.org/

***playsound***:https://github.com/TaylorSMarks/playsound

***Sklearn***:https://scikit-learn.org/stable/

### Tổng quan

- **Introduction**: Giới thiệu về dự án và mục tiêu của nó.
- **Model Architecture**: Mô tả kiến trúc của mô hình CNN 3D.
- **Requirements**: Các thư viện cần thiết để chạy mã.
- **Usage**: Hướng dẫn cách huấn luyện mô hình và chạy mô hình trên luồng video.
- **Acknowledgements**: Các thư viện và công cụ được sử dụng trong dự án.




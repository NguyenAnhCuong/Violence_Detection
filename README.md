<h1 align="center"> NHẬN DIỆN HÀNH VI BẠO LỰC TRONG ĐÁM ĐÔNG BẰNG CNN 3D</h1>

<div align="center">
  
  <p align="center">
    <img src="./logoDaiNam.png" alt="Dai Nam Logo" width="200"/>
    <img src="./LogoAIoTLab.png" alt="AIoTLab Logo" width="200"/>
  </p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>

<h2 align="center">💡 Nhận diện hành vi bạo lực trong đám đông</h2>
    Kho lưu trữ này chứa một mô hình học sâu để phát hiện bạo lực trong luồng video. Mô hình được xây dựng bằng Mạng nơ-ron tích chập 3D (CNN) và được đào tạo để phân loại các clip video thành "Bạo lực" hoặc "Không bạo lực".
<p align="left">
  
</p>

---

## 📦Mục lục

- [Giới thiệu](#gioi-thieu)
- [Hệ thống](#he-thong)
- [Thư viện](#thu-vien)
- [Huấn luyện mô hình](#huan-luyen-mo-hinh)
- [Chạy mô hình](#chay-mo-hinh)
- [Công cụ sử dụng](#cong-cu-su-dung)
- [Poster](#poster)
- [Đóng góp](#dong-gop)

## 🌟Giới thiệu

Mục tiêu của dự án này là phát triển một hệ thống phát hiện bạo lực thời gian thực bằng mô hình CNN 3D. Mô hình xử lý các khung hình video và dự đoán liệu video có chứa các hoạt động bạo lực hay không.

## 🚀Hệ thống

<div align="center">
    <img src="./architecture.png" alt="Dai Nam Logo" width="800" height="400"/>
</div>

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

## 📖Thư viện

pip install tensorflow opencv-python numpy playsound

## 🛠️Huận luyện mô hình

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

## 💻Chạy mô hình

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

## ⚙️Công cụ sử dụng

**_TensorFlow_**:https://www.tensorflow.org/

**_OpenCV_**:https://opencv.org/

**_NumPy_**:https://numpy.org/

**_playsound_**:https://github.com/TaylorSMarks/playsound

**_Sklearn_**:https://scikit-learn.org/stable/

## 📰Poster

<div align="center">
    <img src="./Poster01.png" alt="Dai Nam Logo"/>
</div>

## 🤝Đóng góp

Dự án được phát triển bởi 3 thành viên:

<table>
        <thead>
            <tr>
                <th style="text-align:center">Họ và Tên</th>
                <th colspan="4" style="text-align:center">Vai Trò</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Nguyễn Ánh Cương</td>
                <td>Phát triển toàn bộ mã nguồn, kiểm thử, triển khai dự án.</td>
            </tr>
            <tr>
                <td>Vũ Khánh Hoàn</td>
                <td>Biên soạn tài liệu Overleaf, Poster, Powerpoint, thuyết trình, thực hiện video giới thiệu 
                    và hỗ trợ bài tập lớn.</td>
            </tr>
            <tr>
                <td>Nông Trung Hiếu</td>
                <td>Biên soạn tài liệu Overleaf, Thiết kế slide.</td>
            </tr>
        </tbody>
</table>

2025 NHÓM 9, CNTT16-02, TRƯỜNG ĐẠI HỌC ĐẠI NAM

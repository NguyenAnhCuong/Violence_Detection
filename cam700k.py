import cv2
import numpy as np
import tensorflow as tf
import os
import time
from playsound import playsound  # Thêm playsound để phát âm thanh báo động
import threading  # Chạy âm thanh song song để không làm chậm chương trình

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("./3k-96-30.h5")

# Địa chỉ RTSP của Ezviz camera
rtsp_url = "rtsp://admin:RAJVMI@192.168.1.12:554/h264_stream"

# Kết nối với camera Ezviz
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 16)  # Giảm buffer giúp giảm delay
cap.set(cv2.CAP_PROP_FPS, 16)  # Giới hạn FPS giúp giảm tải

if not cap.isOpened():
    print("Không thể kết nối với camera.")
    exit()

save_path = "saved_frames"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Thông số của mô hình
frame_size = (96, 96)
frame_count = 30
channels = 3
violence_threshold = 0.95
alarm_sound = "./clock-alarm.mp3"  # Đường dẫn đến file âm thanh báo động

# Bộ nhớ đệm để lưu khung hình
frames = []
violence_frames = []

def play_alarm():
    """Hàm phát còi báo động khi phát hiện bạo lực"""
    playsound(alarm_sound)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi đọc frame từ camera.")
        break

    frame_display = cv2.resize(frame, (800, 600))

    frame_resized = cv2.resize(frame, frame_size)
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_resized = frame_resized / 255.0  # Chuẩn hóa dữ liệu

    frames.append(frame_resized)

    if len(frames) == frame_count:
        video_clip = np.array(frames, dtype=np.float32)
        video_clip = np.expand_dims(video_clip, axis=0)

        prediction = model.predict(video_clip)
        violence_prob = prediction[0, 0]
        non_violence_prob = 1 - violence_prob
        label = "Bạo lực" if violence_prob > violence_threshold else "Không bạo lực"
        color = (0, 0, 255) if violence_prob > violence_threshold else (0, 255, 0)
        
        print(f"Dự đoán: {label} - Bạo lực: {violence_prob:.2f} - Không bạo lực: {non_violence_prob:.2f}")

        if violence_prob > violence_threshold:  
            # violence_frames.extend(frames)

            # if len(violence_frames) >= frame_count:
            #     timestamp = time.strftime("%Y%m%d_%H%M%S")
            #     for i, vf in enumerate(violence_frames[:frame_count]):
            #         image_filename = f"{save_path}/{timestamp}_baoluc_{i}.jpg"
            #         cv2.imwrite(image_filename, (vf * 255).astype(np.uint8))
            #         print(f"🚨 Ảnh đã lưu: {image_filename}")

            #     violence_frames = violence_frames[frame_count:]

            # Chạy âm thanh báo động trên luồng riêng để không làm chậm chương trình
            threading.Thread(target=play_alarm, daemon=True).start()
        cv2.putText(frame_display, f"{label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Giữ lại 4 khung hình cuối để tạo độ trượt
        frames = frames[-4:]

    # Hiển thị camera với overlay thông tin dự đoán
    cv2.imshow("Ezviz Camera", frame_display)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("./3k-96-30.h5")

# Địa chỉ RTSP của Ezviz camera (thay thông tin của bạn)
rtsp_url = "rtsp://admin:RAJVMI@192.168.30.80:554/h264_stream"


# Kết nối với camera Ezviz
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 16)  # Giảm buffer giúp giảm delay
cap.set(cv2.CAP_PROP_FPS, 16)  # Giới hạn FPS giúp giảm tải

# Kiểm tra kết nối
if not cap.isOpened():
    print("Không thể kết nối với camera.")
    exit()

save_path = "saved_frames"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Thông số của mô hình
frame_size = (96, 96)  # Kích thước ảnh đầu vào cho model
frame_count = 30        # Số khung hình đầu vào của mô hình
channels = 3           # RGB có 3 kênh màu
violence_threshold = 0.95

# Bộ nhớ đệm để lưu khung hình
frames = []
violence_frames = []  # Bộ nhớ đệm để lưu khung hình khi phát hiện bạo lực

while True:
    start_time = time.time()  # Bắt đầu đo thời gian

    ret, frame = cap.read()
    if not ret:
        print("Lỗi đọc frame từ camera.")
        break

    # Hiển thị camera với độ phân giải gốc 1280x720
    frame_display = cv2.resize(frame, (800, 600))

    # Resize ảnh về 96x96 để phù hợp với mô hình
    frame_resized = cv2.resize(frame, frame_size)
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    frame_resized = frame_resized / 255.0  # Chuẩn hóa dữ liệu về khoảng [0,1]
    
    frames.append(frame_resized)

    # Nếu đủ 30 khung hình, tiến hành dự đoán
    if len(frames) == frame_count:
        # Chuyển đổi thành tensor với shape (1, 30, 96, 96, 3)
        video_clip = np.array(frames, dtype=np.float32)
        video_clip = np.expand_dims(video_clip, axis=0)

        # Dự đoán bằng mô hình CNN 3D
        prediction_start_time = time.time()  # Bắt đầu đo thời gian dự đoán
        prediction = model.predict(video_clip)
        prediction_end_time = time.time()  # Kết thúc đo thời gian dự đoán

        predicted_class = np.argmax(prediction, axis=1)[0]

        # Hiển thị kết quả dự đoán
        violence_prob = prediction[0, 0]
        non_violence_prob = 1 - violence_prob
        label = "Bao luc" if violence_prob > violence_threshold else "Khong bao luc"
        color = (0, 0, 255) if violence_prob > violence_threshold else (0, 255, 0)

        end_time = time.time()  # Kết thúc đo thời gian
        total_time = end_time - start_time
        prediction_time = prediction_end_time - prediction_start_time
        
        cv2.putText(frame_display, f"Du doan: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        print(f"Dự đoán: {label} - Bao luc: {violence_prob:.2f} - ko bao luc: {non_violence_prob:.2f}")
        print(f"Thời gian xử lý tổng cộng: {total_time:.4f} giây")
        print(f"Thời gian dự đoán: {prediction_time:.4f} giây")
        if violence_prob > violence_threshold:
            # Vẽ bounding box quanh toàn bộ khung hình
            cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], frame_display.shape[0]), color, 2)
            
            # Lưu 30 khung hình liên tiếp khi phát hiện bạo lực
            # violence_frames.extend(frames)
            # if len(violence_frames) >= frame_count:
            #     timestamp = time.strftime("%Y%m%d_%H%M%S")
            #     for i, vf in enumerate(violence_frames[:frame_count]):
            #         image_filename = f"{save_path}/{timestamp}_baoluc_{i}.jpg"
            #         cv2.imwrite(image_filename, (vf * 255).astype(np.uint8))  # Lưu ảnh gốc
            #         print(f"🚨 Ảnh đã lưu: {image_filename}")
            #     violence_frames = violence_frames[frame_count:]  # Xóa các khung hình đã lưu

        # Giữ lại 4 khung hình cuối để tạo độ trượt (Sliding Window)
        frames = frames[-4:]

    # Hiển thị camera với overlay thông tin dự đoán
    cv2.imshow("Ezviz Camera", frame_display)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

    

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
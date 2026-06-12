import cv2, mediapipe as mp, csv, os
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    output_segmentation_masks=False
)
pose = vision.PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# 시퀀스 길이
SEQ_LENGTH = 30

COLS = ["sequence_id", "frame_idx"] + [f"lm_{i}_{c}" for i in range(33) for c in ["x","y","z","v"]] + ["label"]

print("="*50)
print("시계열 데이터 수집기")
mode = input("데이터를 저장할 폴더를 선택하세요 (1: Train, 2: Test) [기본값: 1]: ")
if mode == '2':
    save_dir = "data/test"
    print(">> 'data/test' 폴더에 저장합니다.")
else:
    save_dir = "data/train"
    print(">> 'data/train' 폴더에 저장합니다.")

csv_path = os.path.join(save_dir, "pose_sequence_data.csv")
is_new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if is_new_file:
        writer.writerow(COLS)

label_map = {"g": "Good", "b": "Bad", "w": "Warning"}
current_label = None
print("="*50)
print("g=Good, b=Bad, w=Warning 세팅 후")
print("스페이스바(Space)를 누르면 30프레임 동안 녹화합니다.")
print("q=종료")
print("="*50)

sequence_buffer = []
is_recording = False
sequence_id = 0

# 기존 sequence_id 찾기
if not is_new_file:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        if not df.empty and "sequence_id" in df.columns:
            sequence_id = df["sequence_id"].max() + 1
    except:
        pass

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = pose.detect(mp_image)

    row = []
    if results.pose_landmarks:
        lm = results.pose_landmarks[0]
        for point in lm:
            row.extend([point.x, point.y, point.z, point.visibility])
            
        if is_recording:
            sequence_buffer.append(row)
            cv2.putText(frame, f"Rec... {len(sequence_buffer)}/{SEQ_LENGTH}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if len(sequence_buffer) == SEQ_LENGTH:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for idx, seq_row in enumerate(sequence_buffer):
                        writer.writerow([sequence_id, idx] + seq_row + [current_label])
                print(f"[{sequence_id}] {current_label} 시퀀스 저장 완료!")
                sequence_id += 1
                is_recording = False
                sequence_buffer.clear()

    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sequence Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif chr(key) in label_map:
        current_label = label_map[chr(key)]
        print(f"라벨 변경: {current_label}")
    elif key == ord(" ") and current_label and not is_recording:
        print(f"{current_label} 데이터 수집 시작...")
        is_recording = True
        sequence_buffer.clear()

cap.release()
cv2.destroyAllWindows()
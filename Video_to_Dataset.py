import cv2
import mediapipe as mp
import csv
import os
import glob
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EXERCISES = ("squat", "pushup", "pullup")
LABELS = ("Good", "Bad")

def setup_pose_detector():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)

def process_video(video_path, label, detector, csv_writer, sequence_id_start, seq_length=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return sequence_id_start
    
    sequence_buffer = []
    current_seq_id = sequence_id_start
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 웹캠(local_test_v2.py)과 동일하게 좌우 반전(거울 모드) 적용
        frame = cv2.flip(frame, 1)
            
        # Mediapipe 처리를 위해 RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks[0]
            
            # 골반 중심점 및 몸통 길이 계산 (상대 좌표 정규화)
            cx = (lm[23].x + lm[24].x) / 2.0
            cy = (lm[23].y + lm[24].y) / 2.0
            cz = (lm[23].z + lm[24].z) / 2.0
            
            sx = (lm[11].x + lm[12].x) / 2.0
            sy = (lm[11].y + lm[12].y) / 2.0
            sz = (lm[11].z + lm[12].z) / 2.0
            
            dist = ((cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2) ** 0.5
            scale = dist if dist > 0.01 else 1.0
            
            row = []
            for point in lm:
                nx = (point.x - cx) / scale
                ny = (point.y - cy) / scale
                nz = (point.z - cz) / scale
                row.extend([nx, ny, nz, point.visibility])
            
            sequence_buffer.append(row)
            
            # 버퍼가 30프레임에 도달하면 파일에 기록하고 초기화
            if len(sequence_buffer) == seq_length:
                for idx, seq_row in enumerate(sequence_buffer):
                    csv_writer.writerow([current_seq_id, idx] + seq_row + [label])
                
                current_seq_id += 1
                sequence_buffer.clear()
        
    cap.release()
    print(f"[{label}] {os.path.basename(video_path)} 처리 완료 -> {current_seq_id - sequence_id_start}개의 시퀀스 생성")
    return current_seq_id

def main():
    video_base_dir = "videos"
    COLS = ["sequence_id", "frame_idx"] + [f"lm_{i}_{c}" for i in range(33) for c in ["x","y","z","v"]] + ["label"]
    
    print("Mediapipe 모델을 로드하는 중...")
    detector = setup_pose_detector()
    
    print("="*50)
    print("🎬 동영상 데이터셋 변환기")
    print("1. 'videos/squat/Good', 'videos/squat/Bad' 처럼 운동별 폴더에 동영상을 넣어주세요.")
    print("   사용 가능 운동: squat, pushup, pullup")
    print("2. 스크립트가 해당 영상들을 읽어 30프레임 단위로 쪼갠 뒤 자동 저장합니다.")
    print("="*50)
    
    videos_found = False
    for exercise in EXERCISES:
        output_dir = os.path.join("data", exercise, "train")
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "video_pose_data.csv")
        is_new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

        for label in LABELS:
            os.makedirs(os.path.join(video_base_dir, exercise, label), exist_ok=True)

        sequence_id = 0
        if not is_new_file:
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and "sequence_id" in df.columns:
                    sequence_id = df["sequence_id"].max() + 1
            except Exception:
                pass

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(COLS)

            for label in LABELS:
                label_dir = os.path.join(video_base_dir, exercise, label)
                video_files = []
                for ext in ('*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV'):
                    video_files.extend(glob.glob(os.path.join(label_dir, ext)))

                for video_file in video_files:
                    videos_found = True
                    sequence_id = process_video(video_file, label, detector, writer, sequence_id)
                
    if not videos_found:
        print("\n[알림] 영상을 찾을 수 없습니다.")
        print(f"👉 프로젝트 폴더 내 '{video_base_dir}/squat/Good' 등에 테스트용 동영상을 먼저 넣어주세요!")
    else:
        print("\n🎉 변환 완료! 운동별 data/{exercise}/train/video_pose_data.csv 에 누적 저장되었습니다.")

if __name__ == "__main__":
    main()

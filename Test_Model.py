import cv2
import mediapipe as mp
import os
import glob
import torch
import numpy as np
from Video_to_Dataset import setup_pose_detector, EXERCISES, LABELS
from Classifier import PoseClassifier

def evaluate_video(video_path, expected_label, detector, classifier, seq_length=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []

    predictions = []
    sequence_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 웹캠/학습 데이터셋 변환기와 동일하게 좌우 반전(거울 모드) 적용
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
            
            # 버퍼가 30프레임에 도달하면 모델에 예측 요청
            if len(sequence_buffer) == seq_length:
                seq_tensor = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    outputs = classifier.model(seq_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    pred_label = classifier.labels[predicted.item()]
                    predictions.append(pred_label)
                
                sequence_buffer.clear()
                
    cap.release()
    return predictions

def main():
    test_video_dir = "test_videos"
    print("="*50)
    print("📊 모델 성능 평가기 (Test Videos 기반)")
    print(f"'{test_video_dir}/운동명/Good' 또는 'Bad' 폴더에 있는 영상을 분석하여")
    print("현재 학습된 모델의 실제 영상 대비 정확도를 평가합니다.")
    print("="*50)
    
    print("Mediapipe 모델을 로드하는 중...")
    detector = setup_pose_detector()
    
    for exercise in EXERCISES:
        model_path = f"models/{exercise}_gru.pth"
        if not os.path.exists(model_path):
            continue
            
        print(f"\n▶ [{exercise.upper()}] 모델 평가 시작...")
        classifier = PoseClassifier(exercise=exercise, seq_length=30)
        if not classifier.ready:
            print(f"  {exercise} 모델을 로드할 수 없습니다.")
            continue
            
        total_seqs = 0
        correct_seqs = 0
        videos_found = False
        
        for label in LABELS:
            label_dir = os.path.join(test_video_dir, exercise, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir, exist_ok=True)
                continue
                
            video_files = []
            for ext in ('*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV'):
                video_files.extend(glob.glob(os.path.join(label_dir, ext)))
                
            if not video_files:
                continue
                
            videos_found = True
            for video_file in video_files:
                preds = evaluate_video(video_file, label, detector, classifier)
                
                seq_count = len(preds)
                if seq_count == 0:
                    print(f"  - {os.path.basename(video_file)}: 추출된 시퀀스 없음 (영상이 너무 짧거나 사람이 인식되지 않음)")
                    continue
                    
                correct = sum(1 for p in preds if p == label)
                
                total_seqs += seq_count
                correct_seqs += correct
                
                acc = (correct / seq_count) * 100
                print(f"  - {os.path.basename(video_file)} ({label} 폴더): {acc:.1f}% 정확도 ({correct}/{seq_count} 시퀀스 일치)")
                
        if not videos_found:
            print(f"  테스트할 영상이 없습니다. '{test_video_dir}/{exercise}/Good (또는 Bad)' 폴더에 테스트용 영상을 넣어주세요.")
        elif total_seqs > 0:
            total_acc = (correct_seqs / total_seqs) * 100
            print(f">>> 🏆 [{exercise.upper()}] 최종 테스트 성능: {total_acc:.2f}% ({correct_seqs}/{total_seqs} 시퀀스 정답)")

if __name__ == "__main__":
    main()

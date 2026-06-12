import torch
import numpy as np
from RNN_Trainer import EXERCISES, PoseGRU

class PoseClassifier:
    def __init__(self, exercise: str, model_path=None, seq_length=30):
        if exercise not in EXERCISES:
            raise ValueError(f"지원하지 않는 운동입니다: {exercise}")

        self.seq_length = seq_length
        self.ready = False
        self.buffer = []
        self.labels = ["Good", "Bad"]
        self.exercise = exercise
        self.model_path = model_path or f"models/{exercise}_gru.pth"
        
        try:
            self.model = PoseGRU()
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
            self.model.eval()
            self.ready = True
        except Exception as e:
            print(f"[{exercise}] GRU 모델 로드 실패 (또는 아직 학습되지 않음): {e}")
            self.model = None

    def predict(self, feature_vector: np.ndarray) -> str:
        """
        새로운 프레임 피처를 버퍼에 추가한 뒤,
        길이가 seq_length에 도달하면 운동별 GRU 모델로 Good/Bad 예측 수행
        """
        self.buffer.append(feature_vector)
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)
            
        if not self.ready or len(self.buffer) < self.seq_length:
            return "Good"
            
        try:
            # shape: (1, seq_length, num_features)
            seq_tensor = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(seq_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
            return self.labels[predicted.item()]
        except Exception as e:
            print(f"예측 중 에러: {e}")
            return "Good"

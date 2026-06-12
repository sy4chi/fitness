import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import argparse

SEQ_LENGTH = 30
FEATURES = 132 # 33 * 4 (x,y,z,v)
EXERCISES = ("squat", "pushup", "pullup")
LABEL_MAP = {"Good": 0, "Bad": 1}
LABEL_NAMES = ["Good", "Bad"]

# --- 정밀한 풀업 합성 불량 자세 생성 유틸리티 ---

def make_shallow_pullup(seq):
    """턱걸이 시 올라가는 깊이가 너무 얕은 불량 자세 합성 (어깨 기준 팔꿈치/손목의 수직 수축 범위를 40~60% 축소)"""
    bad_seq = seq.copy()
    alpha = np.random.uniform(0.4, 0.6)
    for t in range(SEQ_LENGTH):
        # 어깨(11: L_shoulder, 12: R_shoulder)의 y 좌표 평균 계산
        sy = (bad_seq[t, 11 * 4 + 1] + bad_seq[t, 12 * 4 + 1]) / 2.0
        # 팔꿈치와 손목(13, 14, 15, 16)의 어깨 대비 수직 상대 거리를 좁힘
        for j in [13, 14, 15, 16]:
            y_diff = bad_seq[t, j * 4 + 1] - sy
            bad_seq[t, j * 4 + 1] = sy + y_diff * alpha
    return bad_seq

def make_asymmetric_pullup(seq):
    """한쪽 팔로만 당기는 좌우 비대칭 불균형 자세 합성 (왼쪽 팔만 수직 운동 범위를 30~50% 축소)"""
    bad_seq = seq.copy()
    alpha = np.random.uniform(0.3, 0.5)
    for t in range(SEQ_LENGTH):
        sy = bad_seq[t, 11 * 4 + 1] # 왼쪽 어깨 y
        for j in [13, 15]: # 왼쪽 팔꿈치, 왼쪽 손목
            y_diff = bad_seq[t, j * 4 + 1] - sy
            bad_seq[t, j * 4 + 1] = sy + y_diff * alpha
    return bad_seq

def make_swing_pullup(seq):
    """철봉에서 과도하게 몸을 흔드는 반동 스윙 자세 합성 (골반 기준 상체 전체를 x축으로 좌우 sinusoidal 흔들림 주입)"""
    bad_seq = seq.copy()
    amp = np.random.uniform(0.18, 0.32)
    phase = np.random.uniform(0, 2 * np.pi)
    for t in range(SEQ_LENGTH):
        x_drift = amp * np.sin(2 * np.pi * t / SEQ_LENGTH + phase)
        # 상체 및 머리 관절들(0: head/nose, 11~16: shoulders/elbows/wrists)만 흔들림 인가
        for j in [0, 11, 12, 13, 14, 15, 16]:
            bad_seq[t, j * 4] += x_drift
    return bad_seq

def make_jitter(seq):
    """카메라 지터 현상과 센서 노이즈 극복을 위한 가우시안 지터 증강"""
    bad_seq = seq.copy()
    noise = np.random.normal(0, np.random.uniform(0.005, 0.012), size=seq.shape)
    for j in range(33):
        noise[:, j * 4 + 3] = 0 # visibility는 유지
    return bad_seq + noise


class PoseDataset(Dataset):
    def __init__(self, data_dir, exercise, is_train=True):
        self.sequences = []
        self.labels = []
        self.is_train = is_train
        self.exercise = exercise
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            return
            
        raw_sequences = []
        raw_labels = []
        
        # 1. 파일에서 오리지널 데이터 로드
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for seq_id, group in df.groupby("sequence_id"):
                    if len(group) == SEQ_LENGTH:
                        features = group.iloc[:, 2:-1].values.astype(np.float32)
                        label_str = group.iloc[0, -1]
                        if label_str not in LABEL_MAP:
                            continue
                        raw_sequences.append(features)
                        raw_labels.append(LABEL_MAP[label_str])
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                
        # 2. 데이터 증강 및 합성 엔진 작동
        if self.is_train and len(raw_sequences) > 0:
            for seq, label in zip(raw_sequences, raw_labels):
                # 기본 원본 추가
                self.sequences.append(seq)
                self.labels.append(label)
                
                # 미세 지터링 추가 (카메라 노이즈 방지용 - 모든 종목 공통 적용)
                self.sequences.append(make_jitter(seq))
                self.labels.append(label)
                
                # 풀업(Pullup)인 경우에만 0개의 Bad 데이터를 채우기 위한 정밀 물리 합성 수행
                if self.exercise == "pullup" and label == 0:
                    # 합성 Bad 1: 얕은 풀업
                    self.sequences.append(make_shallow_pullup(seq))
                    self.labels.append(1)
                    
                    # 합성 Bad 2: 비대칭 풀업
                    self.sequences.append(make_asymmetric_pullup(seq))
                    self.labels.append(1)
                    
                    # 합성 Bad 3: 스윙 반동 풀업
                    self.sequences.append(make_swing_pullup(seq))
                    self.labels.append(1)
        else:
            # 평가/검증 세트는 원본 그대로 유지
            self.sequences = raw_sequences
            self.labels = raw_labels
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.is_train:
            online_noise = np.random.normal(0, 0.003, size=seq.shape).astype(np.float32)
            for j in range(33):
                online_noise[:, j * 4 + 3] = 0
            seq = seq + online_noise
            
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class PoseGRU(nn.Module):
    def __init__(self, input_size=FEATURES, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super(PoseGRU, self).__init__()
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def build_dirs(exercise):
    return os.path.join("data", exercise, "train"), os.path.join("data", exercise, "test")


def train(exercise):
    if exercise not in EXERCISES:
        print(f"지원하지 않는 운동입니다: {exercise} (가능: {', '.join(EXERCISES)})")
        return False

    train_dir, test_dir = build_dirs(exercise)
    
    if not os.path.exists(train_dir):
        print(f"학습 데이터 폴더가 없습니다: {train_dir}")
        return False
        
    full_train_dataset = PoseDataset(train_dir, exercise, is_train=True)
    test_dataset = PoseDataset(test_dir, exercise, is_train=False)
    
    if len(full_train_dataset) == 0:
        print(f"[{exercise}] 유효한 시퀀스 데이터가 없습니다.")
        return False
        
    if len(test_dataset) == 0:
        if len(full_train_dataset) < 10:
            train_dataset = full_train_dataset
            val_dataset = full_train_dataset
        else:
            val_size = int(0.2 * len(full_train_dataset))
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            if hasattr(val_dataset, 'dataset'):
                val_dataset.dataset.is_train = False
    else:
        train_dataset = full_train_dataset
        val_dataset = test_dataset
        
    print(f"[{exercise.upper()}] 정제 증강 후 Train 크기: {len(train_dataset)} 시퀀스, Val/Test 크기: {len(val_dataset)} 시퀀스")
    
    batch_size = min(16, len(train_dataset)) if len(train_dataset) > 0 else 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 클래스 분포 및 가중치 계산
    all_labels = []
    for _, l in train_loader:
        all_labels.extend(l.numpy())
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels, minlength=2)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights) * 2.0
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[{exercise.upper()}] 클래스 분포: Good={class_counts[0]}, Bad={class_counts[1]} -> 손실 가중치: {class_weights}")
    
    model = PoseGRU(dropout=0.2) # 적절한 규제화를 위해 dropout=0.2 적용
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01) # 최적의 초기 학습률
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    
    epochs = 40
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for seqs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} (Acc: {train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.1f}%)")
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for seqs, labels in val_loader:
            outputs = model(seqs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    print(f"🏆 [{exercise.upper()}] 최종 Best Validation Accuracy: {100 * val_correct / val_total:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{exercise}_gru.pth")
    torch.save(model.state_dict(), model_path)
    print(f"최적 성능 모델 저장 완료 -> {model_path}")
    
    # ── 웹 브라우저용 ONNX 자동 컴파일/내보내기 (추가 수동 변환 과정 제거) ──
    try:
        onnx_path = os.path.join("models", f"{exercise}.onnx")
        # 입력 구조 매칭: 배치 크기=1, 시퀀스 길이=30, 특징 개수=132
        dummy_input = torch.randn(1, SEQ_LENGTH, FEATURES)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12
        )
        print(f"🤖 웹 브라우저용 최신 ONNX 모델 자동 변환 완료 -> {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX 자동 변환 실패 (오류: {e})")
        
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="운동별 합성 증강 기법을 적용한 고강도 GRU 자세 분류 모델 학습")
    parser.add_argument(
        "--exercise",
        choices=EXERCISES + ("all",),
        default="all",
        help="학습할 운동 종목. 기본값은 all (전체 학습)",
    )
    args = parser.parse_args()

    targets = EXERCISES if args.exercise == "all" else (args.exercise,)
    for target in targets:
        train(target)

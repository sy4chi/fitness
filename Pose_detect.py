#필요 라이브러리 다운로드 시 주석해제 바람.
#pip install fastapi uvicorn websockets mediapipe opencv-python numpy pandas scikit-learn joblib python-multipart
import mediapipe as mp
import numpy as np
import cv2
import base64

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseLandmark:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

LANDMARK = PoseLandmark

class PoseDetect:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def decode_frame(self, b64_str: str) -> np.ndarray:
        """Base64 → OpenCV 이미지 변환"""
        img_data = base64.b64decode(b64_str.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def extract_landmarks(self, frame: np.ndarray):
        """MediaPipe로 랜드마크 추출"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks[0]
        # (x, y, z, visibility) × 33개 → 132차원 벡터
        return lm

    VIS_THRESH = 0.65

    def visible(self, lm, *indices) -> bool:
        return all(lm[idx].visibility >= self.VIS_THRESH for idx in indices)

    def best_angle(self, lm, left_indices, right_indices):
        lv = min(lm[idx].visibility for idx in left_indices)
        rv = min(lm[idx].visibility for idx in right_indices)
        if lv < self.VIS_THRESH and rv < self.VIS_THRESH:
            return None
        if lv >= rv:
            a, b, c = [lm[idx] for idx in left_indices]
        else:
            a, b, c = [lm[idx] for idx in right_indices]
        return self.calculate_angle(a, b, c)

    def calculate_angle(self, a, b, c) -> float:
        """
        세 점 A-B-C 사이의 각도 계산 (B가 꼭짓점)
        arctan2를 이용해 2D 벡터 각도 차이 계산
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def get_exercise_angles(self, lm, exercise: str) -> dict:
        """운동별 핵심 관절 각도 추출 (가시성 체크 적용)"""
        angles = {}
        if exercise == "squat":
            knee = self.best_angle(lm,
                                   [LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE, LANDMARK.LEFT_ANKLE],
                                   [LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE, LANDMARK.RIGHT_ANKLE])
            trunk = self.best_angle(lm,
                                    [LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_KNEE],
                                    [LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_KNEE])
            if knee is not None: angles["knee"] = knee
            if trunk is not None: angles["trunk"] = trunk

        elif exercise == "pushup":
            elbow = self.best_angle(lm,
                                    [LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST],
                                    [LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST])
            body_line = self.best_angle(lm,
                                        [LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_HIP, LANDMARK.LEFT_ANKLE],
                                        [LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_HIP, LANDMARK.RIGHT_ANKLE])
            if elbow is not None: angles["elbow"] = elbow
            if body_line is not None: angles["body_line"] = body_line

        elif exercise == "pullup":
            if self.visible(lm, LANDMARK.LEFT_SHOULDER, LANDMARK.LEFT_ELBOW, LANDMARK.LEFT_WRIST):
                angles["left_elbow"] = self.calculate_angle(
                    lm[LANDMARK.LEFT_SHOULDER], lm[LANDMARK.LEFT_ELBOW], lm[LANDMARK.LEFT_WRIST])
            if self.visible(lm, LANDMARK.RIGHT_SHOULDER, LANDMARK.RIGHT_ELBOW, LANDMARK.RIGHT_WRIST):
                angles["right_elbow"] = self.calculate_angle(
                    lm[LANDMARK.RIGHT_SHOULDER], lm[LANDMARK.RIGHT_ELBOW], lm[LANDMARK.RIGHT_WRIST])

        return angles

    def get_feature_vector(self, lm) -> np.ndarray:
        """
        GRU 모델 입력용 132차원 벡터 (33 랜드마크 × 4)
        상대 좌표계 변환: 골반 중심(0,0,0) 및 몸통 길이 기준 정규화
        """
        # 1. 골반 중심점 계산 (23: left_hip, 24: right_hip)
        cx = (lm[23].x + lm[24].x) / 2.0
        cy = (lm[23].y + lm[24].y) / 2.0
        cz = (lm[23].z + lm[24].z) / 2.0

        # 2. 어깨 중심점 계산 (11: left_shoulder, 12: right_shoulder)
        sx = (lm[11].x + lm[12].x) / 2.0
        sy = (lm[11].y + lm[12].y) / 2.0
        sz = (lm[11].z + lm[12].z) / 2.0

        # 3. 몸통 길이 (스케일)
        dist = ((cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2) ** 0.5
        scale = dist if dist > 0.01 else 1.0

        coords = []
        for point in lm:
            nx = (point.x - cx) / scale
            ny = (point.y - cy) / scale
            nz = (point.z - cz) / scale
            coords.extend([nx, ny, nz, point.visibility])
            
        return np.array(coords, dtype=np.float32)
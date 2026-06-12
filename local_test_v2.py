import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
import os, urllib.request
from TTS_Manager import TTSManager
from Classifier import PoseClassifier
from PIL import ImageFont, ImageDraw, Image

# macOS 기본 한글 폰트
font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

font_cache = {}

def put_korean_texts(img, text_jobs):
    if not text_jobs:
        return img

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    for text, pos, font_size, color in text_jobs:
        if font_size not in font_cache:
            font_cache[font_size] = ImageFont.truetype(font_path, font_size)

        font = font_cache[font_size]
        draw.text(
            pos,
            text,
            font=font,
            fill=(color[0], color[1], color[2])
        )

    return np.array(img_pil)

# ── 모델 다운로드 ──────────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
if not os.path.exists(MODEL_PATH):
    print("모델 다운로드 중... (약 7MB)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("완료")

# ── 랜드마크 인덱스 ────────────────────────────────────────────────────────────
LM = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow": 13,     "right_elbow": 14,
    "left_wrist": 15,     "right_wrist": 16,
    "left_hip": 23,       "right_hip": 24,
    "left_knee": 25,      "right_knee": 26,
    "left_ankle": 27,     "right_ankle": 28,
}

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

CAM_GUIDE = {
    "squat":  "카메라: 측면  |  발끝 ~ 머리 전신",
    "pushup": "카메라: 측면  |  손목 ~ 발목 전신",
    "pullup": "카메라: 후면  |  전신 후면",
}

VIS_THRESH = 0.65


# ── 1. 각도 계산 ───────────────────────────────────────────────────────────────
def calculate_angle(a, b, c) -> float:
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def visible(lm_list, *names) -> bool:
    return all(lm_list[LM[n]].visibility >= VIS_THRESH for n in names)

def best_angle(lm_list, left_pts, right_pts):
    lv = min(lm_list[LM[n]].visibility for n in left_pts)
    rv = min(lm_list[LM[n]].visibility for n in right_pts)
    if lv < VIS_THRESH and rv < VIS_THRESH:
        return None
    if lv >= rv:
        a, b, c = [lm_list[LM[n]] for n in left_pts]
    else:
        a, b, c = [lm_list[LM[n]] for n in right_pts]
    return calculate_angle(a, b, c)

def get_angles(lm_list, exercise: str) -> dict:
    angles = {}
    if exercise == "squat":
        knee  = best_angle(lm_list,
                           ["left_hip","left_knee","left_ankle"],
                           ["right_hip","right_knee","right_ankle"])
        trunk = best_angle(lm_list,
                           ["left_shoulder","left_hip","left_knee"],
                           ["right_shoulder","right_hip","right_knee"])
        if knee  is not None: angles["knee"]  = knee
        if trunk is not None: angles["trunk"] = trunk

    elif exercise == "pushup":
        elbow = best_angle(lm_list,
                           ["left_shoulder","left_elbow","left_wrist"],
                           ["right_shoulder","right_elbow","right_wrist"])
        body  = best_angle(lm_list,
                           ["left_shoulder","left_hip","left_ankle"],
                           ["right_shoulder","right_hip","right_ankle"])
        if elbow is not None: angles["elbow"] = elbow
        if body  is not None: angles["body"]  = body

    elif exercise == "pullup":
        if visible(lm_list, "left_shoulder","left_elbow","left_wrist"):
            angles["L.elbow"] = calculate_angle(
                lm_list[LM["left_shoulder"]],
                lm_list[LM["left_elbow"]],
                lm_list[LM["left_wrist"]])
        if visible(lm_list, "right_shoulder","right_elbow","right_wrist"):
            angles["R.elbow"] = calculate_angle(
                lm_list[LM["right_shoulder"]],
                lm_list[LM["right_elbow"]],
                lm_list[LM["right_wrist"]])
    return angles


def get_feature_vector(lm_list) -> np.ndarray:
    """
    GRU 모델 입력용 132차원 벡터 (33 랜드마크 × 4)
    상대 좌표계 변환: 골반 중심(0,0,0) 및 몸통 길이 기준 정규화
    """
    # 1. 골반 중심점 계산 (23: left_hip, 24: right_hip)
    cx = (lm_list[23].x + lm_list[24].x) / 2.0
    cy = (lm_list[23].y + lm_list[24].y) / 2.0
    cz = (lm_list[23].z + lm_list[24].z) / 2.0

    # 2. 어깨 중심점 계산 (11: left_shoulder, 12: right_shoulder)
    sx = (lm_list[11].x + lm_list[12].x) / 2.0
    sy = (lm_list[11].y + lm_list[12].y) / 2.0
    sz = (lm_list[11].z + lm_list[12].z) / 2.0

    # 3. 몸통 길이 (스케일)
    dist = ((cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2) ** 0.5
    scale = dist if dist > 0.01 else 1.0

    coords = []
    for point in lm_list:
        nx = (point.x - cx) / scale
        ny = (point.y - cy) / scale
        nz = (point.z - cz) / scale
        coords.extend([nx, ny, nz, point.visibility])
        
    return np.array(coords, dtype=np.float32)


# ── 2. 상태 머신 ───────────────────────────────────────────────────────────────
THRESHOLDS = {
    "squat":  {"key": "knee",    "down": 95,  "up": 155, "inv": False},
    "pushup": {"key": "elbow",   "down": 85,  "up": 150, "inv": False},
    "pullup": {"key": "L.elbow", "key2": "R.elbow",
               "down": 150, "up": 65, "inv": True},
}

class StateMachine:
    def __init__(self):
        self.state = "ready"
        self.count = 0
        self.rep_scores   = []
        self._down_scores = []

    def reset(self):
        self.state = "ready"; self.count = 0
        self.rep_scores = []; self._down_scores = []

    def update(self, angles: dict, exercise: str, quality: float):
        cfg = THRESHOLDS[exercise]
        prev_state = self.state

        if "key2" in cfg:
            vals  = [angles[k] for k in (cfg["key"], cfg["key2"]) if k in angles]
            angle = np.mean(vals) if vals else (0 if cfg["inv"] else 180)
        else:
            angle = angles.get(cfg["key"], 0 if cfg["inv"] else 180)

        count_increased = False

        if not cfg["inv"]:
            if self.state == "ready" and angle < cfg["down"]:
                self.state = "down"; self._down_scores = []
            elif self.state == "down":
                self._down_scores.append(quality)
                if angle > cfg["up"]:
                    self.state = "ready"; self.count += 1
                    self.rep_scores.append(max(self._down_scores) if self._down_scores else quality)
                    self._down_scores = []
                    count_increased = True
        else:
            if self.state == "ready" and angle > cfg["down"]:
                self.state = "down"; self._down_scores = []
            elif self.state == "down":
                self._down_scores.append(quality)
                if angle < cfg["up"]:
                    self.state = "ready"; self.count += 1
                    self.rep_scores.append(max(self._down_scores) if self._down_scores else quality)
                    self._down_scores = []
                    count_increased = True

        return self.state, (prev_state != self.state), count_increased


# ── 3. 점수 계산 ───────────────────────────────────────────────────────────────
IDEAL = {
    "squat":  {"knee": 90,    "trunk": 70},
    "pushup": {"elbow": 90,   "body": 175},
    "pullup": {"L.elbow": 55, "R.elbow": 55},
}

def calc_score(angles, exercise, shoulder_history, state, pose_label="Good"):
    ideal = IDEAL.get(exercise, {})
    if state == "down" and angles:
        errors   = [min(abs(angles[j]-t), 90) for j,t in ideal.items() if j in angles]
        accuracy = 100 - (np.mean(errors)*(100/90)) if errors else 50.0
    else:
        accuracy = None

    stability = 100.0
    if len(shoulder_history) >= 10:
        std = np.std(shoulder_history[-10:])
        stability = max(0.0, 100 - std * 800)

    # GRU 분류 결과 반영: Bad이면 30점 감점
    label_penalty = {"Good": 0, "Bad": 30}.get(pose_label, 0)

    if accuracy is not None:
        total = accuracy * 0.6 + stability * 0.4 - label_penalty
    else:
        total = stability - label_penalty

    return {
        "total":     round(max(0, min(100, total)), 1),
        "accuracy":  round(accuracy, 1) if accuracy is not None else None,
        "stability": round(stability, 1),
        "pose_label": pose_label,
        "measuring": state == "down",
    }


# ── 4. 피드백 생성 ─────────────────────────────────────────────────────────────
def get_feedback(angles, exercise, score, state) -> str:
    """
    반환값 규칙
    - ""            : TTS 묵음 (ready 상태 / 딱히 할 말 없음)
    - 교정 메시지    : 가장 중요한 것 1개만 (TTS에서 "/" 이어붙이기 제거)
    - 칭찬 메시지    : 10초 쿨다운으로 가끔만 발화
    """
    if not angles or state != "down":
        return ""

    # ── 우선순위 리스트: (priority, tts문장, hud문장)
    issues = []   # (priority, message)

    if exercise == "squat":
        knee  = angles.get("knee",  180)
        trunk = angles.get("trunk", 90)
        if knee > 120:
            issues.append((0, "무릎을 더 깊이 굽혀주세요"))
        elif knee > 105:
            issues.append((1, "조금만 더 내려가세요"))
        if trunk < 50:
            issues.append((0, "등이 너무 앞으로 숙여졌어요, 상체를 세워주세요"))
        elif trunk < 62:
            issues.append((1, "상체를 조금 더 세워주세요"))

    elif exercise == "pushup":
        elbow = angles.get("elbow", 180)
        body  = angles.get("body",  180)
        if elbow > 120:
            issues.append((0, "팔을 더 굽혀주세요"))
        elif elbow > 100:
            issues.append((1, "조금만 더 내려가세요"))
        if body < 150:
            issues.append((0, "엉덩이가 너무 올라갔어요, 몸을 일직선으로 유지해주세요"))
        elif body < 165:
            issues.append((1, "엉덩이를 살짝 내려주세요"))

    elif exercise == "pullup":
        le = angles.get("L.elbow", 180)
        re = angles.get("R.elbow", 180)
        avg = np.mean([v for v in [le, re] if v < 170] or [180])
        if 100 < avg < 140:
            issues.append((0, "턱이 바에 닿을 때까지 올려주세요"))
        elif 80 < avg <= 100:
            issues.append((1, "조금만 더 올라가세요"))
        if abs(le - re) > 20:
            issues.append((2, "양팔 힘을 균등하게 써주세요"))

    if score["stability"] < 50:
        issues.append((0, "몸이 많이 흔들려요, 균형을 잡아주세요"))
    elif score["stability"] < 70:
        issues.append((2, "좌우 균형을 유지해주세요"))

    if not issues:
        acc = score.get("accuracy") or 0
        if acc >= 88:
            return "자세 완벽해요!"
        elif acc >= 75:
            return "좋아요, 그대로 유지해주세요"
        return ""

    # 가장 우선순위 높은(숫자 낮은) 것 1개만 반환
    issues.sort(key=lambda x: x[0])
    return issues[0][1]


# ── 5. HUD 그리기 ──────────────────────────────────────────────────────────────
def score_color(s):
    if s >= 80: return (80, 200, 80)
    if s >= 55: return (40, 180, 220)
    return (60, 60, 220)

def draw_bar(img, x, y, w, h, value, color, label, measuring=True):
    cv2.rectangle(img, (x,y), (x+w, y+h), (50,50,50), -1)
    if measuring and value is not None:
        fill = int(w * max(0, min(100, value)) / 100)
        cv2.rectangle(img, (x,y), (x+fill, y+h), color, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (100,100,100), 1)
    val_str = f"{value:.0f}" if value is not None else "--"
    return (f"{label}: {val_str}", (x, y-16), 13, (200,200,200))

def draw_skeleton(img, lm_list, img_w, img_h):
    pts = {}
    for name, idx in LM.items():
        lm = lm_list[idx]
        if lm.visibility >= VIS_THRESH:
            color, radius = (0,255,120), 5
        elif lm.visibility >= 0.3:
            color, radius = (0,165,255), 3
        else:
            continue
        px, py = int(lm.x*img_w), int(lm.y*img_h)
        pts[idx] = (px, py)
        cv2.circle(img, (px,py), radius, color, -1)
    for i,j in CONNECTIONS:
        if i in pts and j in pts:
            cv2.line(img, pts[i], pts[j], (180,180,180), 1)

def draw_hud(img, exercise, state, count, score, feedback, angles, fps, cam_guide):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (270,h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    ex_label = {"squat":"SQUAT","pushup":"PUSH-UP","pullup":"PULL-UP"}[exercise]
    cv2.putText(img, ex_label, (12,38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220,180,255), 2)

    # GRU AI 판정 표시
    gru_label = score.get("pose_label", "Good")
    gru_color = (80, 200, 80) if gru_label == "Good" else (60, 60, 220)
    cv2.putText(img, f"AI: {gru_label}", (180, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, gru_color, 2)

    cv2.putText(img, str(count), (12,105),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, score_color(score["total"]), 4)
    cv2.putText(img, "reps", (105,105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)

    korean_texts = []

    st_color = (60,180,255) if state == "down" else (200,200,200)
    st_text  = f"{state.upper()}  {'[측정중]' if score['measuring'] else ''}"
    korean_texts.append((st_text, (12, 120), 16, st_color))

    korean_texts.append(draw_bar(img, 12,160, 240,12, score["accuracy"],  (160,80,240),  "정확도", score["measuring"]))
    korean_texts.append(draw_bar(img, 12,192, 240,12, score["stability"], (80,200,200),  "안정성", True))
    korean_texts.append(draw_bar(img, 12,224, 240,12, score["total"],     score_color(score["total"]), "종합", True))

    # HUD 피드백: 빈 문자열이면 대기 메시지 표시
    hud_fb = feedback if feedback else ("자세 측정 중..." if state == "down" else "준비 자세 유지")
    korean_texts.append((hud_fb, (12, 245), 14, (255,220,100)))

    y0 = 280
    for k,v in list(angles.items())[:3]:
        cv2.putText(img, f"{k}: {v:.0f}deg", (12,y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160,200,160), 1)
        y0 += 18

    cv2.rectangle(img, (0, h-50), (w, h), (20,20,20), -1)
    korean_texts.append((cam_guide, (10, h-40), 15, (255,200,80)))
    
    cv2.putText(img, "[1]Squat [2]Push [3]Pull [r]Reset [q]Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1)
    cv2.putText(img, f"FPS {fps:.0f}", (w-70,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
    if count > 0:
        cv2.putText(img, f"avg {score['total']:.0f}pts / {count}reps",
                    (w-230,48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    score_color(score["total"]), 1)

    img[:] = put_korean_texts(img, korean_texts)


# ── 6. 메인 루프 ───────────────────────────────────────────────────────────────
def main():
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    exercise        = "squat"
    classifier      = PoseClassifier(exercise)
    sm              = StateMachine()
    shoulder_history = []
    prev_time       = time.time()
    fps             = 0.0
    frame_ts        = 0

    tts        = TTSManager()
    prev_count = 0        # 카운트 증가 감지용
    prev_score = 0.0      # 점수 상승 감지용

    print("=" * 55)
    print("운동 자세 인식 v2  —  mediapipe 0.10.x")
    print("1=스쿼트(측면)  2=푸시업(측면)  3=풀업(후면)  r=리셋  q=종료")
    print("=" * 55)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            result   = landmarker.detect_for_video(mp_image, frame_ts)

            angles   = {}
            score    = {"total":0.0,"accuracy":None,"stability":0.0,"pose_label":"Good","measuring":False}
            state    = sm.state
            feedback = ""
            pose_label = "Good"

            if result.pose_landmarks:
                lm_list = result.pose_landmarks[0]
                draw_skeleton(frame, lm_list, img_w, img_h)

                angles = get_angles(lm_list, exercise)
                shoulder_history.append(lm_list[LM["left_shoulder"]].x)
                if len(shoulder_history) > 30:
                    shoulder_history.pop(0)

                # GRU 자세 분류
                feature_vec = get_feature_vector(lm_list)
                pose_label = classifier.predict(feature_vec)

                state, state_changed, count_increased = sm.update(angles, exercise, 0)
                score    = calc_score(angles, exercise, shoulder_history, state, pose_label)
                feedback = get_feedback(angles, exercise, score, state)

                # ── 렙 카운트 완료 → 효과음 + 숫자 TTS ───────────────────────
                if count_increased:
                    tts.speak_count(sm.count)   # "1", "2", "3" ...
                    prev_score = 0.0            # ready 복귀 시 점수 리셋

                # ── 점수 상승 효과음 (down 상태에서만) ───────────────────────
                elif state == "down":
                    cur = score["total"]
                    if cur - prev_score >= 5.0:
                        tts.play_score_effect(cur)
                        prev_score = cur

                # ── 자세 피드백 TTS (빈 문자열이면 speak_feedback이 묵음 처리) ─
                tts.speak_feedback(feedback)

                # 로그 (1초마다)
                now = time.time()
                if now - prev_time >= 1.0:
                    print(f"[{exercise.upper()}] st={state} cnt={sm.count} "
                          f"total={score['total']} acc={score['accuracy']} | "
                          f"{', '.join(f'{k}={v:.0f}' for k,v in angles.items())}")

            # ready로 돌아왔을 때 점수 리셋
            if state == "ready":
                prev_score = 0.0

            now      = time.time()
            fps      = 0.9*fps + 0.1*(1.0/max(now-prev_time, 1e-6))
            prev_time = now

            draw_hud(frame, exercise, state, sm.count, score, feedback,
                     angles, fps, CAM_GUIDE[exercise])
            cv2.imshow("Exercise Pose Tracker v2", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                exercise="squat";  sm.reset(); shoulder_history.clear(); tts.reset_score(); classifier=PoseClassifier(exercise); print(">> 스쿼트")
            elif key == ord("2"):
                exercise="pushup"; sm.reset(); shoulder_history.clear(); tts.reset_score(); classifier=PoseClassifier(exercise); print(">> 푸시업")
            elif key == ord("3"):
                exercise="pullup"; sm.reset(); shoulder_history.clear(); tts.reset_score(); classifier=PoseClassifier(exercise); print(">> 풀업")
            elif key == ord("r"):
                sm.reset(); shoulder_history.clear(); tts.reset_score(); classifier=PoseClassifier(exercise); print(">> 리셋")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== 세션 종료 ===")
    if sm.rep_scores:
        print(f"총 횟수  : {sm.count}")
        print(f"평균 점수: {np.mean(sm.rep_scores):.1f}")
        print(f"최고 점수: {max(sm.rep_scores):.1f}")
        print(f"최저 점수: {min(sm.rep_scores):.1f}")

if __name__ == "__main__":
    main()

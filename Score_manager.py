import numpy as np

IDEAL_ANGLES = {
    "squat":  {"knee": 90,  "trunk": 75},
    "pushup": {"elbow": 90, "body_line": 180},
    "pullup": {"left_elbow": 60, "right_elbow": 60}
}

class ScoreEngine:
    def calculate(self, angles: dict, exercise: str,
                  pose_label: str, landmark_history: list, state: str) -> dict:
        ideal = IDEAL_ANGLES.get(exercise, {})

        if state == "down" and angles:
            angle_errors = []
            for joint, target in ideal.items():
                if joint in angles:
                    err = abs(angles[joint] - target)
                    angle_errors.append(min(err, 90))
            accuracy = 100 - (np.mean(angle_errors) * (100 / 90)) if angle_errors else 50
        else:
            accuracy = None

        # 안정성: 최근 10프레임 어깨 x좌표 표준편차
        stability = 100.0
        if len(landmark_history) >= 10:
            shoulder_xs = [h["left_shoulder_x"] for h in landmark_history[-10:]]
            std = np.std(shoulder_xs)
            stability = max(0, 100 - std * 1000)

        label_penalty = {"Good": 0, "Bad": 30}.get(pose_label, 0)

        if accuracy is not None:
            total = accuracy * 0.6 + stability * 0.4 - label_penalty
        else:
            total = stability - label_penalty

        return {
            "total":      round(max(0, min(100, total)), 1),
            "accuracy":   round(accuracy, 1) if accuracy is not None else None,
            "stability":  round(stability, 1),
            "pose_label": pose_label,
            "measuring":  state == "down"
        }

    def generate_feedback(self, angles: dict, exercise: str, score: dict, state: str) -> str:
        """
        자연스러운 구어체 피드백 반환.
        - ready 상태 → 빈 문자열 (TTS 무음)
        - down 상태 → 구체적인 교정 메시지 우선, 없으면 칭찬
        여러 문제가 있을 때는 가장 중요한 것 하나만 말함 (우선순위 순)
        """
        if not angles:
            return ""          # 감지 안 됨 → TTS 묵음

        if state != "down":
            return ""          # ready 상태에서는 말 안 함

        feedbacks = []         # (우선순위, 메시지) 리스트

        # ── 스쿼트 ───────────────────────────────────────────────────
        if exercise == "squat":
            knee  = angles.get("knee",  180)
            trunk = angles.get("trunk", 90)

            if knee > 120:
                feedbacks.append((0, "무릎을 더 깊이 굽혀주세요"))
            elif knee > 105:
                feedbacks.append((1, "조금만 더 내려가세요"))

            if trunk < 50:
                feedbacks.append((0, "상체가 너무 앞으로 쏠렸어요, 등을 세워주세요"))
            elif trunk < 60:
                feedbacks.append((1, "상체를 조금 더 세워주세요"))

        # ── 푸쉬업 ───────────────────────────────────────────────────
        elif exercise == "pushup":
            elbow     = angles.get("elbow",     180)
            body_line = angles.get("body_line", 180)

            if elbow > 120:
                feedbacks.append((0, "팔을 더 굽혀주세요"))
            elif elbow > 100:
                feedbacks.append((1, "조금만 더 내려가세요"))

            if body_line < 150:
                feedbacks.append((0, "엉덩이가 너무 올라갔어요, 몸을 일직선으로 유지해주세요"))
            elif body_line < 160:
                feedbacks.append((1, "엉덩이를 살짝 내려주세요"))

        # ── 풀업 ─────────────────────────────────────────────────────
        elif exercise == "pullup":
            le = angles.get("left_elbow",  180)
            re = angles.get("right_elbow", 180)
            avg_elbow = np.mean([v for v in [le, re] if v < 170])

            if 100 < avg_elbow < 140:
                feedbacks.append((0, "턱이 바에 닿을 때까지 올려주세요"))
            elif 80 < avg_elbow <= 100:
                feedbacks.append((1, "조금만 더 올려주세요"))

            # 좌우 비대칭 체크
            if abs(le - re) > 20:
                feedbacks.append((1, "양팔 힘을 균등하게 써주세요"))

        # ── 공통: 흔들림 ─────────────────────────────────────────────
        if score["stability"] < 50:
            feedbacks.append((0, "몸이 많이 흔들려요, 균형을 잡아주세요"))
        elif score["stability"] < 70:
            feedbacks.append((2, "좌우 균형을 유지해주세요"))

        if not feedbacks:
            # 교정할 게 없으면 칭찬
            acc = score.get("accuracy") or 0
            if acc >= 90:
                return "자세가 아주 좋아요!"
            elif acc >= 75:
                return "좋아요, 그대로 유지해주세요"
            else:
                return ""   # 점수 낮지만 딱히 뭐라 말할 게 없을 때 → 묵음

        # 우선순위 낮은 숫자(=더 중요한 것) 먼저
        feedbacks.sort(key=lambda x: x[0])
        return feedbacks[0][1]   # 가장 중요한 피드백 하나만 반환

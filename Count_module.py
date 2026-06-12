import numpy as np

THRESHOLDS = {
    "squat": {
        "down_angle": 95,   # 무릎 각도 95° 이하 = down 상태 (강화)
        "up_angle": 155,    # 무릎 각도 155° 이상 = up 복귀
        "key_joint": "knee" # 좌/우 최적 각도로 변경됨
    },
    "pushup": {
        "down_angle": 85,
        "up_angle": 150,
        "key_joint": "elbow"
    },
    "pullup": {
        "down_angle": 150,  # 팔 펴진 상태가 down
        "up_angle": 65,     # 팔 굽힌 상태가 up
        "key_joint": "left_elbow",
        "key_joint2": "right_elbow",
        "inverted": True    # 풀업은 각도 방향 반대
    }
}

class ExerciseStateMachine:
    def __init__(self, exercise: str):
        self.exercise = exercise
        self.state = "ready"   # ready / down
        self.count = 0
        self.rep_scores = []   # 완료된 각 횟수별 점수
        self._down_scores = []

    def update(self, angles: dict, quality_score: float) -> dict:
        cfg = THRESHOLDS[self.exercise]
        prev_state = self.state

        if "key_joint2" in cfg:
            vals = [angles[k] for k in (cfg["key_joint"], cfg["key_joint2"]) if k in angles]
            angle = np.mean(vals) if vals else (0 if cfg.get("inverted") else 180)
        else:
            angle = angles.get(cfg["key_joint"], 0 if cfg.get("inverted") else 180)

        inverted = cfg.get("inverted", False)

        if not inverted:
            if self.state == "ready" and angle < cfg["down_angle"]:
                self.state = "down"
                self._down_scores = []
            elif self.state == "down":
                self._down_scores.append(quality_score)
                if angle > cfg["up_angle"]:
                    self.state = "ready"
                    self.count += 1
                    self.rep_scores.append(max(self._down_scores) if self._down_scores else quality_score)
                    self._down_scores = []
        else:
            if self.state == "ready" and angle > cfg["down_angle"]:
                self.state = "down"
                self._down_scores = []
            elif self.state == "down":
                self._down_scores.append(quality_score)
                if angle < cfg["up_angle"]:
                    self.state = "ready"
                    self.count += 1
                    self.rep_scores.append(max(self._down_scores) if self._down_scores else quality_score)
                    self._down_scores = []

        return {
            "state": self.state,
            "count": self.count,
            "state_changed": prev_state != self.state,
            "measuring": self.state == "down"
        }
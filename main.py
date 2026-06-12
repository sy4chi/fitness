from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json, cv2, numpy as np
import os

from Pose_detect import PoseDetect
from Count_module import ExerciseStateMachine
from Score_manager import ScoreEngine
from Classifier import PoseClassifier
from TTS_Manager import TTSManager

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount models directory to serve ONNX files
if os.path.exists("models"):
    app.mount("/models", StaticFiles(directory="models"), name="models")

# Mount static directory to serve local JS/WASM assets
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the pose landmarker model asset directly to browser
@app.get("/pose_landmarker_full.task")
async def get_pose_task():
    task_path = "pose_landmarker_full.task"
    if os.path.exists(task_path):
        return FileResponse(task_path)
    return HTMLResponse(content="Pose landmarker task file not found.", status_code=404)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
        <div style="font-family: sans-serif; text-align: center; margin-top: 100px;">
            <h1>🏋️‍♂️ AI 운동 자세 분석 서버 구동 중</h1>
            <p>index.html 파일이 아직 생성되지 않았습니다.</p>
        </div>
    """)

@app.websocket("/ws/{exercise}")
async def exercise_ws(websocket: WebSocket, exercise: str):
    await websocket.accept()

    try:
        analyzer      = PoseDetect()
        state_machine = ExerciseStateMachine(exercise)
        score_engine  = ScoreEngine()
        classifier    = PoseClassifier(exercise)
        tts           = TTSManager()
    except ValueError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    landmark_history = []
    prev_count       = 0       # 이전 렙 카운트
    prev_total_score = 0.0     # 이전 총점 (점수 상승 효과음용)

    try:
        while True:
            # 1. React → Base64 프레임 수신
            data  = await websocket.receive_text()
            msg   = json.loads(data)
            frame = analyzer.decode_frame(msg["frame"])

            # 2. 랜드마크 추출
            lm = analyzer.extract_landmarks(frame)
            if lm is None:
                await websocket.send_text(json.dumps({"error": "no_pose"}))
                continue

            # 3. 관절 각도 계산
            angles = analyzer.get_exercise_angles(lm, exercise)

            # 4. AI 분류
            feature_vec = analyzer.get_feature_vector(lm)
            pose_label  = classifier.predict(feature_vec)

            # 5. 히스토리 저장 (안정성 계산용)
            landmark_history.append({
                "left_shoulder_x":  lm[11].x,
                "right_shoulder_x": lm[12].x
            })
            if len(landmark_history) > 30:
                landmark_history.pop(0)

            # 6. 상태 머신 업데이트
            state_data    = state_machine.update(angles, 0)
            current_count = state_data["count"]

            # 7. 점수 계산
            score_data = score_engine.calculate(
                angles, exercise, pose_label, landmark_history, state_data["state"]
            )
            current_total = score_data["total"]

            # ── 렙 카운트 증가 → 효과음 + 숫자 발화 ──────────────────────
            if current_count > prev_count:
                tts.speak_count(current_count)
                prev_count = current_count

            # ── 점수 상승 → 효과음 (down 상태에서만) ──────────────────────
            if state_data["state"] == "down":
                tts.play_score_effect(current_total)

            # 8. 자세 피드백 생성 및 TTS 발화
            feedback = score_engine.generate_feedback(
                angles, exercise, score_data, state_data["state"]
            )
            tts.speak_feedback(feedback)

            # 점수 추적 업데이트 (ready로 돌아오면 리셋)
            if state_data["state"] == "ready":
                prev_total_score = 0.0
                tts.last_score   = 0.0
            else:
                prev_total_score = current_total

            # 9. 결과 전송
            await websocket.send_text(json.dumps({
                "angles":     angles,
                "state":      state_data,
                "score":      score_data,
                "feedback":   feedback,
                "pose_label": pose_label,
                "landmarks":  [[lm[i].x, lm[i].y] for i in range(33)]
            }))

    except WebSocketDisconnect:
        pass

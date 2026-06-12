"""
TTS_Manager.py
==============
macOS `say` 명령어를 사용하여 자연스러운 한국어(Yuna 음성) TTS를 제공합니다.
pyttsx3를 제거하고, subprocess 기반으로 음성 재생 및 효과음을 처리합니다.

우선순위 큐 구조:
    priority 0 → 카운트 숫자 (최우선, 이전 큐 비움)
    priority 1 → 자세 교정/칭찬 피드백
"""
import threading
import time
import queue
import subprocess
import platform

# macOS 사용 가능한 한국어 음성 우선순위 (say -v '?' 로 확인)
KO_VOICE_PRIORITY = ["Yuna", "Flo", "Shelley", "Sandy", "Reed", "Eddy", "Grandma", "Rocko (Korean (South Korea))"]

def _get_best_korean_voice() -> str:
    """설치된 음성 중 우선순위가 가장 높은 한국어 음성 이름을 반환합니다."""
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5)
        installed = result.stdout + result.stderr
        for voice in KO_VOICE_PRIORITY:
            if voice in installed:
                return voice
    except Exception:
        pass
    return "Yuna"  # 기본값


class TTSManager:
    def __init__(self, feedback_cooldown: float = 4.0):
        self.feedback_cooldown  = feedback_cooldown
        self.last_feedback_time = 0.0
        self.last_score         = 0.0
        self.is_mac             = platform.system() == "Darwin"

        # 최적 한국어 음성 선택
        self.voice = _get_best_korean_voice() if self.is_mac else None
        print(f"[TTS] 사용 음성: {self.voice}")

        # PriorityQueue: (priority, text) — 숫자 낮을수록 먼저 재생
        self.q      = queue.PriorityQueue()
        self._lock  = threading.Lock()

        self.worker = threading.Thread(target=self._tts_loop, daemon=True)
        self.worker.start()

    # ── 워커 루프 ─────────────────────────────────────────────────────────────
    def _tts_loop(self):
        while True:
            try:
                _, text = self.q.get()
                if text is None:
                    break
                self._say(text)
            except Exception as e:
                print(f"[TTS] 재생 오류: {e}")
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass

    def _say(self, text: str):
        """macOS say 명령어로 동기 음성 재생 (워커 스레드 내부에서만 호출)."""
        if not self.is_mac or not text:
            return
        try:
            subprocess.run(
                ["say", "-v", self.voice, "-r", "175", text],
                timeout=15,
                check=False
            )
        except subprocess.TimeoutExpired:
            print(f"[TTS] 타임아웃: {text}")
        except Exception as e:
            print(f"[TTS] say 실행 오류: {e}")

    # ── 효과음 ───────────────────────────────────────────────────────────────
    def _beep(self, sound: str):
        """
        macOS 시스템 사운드 재생 (비동기).
        sound: Tink / Glass / Pop / Funk / Hero / Ping 등
        """
        if not self.is_mac:
            return
        sound_path = f"/System/Library/Sounds/{sound}.aiff"
        try:
            subprocess.Popen(
                ["afplay", sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[TTS] 효과음 오류: {e}")

    # ── 렙 카운트 TTS + 효과음 ────────────────────────────────────────────────
    def speak_count(self, count: int):
        """카운트 완료 시 호출. 쿨다운 없이 즉시 발화."""
        self._beep("Tink")           # 효과음 먼저 (비동기)
        self._clear_queue()
        self.q.put((0, str(count)))  # 숫자 발화 (최우선)

    # ── 점수 상승 효과음 ──────────────────────────────────────────────────────
    def play_score_effect(self, score: float):
        """
        점수가 5점 이상 오를 때만 효과음.
        구간별로 다른 사운드.
        """
        if score - self.last_score >= 5.0:
            if score >= 85:
                self._beep("Glass")   # 고득점: 맑은 소리
            elif score >= 60:
                self._beep("Pop")     # 중간
            else:
                self._beep("Tink")
            self.last_score = score

    def reset_score(self):
        """운동 전환/리셋 시 점수 기준 초기화"""
        self.last_score = 0.0

    # ── 자세 피드백 TTS (쿨다운 적용) ────────────────────────────────────────
    def speak_feedback(self, text: str):
        """
        - 빈 문자열 → 묵음
        - 칭찬("완벽", "좋아요", "아주 좋아요") → 10초 쿨다운
        - 교정 피드백 → feedback_cooldown(기본 4초) 쿨다운 + 경고음(Funk)
        """
        if not text:
            return

        now     = time.time()
        is_good = any(kw in text for kw in ("완벽", "좋아요", "아주 좋아요", "유지"))
        limit   = 10.0 if is_good else self.feedback_cooldown

        if now - self.last_feedback_time < limit:
            return

        self.last_feedback_time = now
        self._clear_queue()

        if not is_good:
            # 잘못된 자세 → 경고음 먼저 울리고 TTS 발화
            self._beep("Funk")

        self.q.put((1, text))   # priority=1 (카운트보다 낮음)

    # ── 하위 호환 래퍼 ───────────────────────────────────────────────────────
    def speak(self, text: str, force: bool = False):
        if force:
            self._clear_queue()
            self.q.put((0, text))
        else:
            self.speak_feedback(text)

    # ── 큐 비우기 ─────────────────────────────────────────────────────────────
    def _clear_queue(self):
        with self._lock:
            try:
                while True:
                    self.q.get_nowait()
                    self.q.task_done()
            except queue.Empty:
                pass

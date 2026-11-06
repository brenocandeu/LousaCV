import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"


class HandTracker:
    """Rastreador de mao baseado no MediaPipe Tasks (otimizado para CPU)."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.6,
        presence_conf: float = 0.6,
        tracking_conf: float = 0.6,
    ) -> None:
        self._ensure_model()

        model_bytes = MODEL_PATH.read_bytes()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_result = None
        self._last_timestamp_ms = 0

    @staticmethod
    def _ensure_model() -> None:
        """Baixa o modelo oficial do MediaPipe caso nao esteja na pasta do projeto."""
        if not MODEL_PATH.exists():
            print("Baixando modelo hand_landmarker.task pela primeira vez...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("Modelo salvo com sucesso!")

    def process(self, frame_bgr: np.ndarray) -> None:
        """Executa a deteccao dos 21 pontos da mao no frame atual."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        now_ms = int(time.perf_counter() * 1000)
        timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        self.last_result = self.detector.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Desenha as conexoes e os 21 pontos (landmarks) da mao no frame."""
        if not self.last_result or not self.last_result.hand_landmarks:
            return frame_bgr

        h, w, _ = frame_bgr.shape
        for hand_landmarks in self.last_result.hand_landmarks:
            vision.drawing_utils.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                vision.drawing_styles.get_default_hand_landmarks_style(),
                vision.drawing_styles.get_default_hand_connections_style(),
            )

            # Exibe os indices (0 a 20) ao lado de cada ponto para facilitar validacao
            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(
                    frame_bgr,
                    str(idx),
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame_bgr

    def get_landmarks_px(self, frame_shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """Retorna lista [(id, x, y), ...] em pixels da primeira mao detectada."""
        if not self.last_result or not self.last_result.hand_landmarks:
            return []

        h, w, _ = frame_shape
        hand = self.last_result.hand_landmarks[0]
        return [(idx, int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(hand)]

    def close(self) -> None:
        self.detector.close()


def main() -> None:
    window_name = "Lousa Magica - Rastreamento de Mao"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Erro: Nao foi possivel acessar a webcam (indice 0).")
        return

    tracker = HandTracker(max_hands=1)
    prev_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Aviso: Falha ao capturar frame da webcam.")
                break

            # Garante resolucao HD 1280x720 caso a webcam nao aceite nativamente via driver
            if frame.shape[1] != 1280 or frame.shape[0] != 720:
                frame = cv2.resize(frame, (1280, 720))

            # Espelhamento horizontal para sensacao de espelho natural
            frame = cv2.flip(frame, 1)

            # Detecta e desenha os pontos da mao
            tracker.process(frame)
            tracker.draw_landmarks(frame)

            # Calculo simples de FPS
            curr_time = time.perf_counter()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)} | Pressione 'Q' ou 'ESC' para sair",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


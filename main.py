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

# Configuracoes da Lousa
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BRUSH_COLOR = (0, 0, 255)  # Vermelho vibrante (BGR), igual ao projeto de referencia
BRUSH_THICKNESS = 12
ERASER_RADIUS = 45
SMOOTHING_ALPHA = 0.5  # Suavizacao exponencial (0.1 = muito suave, 1.0 = sem filtro)


class HandTracker:
    """Rastreador de mao leve e nativo usando MediaPipe Tasks (sem cvzone)."""

    # Pares (ponta_do_dedo, articulacao_intermediaria_PIP) dos 4 dedos verticais
    FINGER_TIPS_PIPS = (
        (8, 6),    # Indicador
        (12, 10),  # Medio
        (16, 14),  # Anelar
        (20, 18),  # Mindinho
    )

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
        if not MODEL_PATH.exists():
            print("Baixando modelo hand_landmarker.task pela primeira vez...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("Modelo salvo com sucesso!")

    def process(self, frame_bgr: np.ndarray) -> bool:
        """Detecta a mao no frame atual e retorna True se encontrar."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        now_ms = int(time.perf_counter() * 1000)
        timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        self.last_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return bool(self.last_result and self.last_result.hand_landmarks)

    def get_fingers_up(self) -> list[bool]:
        """Retorna [indicador, medio, anelar, mindinho] indicando quais estao em pe."""
        if not self.last_result or not self.last_result.hand_landmarks:
            return [False, False, False, False]

        lm = self.last_result.hand_landmarks[0]
        # No OpenCV/MediaPipe o eixo Y cresce para baixo: ponta acima da articulacao => y menor
        return [lm[tip].y < lm[pip].y for tip, pip in self.FINGER_TIPS_PIPS]

    def get_point_px(self, landmark_id: int, width: int, height: int) -> tuple[int, int]:
        """Retorna a coordenada (x, y) em pixels de um landmark especifico."""
        lm = self.last_result.hand_landmarks[0][landmark_id]
        return int(lm.x * width), int(lm.y * height)

    def draw_landmarks(self, frame_bgr: np.ndarray) -> None:
        """Desenha as conexoes da mao no frame."""
        if not self.last_result or not self.last_result.hand_landmarks:
            return

        for hand_landmarks in self.last_result.hand_landmarks:
            vision.drawing_utils.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                vision.drawing_styles.get_default_hand_landmarks_style(),
                vision.drawing_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self.detector.close()


def merge_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Funde o canvas de desenho sobre o frame da webcam usando mascaras binarias O(1)."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    mask_fg = cv2.bitwise_not(mask_inv)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask_fg)
    return cv2.bitwise_or(frame_bg, canvas_fg)


def main() -> None:
    window_name = "Lousa Magica"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Erro: Nao foi possivel acessar a webcam (indice 0).")
        return

    tracker = HandTracker(max_hands=1)

    # Canvas permanente em memoria: evita o loop O(N) lento de redesenhar pontos a cada frame
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    prev_draw_pt: tuple[int, int] | None = None
    prev_erase_pt: tuple[int, int] | None = None
    smooth_pt: tuple[float, float] | None = None

    prev_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Aviso: Falha ao capturar frame da webcam.")
                break

            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Espelha antes de processar para manter coordenadas e textos na orientacao correta
            frame = cv2.flip(frame, 1)

            hand_detected = tracker.process(frame)
            mode_label = "AGUARDANDO MAO"
            overlay_cursor: tuple[str, tuple[int, int]] | None = None

            if hand_detected:
                ind_up, med_up, ane_up, min_up = tracker.get_fingers_up()

                # Coordenada suavizada da ponta do indicador (landmark 8)
                raw_x, raw_y = tracker.get_point_px(8, FRAME_WIDTH, FRAME_HEIGHT)
                if smooth_pt is None:
                    smooth_pt = (float(raw_x), float(raw_y))
                else:
                    smooth_pt = (
                        SMOOTHING_ALPHA * raw_x + (1.0 - SMOOTHING_ALPHA) * smooth_pt[0],
                        SMOOTHING_ALPHA * raw_y + (1.0 - SMOOTHING_ALPHA) * smooth_pt[1],
                    )
                ix, iy = int(smooth_pt[0]), int(smooth_pt[1])

                # 1) MODO DESENHO: Apenas o indicador levantado
                if ind_up and not med_up and not ane_up and not min_up:
                    mode_label = "DESENHO (1 Dedo)"
                    prev_erase_pt = None

                    if prev_draw_pt is None:
                        cv2.circle(canvas, (ix, iy), BRUSH_THICKNESS // 2, BRUSH_COLOR, cv2.FILLED)
                    else:
                        cv2.line(canvas, prev_draw_pt, (ix, iy), BRUSH_COLOR, BRUSH_THICKNESS)

                    prev_draw_pt = (ix, iy)
                    overlay_cursor = ("DRAW", (ix, iy))

                # 2) MODO LIVRE: Indicador e dedo medio em pe
                elif ind_up and med_up and not ane_up and not min_up:
                    mode_label = "LIVRE (2 Dedos)"
                    prev_draw_pt = None
                    prev_erase_pt = None

                    mx, my = tracker.get_point_px(12, FRAME_WIDTH, FRAME_HEIGHT)
                    cx, cy = (ix + mx) // 2, (iy + my) // 2
                    overlay_cursor = ("FREE", (cx, cy))

                # 3) MODO BORRACHA: 4 dedos levantados (indicador, medio, anelar e mindinho)
                elif ind_up and med_up and ane_up and min_up:
                    mode_label = "BORRACHA (4 Dedos)"
                    prev_draw_pt = None

                    # Centro da borracha posicionado no meio das pontas dos 4 dedos
                    pts_4 = [tracker.get_point_px(idx, FRAME_WIDTH, FRAME_HEIGHT) for idx in (8, 12, 16, 20)]
                    ex = sum(p[0] for p in pts_4) // 4
                    ey = sum(p[1] for p in pts_4) // 4

                    cv2.circle(canvas, (ex, ey), ERASER_RADIUS, (0, 0, 0), cv2.FILLED)
                    if prev_erase_pt is not None:
                        cv2.line(canvas, prev_erase_pt, (ex, ey), (0, 0, 0), ERASER_RADIUS * 2)

                    prev_erase_pt = (ex, ey)
                    overlay_cursor = ("ERASE", (ex, ey))

                # Outras poses (ex: mao fechada) -> nao risca nem apaga
                else:
                    mode_label = "PAUSADO"
                    prev_draw_pt = None
                    prev_erase_pt = None
            else:
                prev_draw_pt = None
                prev_erase_pt = None
                smooth_pt = None

            # Funde o canvas permanente no frame (mantem o desenho mesmo sem a mao na tela!)
            frame = merge_canvas(frame, canvas)

            # Desenha os pontos da mao e os cursores visuais por cima da imagem combinada
            tracker.draw_landmarks(frame)

            if overlay_cursor is not None:
                kind, (cx, cy) = overlay_cursor
                if kind == "DRAW":
                    cv2.circle(frame, (cx, cy), 10, BRUSH_COLOR, cv2.FILLED)
                elif kind == "FREE":
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 0), 2)
                elif kind == "ERASE":
                    cv2.circle(frame, (cx, cy), ERASER_RADIUS, (255, 255, 255), 2)

            curr_time = time.perf_counter()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)} | Modo: {mode_label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
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

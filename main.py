import math
import threading
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

# Configuracoes Visuais
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BRUSH_COLOR = (0, 0, 255)  # Vermelho vibrante (BGR)
BRUSH_THICKNESS = 12
ERASER_RADIUS = 45
SMOOTHING_ALPHA = 0.55  # Equilibrio entre fluidez e resposta rapida
DRAW_GRACE_SEC = 0.20   # Tolerancia (200ms) contra micro-falhas de iluminacao no desenho


class CameraStream:
    """Captura frames da webcam em thread dedicada para manter o FPS alto em qualquer hardware."""

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)

        # Forca codec MJPG (destrava 30/60 FPS em webcams USB HD que travam em 10-15 FPS no modo YUY2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.opened = self.cap.isOpened()
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._stopped = False

        if self.opened:
            ok, first = self.cap.read()
            if ok:
                self._frame = self._prepare(first)
                self._frame_id = 1
                self._new_frame_event.set()
            threading.Thread(target=self._reader_loop, daemon=True).start()

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return cv2.flip(frame, 1)

    def _reader_loop(self) -> None:
        while not self._stopped:
            ok, raw = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            prepared = self._prepare(raw)
            with self._lock:
                self._frame = prepared
                self._frame_id += 1
            self._new_frame_event.set()

    def read_next(self, timeout: float = 0.05) -> tuple[bool, np.ndarray | None, int]:
        """Aguarda o proximo frame real da camera sem desperdicar CPU (libera nucleos para a IA)."""
        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        with self._lock:
            if self._frame is None:
                return False, None, 0
            return True, self._frame.copy(), self._frame_id

    def release(self) -> None:
        self._stopped = True
        self._new_frame_event.set()
        time.sleep(0.02)
        self.cap.release()


class HandTracker:
    """
    Rastreador 3D assincrono (LIVE_STREAM) com realce de iluminacao (Gamma + CLAHE)
    e reconhecimento ergonomico invariante ao angulo da mao (nao exige apontar para cima).
    """

    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )

    def __init__(self, max_hands: int = 1) -> None:
        self._ensure_model()
        model_bytes = MODEL_PATH.read_bytes()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.35,
            result_callback=self._on_result,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._dark_gamma_lut = np.array(
            [((i / 255.0) ** 0.65) * 255 for i in range(256)], dtype=np.uint8
        )

        self._lock = threading.Lock()
        self._landmarks_2d = None
        self._landmarks_3d = None
        self._result_seq = 0
        self._busy = False
        self._last_ts_ms = 0
        self._infer_start_t = 0.0
        self._avg_infer_ms = 15.0
        self._infer_size = (640, 360)

    @staticmethod
    def _ensure_model() -> None:
        if not MODEL_PATH.exists():
            print("Baixando modelo hand_landmarker.task pela primeira vez...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("Modelo salvo com sucesso!")

    def _on_result(self, result, _output_image, _timestamp_ms: int) -> None:
        dt_ms = (time.perf_counter() - self._infer_start_t) * 1000.0
        self._avg_infer_ms = 0.85 * self._avg_infer_ms + 0.15 * dt_ms

        if self._avg_infer_ms > 32.0:
            self._infer_size = (448, 252)
        elif self._avg_infer_ms < 18.0:
            self._infer_size = (640, 360)

        lms_2d = result.hand_landmarks[0] if (result and result.hand_landmarks) else None
        lms_3d = result.hand_world_landmarks[0] if (result and result.hand_world_landmarks) else None

        with self._lock:
            self._landmarks_2d = lms_2d
            self._landmarks_3d = lms_3d
            self._result_seq += 1
            self._busy = False

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True

        small = cv2.resize(frame_bgr, self._infer_size, interpolation=cv2.INTER_LINEAR)

        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y_chan = ycrcb[:, :, 0]
        if float(np.mean(y_chan)) < 90.0:
            y_chan = cv2.LUT(y_chan, self._dark_gamma_lut)
        ycrcb[:, :, 0] = self._clahe.apply(y_chan)
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        now_ms = int(time.perf_counter() * 1000)
        ts_ms = max(now_ms, self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        self._infer_start_t = time.perf_counter()

        try:
            self.detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        except Exception:
            with self._lock:
                self._busy = False

    def get_snapshot(self):
        """Retorna (landmarks_2d, landmarks_3d, sequencia) de forma thread-safe."""
        with self._lock:
            return self._landmarks_2d, self._landmarks_3d, self._result_seq

    @staticmethod
    def _dist3(w, a: int, b: int) -> float:
        return math.sqrt(
            (w[a].x - w[b].x) ** 2 + (w[a].y - w[b].y) ** 2 + (w[a].z - w[b].z) ** 2
        )

    @staticmethod
    def _finger_cos3(w, mcp: int, pip: int, tip: int) -> float:
        """Cosseno do angulo 3D entre a base (MCP->PIP) e a ponta (PIP->TIP) do dedo."""
        v1x, v1y, v1z = w[pip].x - w[mcp].x, w[pip].y - w[mcp].y, w[pip].z - w[mcp].z
        v2x, v2y, v2z = w[tip].x - w[pip].x, w[tip].y - w[pip].y, w[tip].z - w[pip].z
        n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
        if n1 < 1e-6 or n2 < 1e-6:
            return -1.0
        return (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)

    @classmethod
    def _is_extended_3d(
        cls, w, mcp: int, pip: int, tip: int, cos_min: float = 0.50, ratio_min: float = 1.10
    ) -> bool:
        """
        Verifica se o dedo esta esticado usando coordenadas 3D reais da mao.
        Funciona em QUALQUER angulo: deitado de lado, apontando para a tela ou para cima!
        """
        return (
            cls._finger_cos3(w, mcp, pip, tip) > cos_min
            and cls._dist3(w, 0, tip) > cls._dist3(w, 0, pip) * ratio_min
        )

    @classmethod
    def classify_gesture(
        cls, lms2d, w3d, width: int, height: int
    ) -> tuple[str, tuple[float, float]]:
        """
        Classifica o gesto ergonomico da mao e retorna (modo, ponto_referencia_xy):
        - 'DRAW':
            1) Indicador estendido em qualquer angulo 3D (deitado na horizontal, apontando
               reto para o monitor com o cotovelo apoiado na mesa, ou na diagonal); OU
            2) Gesto de 'Segurar Caneta' (pinca: polegar encostado na ponta do indicador),
               permitindo escrever com a mao 100% relaxada sem forcar os outros dedos!
        - 'FREE': Indicador + Medio estendidos (em qualquer angulo).
        - 'ERASE': 4 dedos estendidos (Indicador + Medio + Anelar + Mindinho), como antes.
        - 'PAUSED': Mao fechada ou em repouso.
        """
        # Avalia extensao 3D dos 4 dedos independente da rotacao do pulso
        ind_ext = cls._is_extended_3d(w3d, 5, 6, 8, cos_min=0.35, ratio_min=1.05)
        med_ext = cls._is_extended_3d(w3d, 9, 10, 12, cos_min=0.55, ratio_min=1.10)
        ane_ext = cls._is_extended_3d(w3d, 13, 14, 16, cos_min=0.55, ratio_min=1.10)
        min_ext = cls._is_extended_3d(w3d, 17, 18, 20, cos_min=0.50, ratio_min=1.08)

        # Detecta pega ergonomica de caneta/giz (pinca entre polegar 4 e indicador 8)
        palm_2d = max(math.hypot(lms2d[9].x - lms2d[0].x, lms2d[9].y - lms2d[0].y), 1e-6)
        pinch_2d = math.hypot(lms2d[4].x - lms2d[8].x, lms2d[4].y - lms2d[8].y) < 0.35 * palm_2d
        pinch_3d = cls._dist3(w3d, 4, 8) < 0.035
        not_fist = (
            cls._dist3(w3d, 0, 8) > cls._dist3(w3d, 0, 5) * 1.18
            and cls._finger_cos3(w3d, 5, 6, 8) > 0.15
        )
        pen_pinch = pinch_2d and pinch_3d and not_fist and not (med_ext and ane_ext and min_ext)

        # 1) BORRACHA: 4 dedos levantados/estendidos (indicador, medio, anelar e mindinho)
        if ind_ext and med_ext and ane_ext and min_ext and not pen_pinch:
            ex = sum(lms2d[idx].x for idx in (8, 12, 16, 20)) * 0.25 * width
            ey = sum(lms2d[idx].y for idx in (8, 12, 16, 20)) * 0.25 * height
            return "ERASE", (ex, ey)

        # 2) MODO LIVRE: 2 dedos estendidos (indicador + medio) sem pinca
        if ind_ext and med_ext and not ane_ext and not min_ext and not pen_pinch:
            cx = (lms2d[8].x + lms2d[12].x) * 0.5 * width
            cy = (lms2d[8].y + lms2d[12].y) * 0.5 * height
            return "FREE", (cx, cy)

        # 3) DESENHO ERGONOMICO: Pega de caneta (pinca) OU indicador apontado em qualquer angulo 3D
        if pen_pinch:
            px = (lms2d[8].x + lms2d[4].x) * 0.5 * width
            py = (lms2d[8].y + lms2d[4].y) * 0.5 * height
            return "DRAW_PEN", (px, py)

        if ind_ext and not med_ext and not ane_ext and not min_ext:
            ix = lms2d[8].x * width
            iy = lms2d[8].y * height
            return "DRAW_INDEX", (ix, iy)

        return "PAUSED", (lms2d[8].x * width, lms2d[8].y * height)

    @classmethod
    def draw_landmarks_fast(cls, frame: np.ndarray, lms) -> None:
        if lms is None:
            return
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        for a, b in cls.HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (220, 220, 220), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 120), cv2.FILLED)

    def close(self) -> None:
        self.detector.close()


def main() -> None:
    window_name = "Lousa Magica"

    cam = CameraStream(src=0, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    if not cam.opened:
        print("Erro: Nao foi possivel acessar a webcam (indice 0).")
        return

    tracker = HandTracker(max_hands=1)

    # Canvas colorido + mascara binaria 2D sincronizada (fusao em 0.03 ms via cv2.copyTo!)
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    canvas_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    prev_draw_pt: tuple[int, int] | None = None
    prev_erase_pt: tuple[int, int] | None = None
    smooth_pt: tuple[float, float] | None = None

    last_submitted_frame_id = -1
    last_processed_seq = -1
    last_draw_seen_t = 0.0
    non_draw_streak = 0

    mode_label = "AGUARDANDO MAO"
    overlay_cursor: tuple[str, tuple[int, int], int] | None = None

    fps_smooth = 30.0
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame, frame_id = cam.read_next()
            if not ok or frame is None:
                continue

            if frame_id != last_submitted_frame_id:
                tracker.submit_frame(frame)
                last_submitted_frame_id = frame_id

            lms2d, w3d, seq = tracker.get_snapshot()
            now = time.perf_counter()

            if seq != last_processed_seq:
                last_processed_seq = seq

                if lms2d is not None and w3d is not None:
                    gesture, (raw_x, raw_y) = tracker.classify_gesture(
                        lms2d, w3d, FRAME_WIDTH, FRAME_HEIGHT
                    )

                    if smooth_pt is None:
                        smooth_pt = (raw_x, raw_y)
                    else:
                        smooth_pt = (
                            SMOOTHING_ALPHA * raw_x + (1.0 - SMOOTHING_ALPHA) * smooth_pt[0],
                            SMOOTHING_ALPHA * raw_y + (1.0 - SMOOTHING_ALPHA) * smooth_pt[1],
                        )
                    sx, sy = int(smooth_pt[0]), int(smooth_pt[1])

                    # 1) MODO DESENHO: Indicador em qualquer angulo 3D OU Pega de Caneta (Pinca)
                    if gesture in ("DRAW_INDEX", "DRAW_PEN"):
                        mode_label = (
                            "DESENHO (Caneta/Pinca)"
                            if gesture == "DRAW_PEN"
                            else "DESENHO (Indicador Livre 3D)"
                        )
                        non_draw_streak = 0
                        last_draw_seen_t = now
                        prev_erase_pt = None

                        if prev_draw_pt is not None:
                            if math.hypot(sx - prev_draw_pt[0], sy - prev_draw_pt[1]) > 180:
                                prev_draw_pt = None

                        if prev_draw_pt is None:
                            cv2.circle(canvas, (sx, sy), BRUSH_THICKNESS // 2, BRUSH_COLOR, cv2.FILLED)
                            cv2.circle(canvas_mask, (sx, sy), BRUSH_THICKNESS // 2, 255, cv2.FILLED)
                        else:
                            cv2.line(canvas, prev_draw_pt, (sx, sy), BRUSH_COLOR, BRUSH_THICKNESS)
                            cv2.line(canvas_mask, prev_draw_pt, (sx, sy), 255, BRUSH_THICKNESS)

                        prev_draw_pt = (sx, sy)
                        overlay_cursor = ("DRAW", (sx, sy), 10)

                    # 2) MODO BORRACHA: 4 dedos estendidos (como estava antes)
                    elif gesture == "ERASE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            mode_label = "BORRACHA (4 Dedos)"
                            prev_draw_pt = None
                            ex, ey = int(raw_x), int(raw_y)

                            cv2.circle(canvas, (ex, ey), ERASER_RADIUS, (0, 0, 0), cv2.FILLED)
                            cv2.circle(canvas_mask, (ex, ey), ERASER_RADIUS, 0, cv2.FILLED)

                            if prev_erase_pt is not None:
                                cv2.line(canvas, prev_erase_pt, (ex, ey), (0, 0, 0), ERASER_RADIUS * 2)
                                cv2.line(canvas_mask, prev_erase_pt, (ex, ey), 0, ERASER_RADIUS * 2)

                            prev_erase_pt = (ex, ey)
                            overlay_cursor = ("ERASE", (ex, ey), ERASER_RADIUS)

                    # 3) MODO LIVRE: 2 dedos estendidos (indicador + medio)
                    elif gesture == "FREE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            mode_label = "LIVRE (2 Dedos)"
                            prev_draw_pt = None
                            prev_erase_pt = None
                            cx, cy = int(raw_x), int(raw_y)
                            overlay_cursor = ("FREE", (cx, cy), 10)

                    # Mao fechada / soltou a pinca -> pausa o traco
                    else:
                        prev_erase_pt = None
                        non_draw_streak += 1
                        if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC and non_draw_streak < 3:
                            overlay_cursor = ("DRAW", prev_draw_pt, 10)
                        else:
                            mode_label = "LIVRE / PAUSADO"
                            prev_draw_pt = None
                            overlay_cursor = ("FREE", (int(raw_x), int(raw_y)), 8)
                else:
                    prev_erase_pt = None
                    if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                        overlay_cursor = ("DRAW", prev_draw_pt, 10)
                    else:
                        mode_label = "AGUARDANDO MAO"
                        prev_draw_pt = None
                        smooth_pt = None
                        overlay_cursor = None

            # Fusao ultra-rapida O(1) in-place (0.03 ms/frame)
            cv2.copyTo(canvas, canvas_mask, frame)

            tracker.draw_landmarks_fast(frame, lms2d)

            if overlay_cursor is not None:
                kind, (cx, cy), rad = overlay_cursor
                if kind == "DRAW":
                    cv2.circle(frame, (cx, cy), rad, BRUSH_COLOR, cv2.FILLED)
                elif kind == "FREE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 0), 2)
                elif kind == "ERASE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2)

            curr_time = time.perf_counter()
            inst_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            fps_smooth = 0.90 * fps_smooth + 0.10 * inst_fps

            cv2.putText(
                frame,
                f"FPS: {int(fps_smooth)} | Modo: {mode_label}",
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
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

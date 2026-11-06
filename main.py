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
SMOOTHING_ALPHA = 0.55  # Equilibrio entre fluidez e resposta rapida
DRAW_GRACE_SEC = 0.20   # Tolerancia (200ms) contra falhas rapidas de iluminacao no desenho


class CameraStream:
    """Captura frames da webcam em thread dedicada para nao travar o FPS da interface."""

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
    Rastreador assincrono (LIVE_STREAM) com realce automatico de iluminacao (Gamma + CLAHE)
    e escala adaptativa para rodar liso tanto em PCs potentes quanto em notebooks basicos.
    """

    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )
    PALM_IDS = (0, 5, 9, 13, 17)

    def __init__(self, max_hands: int = 1) -> None:
        self._ensure_model()
        model_bytes = MODEL_PATH.read_bytes()

        # Limiares mais tolerantes para nao perder o indicador em sombras ou movimento rapido
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

        # Realce fotometrico leve para salas escuras (ex: sala com projetor ligado) ou contra-luz
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._dark_gamma_lut = np.array(
            [((i / 255.0) ** 0.65) * 255 for i in range(256)], dtype=np.uint8
        )

        self._lock = threading.Lock()
        self._landmarks = None
        self._result_seq = 0
        self._busy = False
        self._last_ts_ms = 0
        self._infer_start_t = 0.0
        self._avg_infer_ms = 15.0
        self._infer_size = (640, 360)  # Ajusta automaticamente se o hardware for mais fraco

    @staticmethod
    def _ensure_model() -> None:
        if not MODEL_PATH.exists():
            print("Baixando modelo hand_landmarker.task pela primeira vez...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("Modelo salvo com sucesso!")

    def _on_result(self, result, _output_image, _timestamp_ms: int) -> None:
        dt_ms = (time.perf_counter() - self._infer_start_t) * 1000.0
        self._avg_infer_ms = 0.85 * self._avg_infer_ms + 0.15 * dt_ms

        # Escala adaptativa de hardware: reduz resolucao interna em CPUs mais lentas
        if self._avg_infer_ms > 32.0:
            self._infer_size = (448, 252)
        elif self._avg_infer_ms < 18.0:
            self._infer_size = (640, 360)

        lms = result.hand_landmarks[0] if (result and result.hand_landmarks) else None
        with self._lock:
            self._landmarks = lms
            self._result_seq += 1
            self._busy = False

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        """Envia o frame para inferencia em background sem bloquear a renderizacao."""
        with self._lock:
            if self._busy:
                return
            self._busy = True

        # Redimensiona copia pequena para a IA (desenho na tela continua em HD 1280x720)
        small = cv2.resize(frame_bgr, self._infer_size, interpolation=cv2.INTER_LINEAR)

        # Normaliza iluminacao no canal Y (corrige ambiente escuro e melhora contraste dos dedos)
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
        """Retorna (landmarks_atuais, sequencia_resultado) de forma thread-safe."""
        with self._lock:
            return self._landmarks, self._result_seq

    @staticmethod
    def _dist(lms, a: int, b: int) -> float:
        return math.hypot(lms[a].x - lms[b].x, lms[a].y - lms[b].y)

    @classmethod
    def get_fingers_up(cls, lms) -> list[bool]:
        """
        Retorna [polegar, indicador, medio, anelar, mindinho].
        Combina posicao vertical com distancia ao punho para funcionar mesmo com a mao inclinada.
        """
        palm = max(cls._dist(lms, 0, 9), 1e-6)

        # Polegar aberto: ponta (4) afastada da base do mindinho (17) e da base do indicador (5)
        thumb_up = (
            cls._dist(lms, 4, 17) > cls._dist(lms, 3, 17) * 1.04
            and cls._dist(lms, 4, 5) > 0.32 * palm
        )

        # Indicador mais tolerante a inclinacao lateral ao escrever e leve perspectiva frontal
        ind_up = (
            (lms[8].y < lms[6].y + 0.015 and cls._dist(lms, 0, 8) > cls._dist(lms, 0, 6) * 1.03)
            or (cls._dist(lms, 0, 8) > cls._dist(lms, 0, 6) * 1.13 and lms[8].y < lms[5].y + 0.025)
        )

        # Medio, Anelar e Mindinho exigem extensao clara para evitar falsos positivos
        med_up = (lms[12].y < lms[10].y) and (cls._dist(lms, 0, 12) > cls._dist(lms, 0, 10) * 1.08)
        ane_up = (lms[16].y < lms[14].y) and (cls._dist(lms, 0, 16) > cls._dist(lms, 0, 14) * 1.08)
        min_up = (lms[20].y < lms[18].y) and (cls._dist(lms, 0, 20) > cls._dist(lms, 0, 18) * 1.06)

        return [thumb_up, ind_up, med_up, ane_up, min_up]

    @classmethod
    def get_palm_eraser(cls, lms, width: int, height: int) -> tuple[int, int, int]:
        """Calcula o centro exato da palma da mao e um raio proporcional para a borracha."""
        px = int(sum(lms[idx].x for idx in cls.PALM_IDS) / len(cls.PALM_IDS) * width)
        py = int(sum(lms[idx].y for idx in cls.PALM_IDS) / len(cls.PALM_IDS) * height)

        palm_px = math.hypot((lms[9].x - lms[0].x) * width, (lms[9].y - lms[0].y) * height)
        radius = int(max(45, min(85, palm_px * 0.55)))
        return px, py, radius

    @classmethod
    def draw_landmarks_fast(cls, frame: np.ndarray, lms) -> None:
        """Desenho leve dos 21 pontos em OpenCV puro (~20x mais rapido que drawing_utils)."""
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

            # Envia para a IA apenas quando chegar um frame novo da camera
            if frame_id != last_submitted_frame_id:
                tracker.submit_frame(frame)
                last_submitted_frame_id = frame_id

            lms, seq = tracker.get_snapshot()
            now = time.perf_counter()

            # Atualiza estado logico e tracos quando chega resultado novo da IA
            if seq != last_processed_seq:
                last_processed_seq = seq

                if lms is not None:
                    thumb_up, ind_up, med_up, ane_up, min_up = tracker.get_fingers_up(lms)

                    raw_x = lms[8].x * FRAME_WIDTH
                    raw_y = lms[8].y * FRAME_HEIGHT
                    if smooth_pt is None:
                        smooth_pt = (raw_x, raw_y)
                    else:
                        smooth_pt = (
                            SMOOTHING_ALPHA * raw_x + (1.0 - SMOOTHING_ALPHA) * smooth_pt[0],
                            SMOOTHING_ALPHA * raw_y + (1.0 - SMOOTHING_ALPHA) * smooth_pt[1],
                        )
                    ix, iy = int(smooth_pt[0]), int(smooth_pt[1])

                    # 1) MODO DESENHO: Apenas indicador levantado (polegar livre)
                    if ind_up and not med_up and not ane_up and not min_up:
                        mode_label = "DESENHO (Indicador)"
                        non_draw_streak = 0
                        last_draw_seen_t = now
                        prev_erase_pt = None

                        if prev_draw_pt is not None:
                            # Evita ligar pontos distantes caso a mao tenha saltado na tela
                            jump = math.hypot(ix - prev_draw_pt[0], iy - prev_draw_pt[1])
                            if jump > 180:
                                prev_draw_pt = None

                        if prev_draw_pt is None:
                            cv2.circle(canvas, (ix, iy), BRUSH_THICKNESS // 2, BRUSH_COLOR, cv2.FILLED)
                            cv2.circle(canvas_mask, (ix, iy), BRUSH_THICKNESS // 2, 255, cv2.FILLED)
                        else:
                            cv2.line(canvas, prev_draw_pt, (ix, iy), BRUSH_COLOR, BRUSH_THICKNESS)
                            cv2.line(canvas_mask, prev_draw_pt, (ix, iy), 255, BRUSH_THICKNESS)

                        prev_draw_pt = (ix, iy)
                        overlay_cursor = ("DRAW", (ix, iy), 10)

                    # 2) MODO BORRACHA: Palma da mao aberta (os 5 dedos levantados)
                    elif thumb_up and ind_up and med_up and ane_up and min_up:
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            mode_label = "BORRACHA (Palma - 5 Dedos)"
                            prev_draw_pt = None

                            px, py, radius = tracker.get_palm_eraser(lms, FRAME_WIDTH, FRAME_HEIGHT)
                            cv2.circle(canvas, (px, py), radius, (0, 0, 0), cv2.FILLED)
                            cv2.circle(canvas_mask, (px, py), radius, 0, cv2.FILLED)

                            if prev_erase_pt is not None:
                                cv2.line(canvas, prev_erase_pt, (px, py), (0, 0, 0), radius * 2)
                                cv2.line(canvas_mask, prev_erase_pt, (px, py), 0, radius * 2)

                            prev_erase_pt = (px, py)
                            overlay_cursor = ("ERASE", (px, py), radius)

                    # 3) MODO LIVRE: Indicador e dedo medio em pe
                    elif ind_up and med_up and not ane_up and not min_up:
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            mode_label = "LIVRE (2 Dedos)"
                            prev_draw_pt = None
                            prev_erase_pt = None

                            mx = int(lms[12].x * FRAME_WIDTH)
                            my = int(lms[12].y * FRAME_HEIGHT)
                            cx, cy = (ix + mx) // 2, (iy + my) // 2
                            overlay_cursor = ("FREE", (cx, cy), 10)

                    # Outras poses (ex: mao fechada ou transicao rapida)
                    else:
                        prev_erase_pt = None
                        # Mantem o traco vivo por ate 200ms contra falhas rapidas de luz/leitura
                        if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                            mode_label = "DESENHO (Indicador)"
                            overlay_cursor = ("DRAW", prev_draw_pt, 10)
                        else:
                            mode_label = "PAUSADO"
                            prev_draw_pt = None
                            overlay_cursor = None
                else:
                    prev_erase_pt = None
                    if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                        # Segura a conexao do traco caso a mao pisque por 1-3 frames na contraluz
                        overlay_cursor = ("DRAW", prev_draw_pt, 10)
                    else:
                        mode_label = "AGUARDANDO MAO"
                        prev_draw_pt = None
                        smooth_pt = None
                        overlay_cursor = None

            # Fusao ultra-rapida O(1) in-place usando mascara binaria (0.03 ms/frame!)
            cv2.copyTo(canvas, canvas_mask, frame)

            # Renderiza esqueleto leve da mao e cursor do modo ativo
            tracker.draw_landmarks_fast(frame, lms)

            if overlay_cursor is not None:
                kind, (cx, cy), rad = overlay_cursor
                if kind == "DRAW":
                    cv2.circle(frame, (cx, cy), rad, BRUSH_COLOR, cv2.FILLED)
                elif kind == "FREE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 0), 2)
                elif kind == "ERASE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), cv2.FILLED)

            # Calculo suavizado de FPS para leitura estavel
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

            # Se nenhum frame novo chegou ainda da camera, aguarda 1ms sem consumir 100% da CPU
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

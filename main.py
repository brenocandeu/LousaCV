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

# Configuracoes Visuais (HD 1280x720)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
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
        with self._lock:
            return self._landmarks_2d, self._landmarks_3d, self._result_seq

    @staticmethod
    def _dist3(w, a: int, b: int) -> float:
        return math.sqrt(
            (w[a].x - w[b].x) ** 2 + (w[a].y - w[b].y) ** 2 + (w[a].z - w[b].z) ** 2
        )

    @staticmethod
    def _finger_cos3(w, mcp: int, pip: int, tip: int) -> float:
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
        return (
            cls._finger_cos3(w, mcp, pip, tip) > cos_min
            and cls._dist3(w, 0, tip) > cls._dist3(w, 0, pip) * ratio_min
        )

    @classmethod
    def classify_gesture(
        cls, lms2d, w3d, width: int, height: int
    ) -> tuple[str, tuple[float, float]]:
        """
        Classifica o gesto ergonomico da mao e retorna (modo, ponto_xy):
        - 'DRAW_INDEX' / 'DRAW_PEN': Indicador em qualquer angulo 3D OU Pega de Caneta (pinca).
        - 'FREE': 2 dedos estendidos (Indicador + Medio) para modo livre e clique no menu.
        - 'ERASE': 4 dedos estendidos (Indicador, Medio, Anelar e Mindinho).
        - 'PAUSED': Mao fechada ou em repouso.
        """
        ind_ext = cls._is_extended_3d(w3d, 5, 6, 8, cos_min=0.35, ratio_min=1.05)
        med_ext = cls._is_extended_3d(w3d, 9, 10, 12, cos_min=0.55, ratio_min=1.10)
        ane_ext = cls._is_extended_3d(w3d, 13, 14, 16, cos_min=0.55, ratio_min=1.10)
        min_ext = cls._is_extended_3d(w3d, 17, 18, 20, cos_min=0.50, ratio_min=1.08)

        palm_2d = max(math.hypot(lms2d[9].x - lms2d[0].x, lms2d[9].y - lms2d[0].y), 1e-6)
        pinch_2d = math.hypot(lms2d[4].x - lms2d[8].x, lms2d[4].y - lms2d[8].y) < 0.35 * palm_2d
        pinch_3d = cls._dist3(w3d, 4, 8) < 0.035
        not_fist = (
            cls._dist3(w3d, 0, 8) > cls._dist3(w3d, 0, 5) * 1.18
            and cls._finger_cos3(w3d, 5, 6, 8) > 0.15
        )
        pen_pinch = pinch_2d and pinch_3d and not_fist and not (med_ext and ane_ext and min_ext)

        # 1) BORRACHA: 4 dedos estendidos
        if ind_ext and med_ext and ane_ext and min_ext and not pen_pinch:
            ex = sum(lms2d[idx].x for idx in (8, 12, 16, 20)) * 0.25 * width
            ey = sum(lms2d[idx].y for idx in (8, 12, 16, 20)) * 0.25 * height
            return "ERASE", (ex, ey)

        # 2) MODO LIVRE / MENU: 2 dedos estendidos (indicador + medio)
        if ind_ext and med_ext and not ane_ext and not min_ext and not pen_pinch:
            cx = (lms2d[8].x + lms2d[12].x) * 0.5 * width
            cy = (lms2d[8].y + lms2d[12].y) * 0.5 * height
            return "FREE", (cx, cy)

        # 3) DESENHO ERGONOMICO: Pinca (Caneta) OU Indicador livre em 360 graus
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


class SideMenu:
    """
    Menu Lateral Interativo operado com o gesto de 2 dedos:
    1) Botao COR -> Abre o Seletor de Cores HSV (Color Picker estilo Photoshop: Quadrado SV + Barra Hue).
    2) Botao TRACO -> Abre o painel deslizante e presets de grossura do traco.
    3) Botao LIMPAR -> Apaga tudo que esta na tela (com confirmacao rapida anti-esbarrao).
    """

    PICKER_SIZE = 180
    HUE_BAR_W = 28

    def __init__(self) -> None:
        # Estado atual do pincel (inicia em Vermelho vibrante HSV=(0, 255, 255) e espessura 12px)
        self.hue = 0
        self.sat = 255
        self.val = 255
        self.brush_color: tuple[int, int, int] = (0, 0, 255)
        self.brush_thickness: int = 12

        # Dimensoes da barra lateral esquerda
        self.bar_rect = (14, 150, 104, 494)
        self.btn_color_rect = (22, 162, 96, 256)
        self.btn_size_rect = (22, 272, 96, 366)
        self.btn_clear_rect = (22, 382, 96, 476)

        # Painel popup (abre ao lado direito da barra lateral)
        self.active_panel: str | None = None  # None | 'COLOR' | 'SIZE'
        self.color_panel_rect = (116, 150, 366, 396)
        self.sv_origin = (130, 198)  # (x, y) do quadrado 180x180
        self.hue_origin = (324, 198)  # (x, y) da barra arco-iris 28x180

        self.size_panel_rect = (116, 240, 380, 405)
        self.slider_x1 = 138
        self.slider_x2 = 358
        self.slider_y = 305
        self.size_presets = [6, 12, 20, 30]

        # Pre-renderiza matrizes HSV do Color Picker estilo Photoshop (custo O(1) em tempo real)
        s_grid = np.linspace(0, 255, self.PICKER_SIZE, dtype=np.uint8)[None, :]
        v_grid = np.linspace(255, 25, self.PICKER_SIZE, dtype=np.uint8)[:, None]
        self._sv_hsv_buf = np.zeros((self.PICKER_SIZE, self.PICKER_SIZE, 3), dtype=np.uint8)
        self._sv_hsv_buf[:, :, 1] = s_grid
        self._sv_hsv_buf[:, :, 2] = v_grid
        self._sv_bgr_cache = np.zeros_like(self._sv_hsv_buf)

        hue_col = np.linspace(0, 179, self.PICKER_SIZE, dtype=np.uint8)[:, None]
        hue_hsv = np.zeros((self.PICKER_SIZE, self.HUE_BAR_W, 3), dtype=np.uint8)
        hue_hsv[:, :, 0] = hue_col
        hue_hsv[:, :, 1] = 255
        hue_hsv[:, :, 2] = 255
        self._hue_bar_bgr = cv2.cvtColor(hue_hsv, cv2.COLOR_HSV2BGR)

        self._rebuild_sv_square()
        self._update_bgr_from_hsv()

        # Controle de hover/clique com 2 dedos
        self._hover_target: str | None = None
        self._hover_start_t: float = 0.0
        self._latched_btn: str | None = None
        self._clear_progress: float = 0.0
        self._clear_flash_until: float = 0.0

    def _rebuild_sv_square(self) -> None:
        self._sv_hsv_buf[:, :, 0] = self.hue
        self._sv_bgr_cache = cv2.cvtColor(self._sv_hsv_buf, cv2.COLOR_HSV2BGR)

    def _update_bgr_from_hsv(self) -> None:
        px_hsv = np.uint8([[[self.hue, self.sat, self.val]]])
        bgr = cv2.cvtColor(px_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        # Garante intensidade minima para nao sumir na mascara binaria
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        if b + g + r < 30:
            b, g, r = 30, 30, 30
        self.brush_color = (b, g, r)

    @staticmethod
    def _in_rect(x: int, y: int, rect: tuple[int, int, int, int], pad: int = 0) -> bool:
        x1, y1, x2, y2 = rect
        return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)

    def is_over_ui(self, x: int, y: int) -> bool:
        """Retorna True se o ponto estiver sobre a barra lateral ou painel aberto."""
        if self._in_rect(x, y, self.bar_rect, pad=6):
            return True
        if self.active_panel == "COLOR" and self._in_rect(x, y, self.color_panel_rect, pad=6):
            return True
        if self.active_panel == "SIZE" and self._in_rect(x, y, self.size_panel_rect, pad=6):
            return True
        return False

    def close_panels(self) -> None:
        self.active_panel = None

    def reset_hover(self) -> None:
        self._hover_target = None
        self._latched_btn = None
        self._clear_progress = 0.0

    def update_two_fingers(self, cx: int, cy: int) -> str | None:
        """
        Processa a interacao do cursor de 2 dedos no menu lateral e nos popups abertos.
        Retorna 'CLEAR_ALL' quando o botao de apagar tudo e confirmado.
        """
        now = time.perf_counter()

        # 1) Interacao direta dentro do Color Picker estilo Photoshop (se estiver aberto)
        if self.active_panel == "COLOR" and self._in_rect(cx, cy, self.color_panel_rect, pad=8):
            self._hover_target = None
            self._latched_btn = None
            self._clear_progress = 0.0

            sx0, sy0 = self.sv_origin
            hx0, hy0 = self.hue_origin

            # Arraste no Quadrado 2D de Saturacao x Brilho
            if (sx0 - 8) <= cx <= (sx0 + self.PICKER_SIZE + 8) and (sy0 - 8) <= cy <= (sy0 + self.PICKER_SIZE + 8):
                rel_x = max(0, min(self.PICKER_SIZE - 1, cx - sx0))
                rel_y = max(0, min(self.PICKER_SIZE - 1, cy - sy0))
                self.sat = int(round((rel_x / (self.PICKER_SIZE - 1)) * 255))
                self.val = int(round(255 - (rel_y / (self.PICKER_SIZE - 1)) * 230))
                self._update_bgr_from_hsv()
                return "COLOR_CHANGED"

            # Arraste na Barra Vertical de Matiz (Arco-iris)
            if (hx0 - 8) <= cx <= (hx0 + self.HUE_BAR_W + 8) and (hy0 - 8) <= cy <= (hy0 + self.PICKER_SIZE + 8):
                rel_y = max(0, min(self.PICKER_SIZE - 1, cy - hy0))
                new_hue = int(round((rel_y / (self.PICKER_SIZE - 1)) * 179))
                if new_hue != self.hue:
                    self.hue = new_hue
                    self._rebuild_sv_square()
                    self._update_bgr_from_hsv()
                return "HUE_CHANGED"

            return None

        # 2) Interacao dentro do Painel de Grossura do Traco (se estiver aberto)
        if self.active_panel == "SIZE" and self._in_rect(cx, cy, self.size_panel_rect, pad=8):
            self._hover_target = None
            self._latched_btn = None
            self._clear_progress = 0.0

            # Barra deslizante horizontal (4px a 36px)
            if (self.slider_x1 - 10) <= cx <= (self.slider_x2 + 10) and (self.slider_y - 24) <= cy <= (self.slider_y + 24):
                ratio = (cx - self.slider_x1) / float(self.slider_x2 - self.slider_x1)
                ratio = max(0.0, min(1.0, ratio))
                self.brush_thickness = int(round(4 + ratio * 32))
                return "SIZE_CHANGED"

            # Botoes rapidos de espessura (presets)
            for idx, preset in enumerate(self.size_presets):
                px = 148 + idx * 58
                py = 365
                if math.hypot(cx - px, cy - py) <= 22:
                    self.brush_thickness = preset
                    return "SIZE_CHANGED"

            return None

        # 3) Botoes principais da Barra Lateral (COR, TRACO, LIMPAR)
        hit: str | None = None
        if self._in_rect(cx, cy, self.btn_color_rect):
            hit = "BTN_COLOR"
        elif self._in_rect(cx, cy, self.btn_size_rect):
            hit = "BTN_SIZE"
        elif self._in_rect(cx, cy, self.btn_clear_rect):
            hit = "BTN_CLEAR"

        if hit is None:
            self._hover_target = None
            self._latched_btn = None
            self._clear_progress = 0.0
            return None

        if hit != self._hover_target:
            self._hover_target = hit
            self._hover_start_t = now
            self._clear_progress = 0.0

        elapsed = now - self._hover_start_t

        # Botao COR e TRACO: clique rapido por hover (0.22s) com trava para nao piscar
        if hit in ("BTN_COLOR", "BTN_SIZE"):
            self._clear_progress = 0.0
            if self._latched_btn != hit and elapsed >= 0.22:
                self._latched_btn = hit
                target_panel = "COLOR" if hit == "BTN_COLOR" else "SIZE"
                self.active_panel = None if self.active_panel == target_panel else target_panel
                return f"TOGGLE_{target_panel}"
            return None

        # Botao LIMPAR TUDO: segurar 0.45s preenche barra de confirmacao e limpa a tela
        if hit == "BTN_CLEAR":
            if self._latched_btn == "BTN_CLEAR":
                return None
            self._clear_progress = min(1.0, elapsed / 0.45)
            if elapsed >= 0.45:
                self._latched_btn = "BTN_CLEAR"
                self._clear_progress = 0.0
                self._clear_flash_until = now + 0.30
                self.active_panel = None
                return "CLEAR_ALL"

        return None

    @staticmethod
    def _draw_translucent_box(
        frame: np.ndarray, rect: tuple[int, int, int, int], bg_bgr: tuple[int, int, int], alpha: float = 0.80
    ) -> None:
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]
        overlay = np.full_like(roi, bg_bgr)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    def draw(self, frame: np.ndarray, cursor_2f: tuple[int, int] | None = None) -> None:
        """Desenha a barra lateral e o painel popup ativo (Color Picker ou Grossura)."""
        now = time.perf_counter()

        # Fundo da Barra Lateral
        self._draw_translucent_box(frame, self.bar_rect, (28, 28, 34), alpha=0.78)
        cv2.rectangle(frame, (self.bar_rect[0], self.bar_rect[1]), (self.bar_rect[2], self.bar_rect[3]), (95, 95, 110), 1)

        # --- BOTAO 1: COR ---
        bx1, by1, bx2, by2 = self.btn_color_rect
        btn1_bg = (65, 65, 80) if self.active_panel == "COLOR" else (42, 42, 50)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn1_bg, cv2.FILLED)
        border1 = (255, 255, 255) if (self.active_panel == "COLOR" or self._hover_target == "BTN_COLOR") else (110, 110, 125)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border1, 2 if self.active_panel == "COLOR" else 1)

        ccx, ccy = (bx1 + bx2) // 2, by1 + 36
        cv2.circle(frame, (ccx, ccy), 20, self.brush_color, cv2.FILLED)
        cv2.circle(frame, (ccx, ccy), 21, (240, 240, 240), 2)
        cv2.putText(frame, "COR", (bx1 + 19, by2 - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

        # --- BOTAO 2: GROSSURA DO TRACO ---
        sx1, sy1, sx2, sy2 = self.btn_size_rect
        btn2_bg = (65, 65, 80) if self.active_panel == "SIZE" else (42, 42, 50)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), btn2_bg, cv2.FILLED)
        border2 = (255, 255, 255) if (self.active_panel == "SIZE" or self._hover_target == "BTN_SIZE") else (110, 110, 125)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), border2, 2 if self.active_panel == "SIZE" else 1)

        scx, scy = (sx1 + sx2) // 2, sy1 + 36
        preview_r = max(2, min(20, self.brush_thickness // 2))
        cv2.circle(frame, (scx, scy), preview_r, self.brush_color, cv2.FILLED)
        cv2.circle(frame, (scx, scy), preview_r + 1, (230, 230, 230), 1)
        cv2.putText(
            frame,
            f"{self.brush_thickness}px",
            (sx1 + 17, sy2 - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # --- BOTAO 3: APAGAR TUDO ---
        cx1, cy1, cx2, cy2 = self.btn_clear_rect
        clear_bg = (90, 90, 235) if now < self._clear_flash_until else (45, 40, 95)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), clear_bg, cv2.FILLED)

        # Barra de preenchimento progressivo ao segurar 2 dedos em LIMPAR
        if self._clear_progress > 0.0:
            prog_h = int((cy2 - cy1) * self._clear_progress)
            cv2.rectangle(frame, (cx1, cy2 - prog_h), (cx2, cy2), (60, 60, 215), cv2.FILLED)

        border3 = (255, 255, 255) if self._hover_target == "BTN_CLEAR" else (110, 100, 160)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), border3, 2 if self._hover_target == "BTN_CLEAR" else 1)

        # Icone simples de lixeira / X + rotulo LIMPAR
        icx, icy = (cx1 + cx2) // 2, cy1 + 34
        cv2.line(frame, (icx - 11, icy - 11), (icx + 11, icy + 11), (255, 210, 210), 2, cv2.LINE_AA)
        cv2.line(frame, (icx + 11, icy - 11), (icx - 11, icy + 11), (255, 210, 210), 2, cv2.LINE_AA)
        cv2.putText(frame, "LIMPAR", (cx1 + 8, cy2 - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)

        # --- POPUP 1: SELETOR DE CORES HSV (ESTILO PHOTOSHOP) ---
        if self.active_panel == "COLOR":
            px1, py1, px2, py2 = self.color_panel_rect
            self._draw_translucent_box(frame, self.color_panel_rect, (24, 24, 30), alpha=0.88)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (160, 160, 175), 1)

            cv2.putText(
                frame,
                "Seletor de Cor",
                (px1 + 14, py1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            # Amostra retangular da cor escolhida no topo direito do popup
            cv2.rectangle(frame, (px2 - 72, py1 + 10), (px2 - 14, py1 + 36), self.brush_color, cv2.FILLED)
            cv2.rectangle(frame, (px2 - 72, py1 + 10), (px2 - 14, py1 + 36), (235, 235, 235), 1)

            # Quadrado 2D Saturacao x Brilho
            sx0, sy0 = self.sv_origin
            frame[sy0 : sy0 + self.PICKER_SIZE, sx0 : sx0 + self.PICKER_SIZE] = self._sv_bgr_cache
            cv2.rectangle(frame, (sx0, sy0), (sx0 + self.PICKER_SIZE, sy0 + self.PICKER_SIZE), (180, 180, 180), 1)

            # Mira circular dentro do Quadrado SV
            sel_x = sx0 + int((self.sat / 255.0) * (self.PICKER_SIZE - 1))
            sel_y = sy0 + int(((255 - self.val) / 230.0) * (self.PICKER_SIZE - 1))
            sel_y = max(sy0, min(sy0 + self.PICKER_SIZE - 1, sel_y))
            cv2.circle(frame, (sel_x, sel_y), 7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (sel_x, sel_y), 6, (255, 255, 255), 2, cv2.LINE_AA)

            # Barra Vertical de Matiz (Hue / Arco-iris)
            hx0, hy0 = self.hue_origin
            frame[hy0 : hy0 + self.PICKER_SIZE, hx0 : hx0 + self.HUE_BAR_W] = self._hue_bar_bgr
            cv2.rectangle(frame, (hx0, hy0), (hx0 + self.HUE_BAR_W, hy0 + self.PICKER_SIZE), (180, 180, 180), 1)

            hue_y = hy0 + int((self.hue / 179.0) * (self.PICKER_SIZE - 1))
            cv2.rectangle(frame, (hx0 - 3, hue_y - 3), (hx0 + self.HUE_BAR_W + 3, hue_y + 3), (255, 255, 255), 2)

        # --- POPUP 2: PAINEL DE GROSSURA DO TRACO ---
        elif self.active_panel == "SIZE":
            px1, py1, px2, py2 = self.size_panel_rect
            self._draw_translucent_box(frame, self.size_panel_rect, (24, 24, 30), alpha=0.88)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (160, 160, 175), 1)

            cv2.putText(
                frame,
                f"Grossura do Traco: {self.brush_thickness}px",
                (px1 + 14, py1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            # Trilho do Slider Horizontal
            cv2.line(frame, (self.slider_x1, self.slider_y), (self.slider_x2, self.slider_y), (110, 110, 125), 6, cv2.LINE_AA)
            knob_ratio = (self.brush_thickness - 4) / 32.0
            knob_x = int(self.slider_x1 + knob_ratio * (self.slider_x2 - self.slider_x1))
            cv2.line(frame, (self.slider_x1, self.slider_y), (knob_x, self.slider_y), self.brush_color, 6, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, self.slider_y), 11, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, self.slider_y), 8, self.brush_color, cv2.FILLED, cv2.LINE_AA)

            # Botoes de tamanho rapido (presets)
            for idx, preset in enumerate(self.size_presets):
                cx_p = 148 + idx * 58
                cy_p = 365
                is_curr = abs(self.brush_thickness - preset) <= 1
                cv2.circle(frame, (cx_p, cy_p), 20, (65, 65, 80) if is_curr else (40, 40, 48), cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, (cx_p, cy_p), 20, (255, 255, 255) if is_curr else (105, 105, 120), 2 if is_curr else 1, cv2.LINE_AA)
                cv2.circle(frame, (cx_p, cy_p), max(2, preset // 2), self.brush_color, cv2.FILLED, cv2.LINE_AA)


def main() -> None:
    window_name = "Lousa Magica"

    cam = CameraStream(src=0, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    if not cam.opened:
        print("Erro: Nao foi possivel acessar a webcam (indice 0).")
        return

    tracker = HandTracker(max_hands=1)
    menu = SideMenu()

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
    two_finger_cursor: tuple[int, int] | None = None

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
                        menu.reset_hover()
                        two_finger_cursor = None
                        non_draw_streak = 0
                        last_draw_seen_t = now
                        prev_erase_pt = None

                        # Evita riscar por cima da barra lateral; fecha popups ao voltar a desenhar na lousa
                        if menu.is_over_ui(sx, sy):
                            prev_draw_pt = None
                            mode_label = "MENU (Use 2 dedos para clicar)"
                            overlay_cursor = ("FREE", (sx, sy), 9)
                        else:
                            if menu.active_panel is not None:
                                menu.close_panels()

                            mode_label = (
                                "DESENHO (Caneta/Pinca)"
                                if gesture == "DRAW_PEN"
                                else "DESENHO (Indicador Livre 3D)"
                            )
                            if prev_draw_pt is not None and math.hypot(sx - prev_draw_pt[0], sy - prev_draw_pt[1]) > 180:
                                prev_draw_pt = None

                            color = menu.brush_color
                            thick = menu.brush_thickness

                            if prev_draw_pt is None:
                                cv2.circle(canvas, (sx, sy), thick // 2, color, cv2.FILLED)
                                cv2.circle(canvas_mask, (sx, sy), thick // 2, 255, cv2.FILLED)
                            else:
                                cv2.line(canvas, prev_draw_pt, (sx, sy), color, thick)
                                cv2.line(canvas_mask, prev_draw_pt, (sx, sy), 255, thick)

                            prev_draw_pt = (sx, sy)
                            overlay_cursor = ("DRAW", (sx, sy), max(6, thick // 2))

                    # 2) MODO BORRACHA: 4 dedos estendidos
                    elif gesture == "ERASE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            menu.reset_hover()
                            two_finger_cursor = None
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

                    # 3) MODO LIVRE / SELECIONAR NO MENU: 2 dedos estendidos (indicador + medio)
                    elif gesture == "FREE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            mode_label = "LIVRE / MENU (2 Dedos)"
                            prev_draw_pt = None
                            prev_erase_pt = None

                            two_finger_cursor = (sx, sy)
                            overlay_cursor = ("FREE", (sx, sy), 10)

                            action = menu.update_two_fingers(sx, sy)
                            if action == "CLEAR_ALL":
                                canvas.fill(0)
                                canvas_mask.fill(0)

                    # Mao fechada / soltou a pinca -> pausa o traco
                    else:
                        prev_erase_pt = None
                        two_finger_cursor = None
                        menu.reset_hover()
                        non_draw_streak += 1
                        if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC and non_draw_streak < 3:
                            overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                        else:
                            mode_label = "LIVRE / PAUSADO"
                            prev_draw_pt = None
                            overlay_cursor = ("FREE", (int(raw_x), int(raw_y)), 8)
                else:
                    prev_erase_pt = None
                    two_finger_cursor = None
                    menu.reset_hover()
                    if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                        overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                    else:
                        mode_label = "AGUARDANDO MAO"
                        prev_draw_pt = None
                        smooth_pt = None
                        overlay_cursor = None

            # Fusao ultra-rapida O(1) in-place (0.03 ms/frame)
            cv2.copyTo(canvas, canvas_mask, frame)

            # Desenha esqueleto da mao, Menu Lateral (e popups) e cursor ativo
            tracker.draw_landmarks_fast(frame, lms2d)
            menu.draw(frame, two_finger_cursor)

            if overlay_cursor is not None:
                kind, (cx, cy), rad = overlay_cursor
                if kind == "DRAW":
                    cv2.circle(frame, (cx, cy), rad, menu.brush_color, cv2.FILLED)
                    cv2.circle(frame, (cx, cy), rad + 2, (255, 255, 255), 1)
                elif kind == "FREE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 3, menu.brush_color, cv2.FILLED)
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

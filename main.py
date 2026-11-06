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
    e reconhecimento ergonomico invariante ao angulo da mao:
    - 1 dedo (ou Pinca Caneta) = Desenhar
    - 2 dedos = Movimento Livre (Hover)
    - 3 dedos = Clique / Arraste no Menu
    - 4 dedos = Borracha
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
        Classifica o gesto 3D da mao e retorna (modo, ponto_xy):
        - 'DRAW_INDEX' / 'DRAW_PEN': 1 dedo em qualquer angulo 3D OU Pinca (Caneta).
        - 'FREE': 2 dedos (Indicador + Medio) -> Mover cursor livremente sem clicar.
        - 'CLICK': 3 dedos (Indicador + Medio + Anelar) -> Simula clique/arraste no menu!
          Nota: usa a mesma ancora (Indicador + Medio)/2 do modo FREE para o cursor nao pular ao clicar!
        - 'ERASE': 4 dedos (Indicador + Medio + Anelar + Mindinho) -> Borracha.
        - 'PAUSED': Mao fechada ou em repouso.
        """
        ind_ext = cls._is_extended_3d(w3d, 5, 6, 8, cos_min=0.35, ratio_min=1.05)
        med_ext = cls._is_extended_3d(w3d, 9, 10, 12, cos_min=0.50, ratio_min=1.08)
        ane_ext = cls._is_extended_3d(w3d, 13, 14, 16, cos_min=0.45, ratio_min=1.07)
        # Mindinho exige extensao bem clara para nao confundir 3 dedos (clique) com 4 dedos (borracha)
        min_ext = cls._is_extended_3d(w3d, 17, 18, 20, cos_min=0.68, ratio_min=1.13)

        palm_2d = max(math.hypot(lms2d[9].x - lms2d[0].x, lms2d[9].y - lms2d[0].y), 1e-6)
        pinch_2d = math.hypot(lms2d[4].x - lms2d[8].x, lms2d[4].y - lms2d[8].y) < 0.35 * palm_2d
        pinch_3d = cls._dist3(w3d, 4, 8) < 0.035
        not_fist = (
            cls._dist3(w3d, 0, 8) > cls._dist3(w3d, 0, 5) * 1.18
            and cls._finger_cos3(w3d, 5, 6, 8) > 0.15
        )
        pen_pinch = pinch_2d and pinch_3d and not_fist and not (med_ext and ane_ext)

        # Ancora estavel do cursor (ponto medio entre indicador e medio: nao desloca ao levantar o 3o dedo!)
        cursor_x = (lms2d[8].x + lms2d[12].x) * 0.5 * width
        cursor_y = (lms2d[8].y + lms2d[12].y) * 0.5 * height

        # 1) BORRACHA: 4 dedos claramente estendidos
        if ind_ext and med_ext and ane_ext and min_ext and not pen_pinch:
            ex = sum(lms2d[idx].x for idx in (8, 12, 16, 20)) * 0.25 * width
            ey = sum(lms2d[idx].y for idx in (8, 12, 16, 20)) * 0.25 * height
            return "ERASE", (ex, ey)

        # 2) CLIQUE / SELECAO NO MENU: 3 dedos estendidos (indicador + medio + anelar)
        if ind_ext and med_ext and ane_ext and not min_ext and not pen_pinch:
            return "CLICK", (cursor_x, cursor_y)

        # 3) MODO LIVRE (HOVER): 2 dedos estendidos (indicador + medio)
        if ind_ext and med_ext and not ane_ext and not min_ext and not pen_pinch:
            return "FREE", (cursor_x, cursor_y)

        # 4) DESENHO ERGONOMICO: Pinca (Caneta) OU Indicador livre em 360 graus
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
    Menu Lateral Direito com Design Moderno (Glassmorphism + Cantos Arredondados):
    - 2 dedos (FREE)  -> Move livremente entre as opcoes (apenas efeito visual de hover).
    - 3 dedos (CLICK) -> Clica para abrir/fechar opcoes, arrasta cor no Color Picker ou espessura no Slider.
    """

    PICKER_SIZE = 184
    HUE_BAR_W = 26

    def __init__(self) -> None:
        self.hue = 0
        self.sat = 255
        self.val = 255
        self.brush_color: tuple[int, int, int] = (0, 0, 255)
        self.brush_thickness: int = 12

        # Dock flutuante no LADO DIREITO da tela (x = 1172..1264, centralizada em Y)
        self.bar_rect = (1172, 165, 1264, 525)
        self.btn_color_rect = (1182, 180, 1254, 276)
        self.btn_size_rect = (1182, 294, 1254, 390)
        self.btn_clear_rect = (1182, 408, 1254, 504)

        # Popups abrem a ESQUERDA da dock direita
        self.active_panel: str | None = None  # None | 'COLOR' | 'SIZE'

        self.color_panel_rect = (890, 165, 1158, 425)
        self.sv_origin = (908, 222)   # Quadrado 184x184
        self.hue_origin = (1108, 222)  # Barra Hue 26x184

        self.size_panel_rect = (880, 245, 1158, 425)
        self.slider_x1 = 906
        self.slider_x2 = 1132
        self.slider_y = 320
        self.size_presets = [6, 12, 20, 30]

        # Cache de mascaras arredondadas para renderizacao Glassmorphism ultra-rapida
        self._mask_cache: dict[tuple[int, int, int], np.ndarray] = {}

        # Pre-renderiza matrizes HSV do Color Picker estilo Photoshop
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

        # Estados de interacao (Hover 2 dedos vs Clique 3 dedos)
        self.hovered_item: str | None = None
        self._was_clicking: bool = False
        self._last_btn_click_t: float = 0.0
        self._clear_flash_until: float = 0.0

    def _get_rounded_mask(self, w: int, h: int, r: int) -> np.ndarray:
        key = (w, h, r)
        if key not in self._mask_cache:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(m, (r, 0), (w - r, h), 255, cv2.FILLED)
            cv2.rectangle(m, (0, r), (w, h - r), 255, cv2.FILLED)
            for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
                cv2.circle(m, (cx, cy), r, 255, cv2.FILLED, cv2.LINE_AA)
            self._mask_cache[key] = m
        return self._mask_cache[key]

    @staticmethod
    def _draw_rounded_border(
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        radius: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        x1, y1, x2, y2 = rect
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_rounded_fill(
        self,
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        radius: int,
        color: tuple[int, int, int],
        alpha: float = 1.0,
        frosted: bool = False,
        shadow: bool = False,
    ) -> None:
        x1, y1, x2, y2 = rect
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return

        if shadow:
            sx1 = max(0, x1 - 4)
            sy1 = max(0, y1 + 5)
            sx2 = min(FRAME_WIDTH, x2 + 4)
            sy2 = min(FRAME_HEIGHT, y2 + 9)
            s_mask = self._get_rounded_mask(sx2 - sx1, sy2 - sy1, radius + 2)
            s_roi = frame[sy1:sy2, sx1:sx2]
            dark_s = (s_roi.astype(np.uint16) * 55 // 100).astype(np.uint8)
            cv2.copyTo(dark_s, s_mask, s_roi)

        roi = frame[y1:y2, x1:x2]
        base = cv2.GaussianBlur(roi, (15, 15), 0) if frosted else roi
        solid = np.full_like(roi, color)
        blended = solid if alpha >= 0.99 else cv2.addWeighted(solid, alpha, base, 1.0 - alpha, 0)
        mask = self._get_rounded_mask(w, h, radius)
        cv2.copyTo(blended, mask, roi)

    def _rebuild_sv_square(self) -> None:
        self._sv_hsv_buf[:, :, 0] = self.hue
        self._sv_bgr_cache = cv2.cvtColor(self._sv_hsv_buf, cv2.COLOR_HSV2BGR)

    def _update_bgr_from_hsv(self) -> None:
        px_hsv = np.uint8([[[self.hue, self.sat, self.val]]])
        bgr = cv2.cvtColor(px_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        if b + g + r < 30:
            b, g, r = 30, 30, 30
        self.brush_color = (b, g, r)

    @property
    def hex_color(self) -> str:
        b, g, r = self.brush_color
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _in_rect(x: int, y: int, rect: tuple[int, int, int, int], pad: int = 0) -> bool:
        x1, y1, x2, y2 = rect
        return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)

    def is_over_ui(self, x: int, y: int, pad: int = 10) -> bool:
        if self._in_rect(x, y, self.bar_rect, pad=pad):
            return True
        if self.active_panel == "COLOR" and self._in_rect(x, y, self.color_panel_rect, pad=pad):
            return True
        if self.active_panel == "SIZE" and self._in_rect(x, y, self.size_panel_rect, pad=pad):
            return True
        return False

    def close_panels(self) -> None:
        self.active_panel = None

    def reset_interaction(self) -> None:
        self.hovered_item = None
        self._was_clicking = False

    def update_cursor(self, cx: int, cy: int, is_clicking: bool) -> str | None:
        """
        Atualiza o Menu Lateral:
        - is_clicking == False (2 dedos): apenas atualiza o destaque de Hover (movimento livre!).
        - is_clicking == True  (3 dedos):
            * Nos botoes da dock: dispara clique unico na borda de subida (just_clicked).
            * No Color Picker e Slider: permite clicar e arrastar suavemente com os 3 dedos!
        """
        now = time.perf_counter()
        just_clicked = is_clicking and not self._was_clicking
        self._was_clicking = is_clicking

        # Detecta qual elemento esta sob o cursor para efeito visual de hover
        self.hovered_item = None
        if self._in_rect(cx, cy, self.btn_color_rect, pad=4):
            self.hovered_item = "BTN_COLOR"
        elif self._in_rect(cx, cy, self.btn_size_rect, pad=4):
            self.hovered_item = "BTN_SIZE"
        elif self._in_rect(cx, cy, self.btn_clear_rect, pad=4):
            self.hovered_item = "BTN_CLEAR"
        elif self.active_panel == "SIZE":
            for idx, preset in enumerate(self.size_presets):
                px = 918 + idx * 62
                py = 380
                if math.hypot(cx - px, cy - py) <= 22:
                    self.hovered_item = f"PRESET_{preset}"
                    break

        if not is_clicking:
            return None

        # 1) Clique / Arraste continuo dentro do Color Picker estilo Photoshop (3 dedos ativos)
        if self.active_panel == "COLOR" and self._in_rect(cx, cy, self.color_panel_rect, pad=10):
            sx0, sy0 = self.sv_origin
            hx0, hy0 = self.hue_origin

            if (sx0 - 10) <= cx <= (sx0 + self.PICKER_SIZE + 10) and (sy0 - 10) <= cy <= (sy0 + self.PICKER_SIZE + 10):
                rel_x = max(0, min(self.PICKER_SIZE - 1, cx - sx0))
                rel_y = max(0, min(self.PICKER_SIZE - 1, cy - sy0))
                self.sat = int(round((rel_x / (self.PICKER_SIZE - 1)) * 255))
                self.val = int(round(255 - (rel_y / (self.PICKER_SIZE - 1)) * 230))
                self._update_bgr_from_hsv()
                return "COLOR_CHANGED"

            if (hx0 - 10) <= cx <= (hx0 + self.HUE_BAR_W + 10) and (hy0 - 10) <= cy <= (hy0 + self.PICKER_SIZE + 10):
                rel_y = max(0, min(self.PICKER_SIZE - 1, cy - hy0))
                new_hue = int(round((rel_y / (self.PICKER_SIZE - 1)) * 179))
                if new_hue != self.hue:
                    self.hue = new_hue
                    self._rebuild_sv_square()
                    self._update_bgr_from_hsv()
                return "HUE_CHANGED"

            return None

        # 2) Clique / Arraste dentro do Painel de Grossura do Traco (3 dedos ativos)
        if self.active_panel == "SIZE" and self._in_rect(cx, cy, self.size_panel_rect, pad=10):
            if (self.slider_x1 - 12) <= cx <= (self.slider_x2 + 12) and (self.slider_y - 26) <= cy <= (self.slider_y + 26):
                ratio = (cx - self.slider_x1) / float(self.slider_x2 - self.slider_x1)
                ratio = max(0.0, min(1.0, ratio))
                self.brush_thickness = int(round(4 + ratio * 32))
                return "SIZE_CHANGED"

            if just_clicked:
                for idx, preset in enumerate(self.size_presets):
                    px = 918 + idx * 62
                    py = 380
                    if math.hypot(cx - px, cy - py) <= 24:
                        self.brush_thickness = preset
                        return "SIZE_CHANGED"
            return None

        # 3) Clique unico (just_clicked com 3 dedos) nos botoes da Dock Lateral Direita
        if just_clicked and (now - self._last_btn_click_t) >= 0.30:
            if self.hovered_item == "BTN_COLOR":
                self._last_btn_click_t = now
                self.active_panel = None if self.active_panel == "COLOR" else "COLOR"
                return "TOGGLE_COLOR"

            if self.hovered_item == "BTN_SIZE":
                self._last_btn_click_t = now
                self.active_panel = None if self.active_panel == "SIZE" else "SIZE"
                return "TOGGLE_SIZE"

            if self.hovered_item == "BTN_CLEAR":
                self._last_btn_click_t = now
                self._clear_flash_until = now + 0.32
                self.active_panel = None
                return "CLEAR_ALL"

        return None

    def draw(self, frame: np.ndarray) -> None:
        """Renderiza a Dock Lateral Direita em vidro fosco (Glassmorphism) e os popups modernos."""
        now = time.perf_counter()

        # Container principal da Dock Direita
        self._draw_rounded_fill(
            frame, self.bar_rect, radius=22, color=(24, 26, 34), alpha=0.76, frosted=True, shadow=True
        )
        self._draw_rounded_border(frame, self.bar_rect, radius=22, color=(85, 92, 112), thickness=1)

        # --- CARD 1: COR ---
        c_active = self.active_panel == "COLOR"
        c_hover = self.hovered_item == "BTN_COLOR"
        c_tint = (62, 68, 88) if c_active else (46, 50, 66) if c_hover else (34, 36, 46)
        self._draw_rounded_fill(frame, self.btn_color_rect, radius=16, color=c_tint, alpha=0.92)
        c_border = (255, 255, 255) if c_active else (185, 195, 220) if c_hover else (68, 72, 90)
        self._draw_rounded_border(frame, self.btn_color_rect, radius=16, color=c_border, thickness=2 if (c_active or c_hover) else 1)

        bx1, by1, bx2, by2 = self.btn_color_rect
        ccx, ccy = (bx1 + bx2) // 2, by1 + 36
        cv2.circle(frame, (ccx, ccy), 22, (235, 235, 245), 2, cv2.LINE_AA)
        cv2.circle(frame, (ccx, ccy), 18, self.brush_color, cv2.FILLED, cv2.LINE_AA)
        # Brilho especular suave no topo da esfera de cor
        cv2.circle(frame, (ccx - 6, ccy - 6), 4, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, "COR", (ccx - 17, by2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 250), 1, cv2.LINE_AA)

        # --- CARD 2: GROSSURA DO TRACO ---
        s_active = self.active_panel == "SIZE"
        s_hover = self.hovered_item == "BTN_SIZE"
        s_tint = (62, 68, 88) if s_active else (46, 50, 66) if s_hover else (34, 36, 46)
        self._draw_rounded_fill(frame, self.btn_size_rect, radius=16, color=s_tint, alpha=0.92)
        s_border = (255, 255, 255) if s_active else (185, 195, 220) if s_hover else (68, 72, 90)
        self._draw_rounded_border(frame, self.btn_size_rect, radius=16, color=s_border, thickness=2 if (s_active or s_hover) else 1)

        sx1, sy1, sx2, sy2 = self.btn_size_rect
        scx, scy = (sx1 + sx2) // 2, sy1 + 36
        cv2.circle(frame, (scx, scy), 21, (90, 96, 115), 1, cv2.LINE_AA)
        preview_r = max(2, min(18, self.brush_thickness // 2))
        cv2.circle(frame, (scx, scy), preview_r, self.brush_color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (scx, scy), preview_r + 1, (240, 240, 245), 1, cv2.LINE_AA)

        size_lbl = f"{self.brush_thickness}px"
        (tw, _), _ = cv2.getTextSize(size_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(frame, size_lbl, (scx - tw // 2, sy2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 240, 250), 1, cv2.LINE_AA)

        # --- CARD 3: LIMPAR TUDO (ICONE VETORIAL DE LIXEIRA) ---
        cl_hover = self.hovered_item == "BTN_CLEAR"
        flashing = now < self._clear_flash_until
        cl_tint = (90, 95, 235) if flashing else (52, 44, 92) if cl_hover else (38, 34, 54)
        self._draw_rounded_fill(frame, self.btn_clear_rect, radius=16, color=cl_tint, alpha=0.92)
        cl_border = (255, 255, 255) if flashing else (160, 150, 255) if cl_hover else (80, 68, 110)
        self._draw_rounded_border(frame, self.btn_clear_rect, radius=16, color=cl_border, thickness=2 if (cl_hover or flashing) else 1)

        cx1, cy1, cx2, cy2 = self.btn_clear_rect
        icx, icy = (cx1 + cx2) // 2, cy1 + 35
        icon_col = (255, 240, 240) if cl_hover else (190, 185, 255)
        # Tampa e alca da lixeira
        cv2.line(frame, (icx - 13, icy - 10), (icx + 13, icy - 10), icon_col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (icx - 5, icy - 14), (icx + 5, icy - 10), icon_col, 1, cv2.LINE_AA)
        # Corpo da lixeira
        trash_body = np.array(
            [[icx - 10, icy - 8], [icx + 10, icy - 8], [icx + 8, icy + 13], [icx - 8, icy + 13]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [trash_body], isClosed=True, color=icon_col, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (icx - 3, icy - 3), (icx - 3, icy + 8), icon_col, 1, cv2.LINE_AA)
        cv2.line(frame, (icx + 3, icy - 3), (icx + 3, icy + 8), icon_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "LIMPAR", (icx - 27, cy2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 245), 1, cv2.LINE_AA)

        # --- POPUP 1: SELETOR DE CORES HSV (ESTILO PHOTOSHOP, A ESQUERDA DA DOCK) ---
        if self.active_panel == "COLOR":
            px1, py1, px2, py2 = self.color_panel_rect
            self._draw_rounded_fill(
                frame, self.color_panel_rect, radius=20, color=(22, 24, 32), alpha=0.86, frosted=True, shadow=True
            )
            self._draw_rounded_border(frame, self.color_panel_rect, radius=20, color=(105, 112, 135), thickness=1)

            cv2.putText(
                frame,
                "PALETA DE COR",
                (px1 + 18, py1 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 235, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                self.hex_color,
                (px2 - 135, py1 + 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (170, 178, 195),
                1,
                cv2.LINE_AA,
            )

            # Pilula arredondada de amostra da cor atual
            pill_rect = (px2 - 64, py1 + 14, px2 - 18, py1 + 42)
            self._draw_rounded_fill(frame, pill_rect, radius=10, color=self.brush_color, alpha=1.0)
            self._draw_rounded_border(frame, pill_rect, radius=10, color=(235, 235, 245), thickness=1)

            # Quadrado 2D Saturacao x Brilho
            sx0, sy0 = self.sv_origin
            frame[sy0 : sy0 + self.PICKER_SIZE, sx0 : sx0 + self.PICKER_SIZE] = self._sv_bgr_cache
            cv2.rectangle(
                frame,
                (sx0, sy0),
                (sx0 + self.PICKER_SIZE, sy0 + self.PICKER_SIZE),
                (140, 145, 165),
                1,
                cv2.LINE_AA,
            )

            sel_x = sx0 + int((self.sat / 255.0) * (self.PICKER_SIZE - 1))
            sel_y = sy0 + int(((255 - self.val) / 230.0) * (self.PICKER_SIZE - 1))
            sel_y = max(sy0, min(sy0 + self.PICKER_SIZE - 1, sel_y))
            cv2.circle(frame, (sel_x, sel_y), 8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (sel_x, sel_y), 7, (255, 255, 255), 2, cv2.LINE_AA)

            # Barra Vertical de Matiz (Hue / Arco-iris)
            hx0, hy0 = self.hue_origin
            frame[hy0 : hy0 + self.PICKER_SIZE, hx0 : hx0 + self.HUE_BAR_W] = self._hue_bar_bgr
            cv2.rectangle(
                frame,
                (hx0, hy0),
                (hx0 + self.HUE_BAR_W, hy0 + self.PICKER_SIZE),
                (140, 145, 165),
                1,
                cv2.LINE_AA,
            )

            hue_y = hy0 + int((self.hue / 179.0) * (self.PICKER_SIZE - 1))
            cv2.rectangle(
                frame,
                (hx0 - 3, hue_y - 4),
                (hx0 + self.HUE_BAR_W + 3, hue_y + 4),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --- POPUP 2: PAINEL DE GROSSURA DO TRACO ---
        elif self.active_panel == "SIZE":
            px1, py1, px2, py2 = self.size_panel_rect
            self._draw_rounded_fill(
                frame, self.size_panel_rect, radius=20, color=(22, 24, 32), alpha=0.86, frosted=True, shadow=True
            )
            self._draw_rounded_border(frame, self.size_panel_rect, radius=20, color=(105, 112, 135), thickness=1)

            cv2.putText(
                frame,
                "GROSSURA DO TRACO",
                (px1 + 18, py1 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (230, 235, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{self.brush_thickness} px",
                (px2 - 68, py1 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Trilho capsula do Slider Horizontal
            cv2.line(
                frame,
                (self.slider_x1, self.slider_y),
                (self.slider_x2, self.slider_y),
                (70, 76, 94),
                8,
                cv2.LINE_AA,
            )
            knob_ratio = (self.brush_thickness - 4) / 32.0
            knob_x = int(self.slider_x1 + knob_ratio * (self.slider_x2 - self.slider_x1))
            cv2.line(
                frame,
                (self.slider_x1, self.slider_y),
                (knob_x, self.slider_y),
                self.brush_color,
                8,
                cv2.LINE_AA,
            )
            cv2.circle(frame, (knob_x, self.slider_y), 12, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, self.slider_y), 8, self.brush_color, cv2.FILLED, cv2.LINE_AA)

            # Presets circulares inferiores
            for idx, preset in enumerate(self.size_presets):
                cx_p = 918 + idx * 62
                cy_p = 380
                is_curr = abs(self.brush_thickness - preset) <= 1
                is_hov = self.hovered_item == f"PRESET_{preset}"
                bg_c = (65, 72, 95) if is_curr else (50, 55, 72) if is_hov else (36, 38, 48)
                bd_c = (255, 255, 255) if is_curr else (180, 190, 215) if is_hov else (85, 92, 112)
                cv2.circle(frame, (cx_p, cy_p), 21, bg_c, cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, (cx_p, cy_p), 21, bd_c, 2 if (is_curr or is_hov) else 1, cv2.LINE_AA)
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

                    # Se estiver sobre o Menu Lateral e o mindinho levantar junto sem querer no clique de 3 dedos,
                    # trata como clique no menu em vez de borracha!
                    if gesture == "ERASE" and menu.is_over_ui(sx, sy, pad=14):
                        cx_ui = int((lms2d[8].x + lms2d[12].x) * 0.5 * FRAME_WIDTH)
                        cy_ui = int((lms2d[8].y + lms2d[12].y) * 0.5 * FRAME_HEIGHT)
                        sx, sy = cx_ui, cy_ui
                        gesture = "CLICK"

                    # 1) MODO DESENHO: Indicador em qualquer angulo 3D OU Pega de Caneta (Pinca)
                    if gesture in ("DRAW_INDEX", "DRAW_PEN"):
                        menu.reset_interaction()
                        non_draw_streak = 0
                        last_draw_seen_t = now
                        prev_erase_pt = None

                        if menu.is_over_ui(sx, sy):
                            prev_draw_pt = None
                            mode_label = "MENU (2 dedos mover | 3 dedos clicar)"
                            overlay_cursor = ("FREE", (sx, sy), 10)
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
                                cv2.circle(canvas, (sx, sy), max(1, thick // 2), color, cv2.FILLED)
                                cv2.circle(canvas_mask, (sx, sy), max(1, thick // 2), 255, cv2.FILLED)
                            else:
                                cv2.line(canvas, prev_draw_pt, (sx, sy), color, thick)
                                cv2.line(canvas_mask, prev_draw_pt, (sx, sy), 255, thick)

                            prev_draw_pt = (sx, sy)
                            overlay_cursor = ("DRAW", (sx, sy), max(6, thick // 2))

                    # 2) MODO BORRACHA: 4 dedos estendidos (fora do menu)
                    elif gesture == "ERASE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            menu.reset_interaction()
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

                    # 3) MODO LIVRE (2 dedos) E CLIQUE NO MENU (3 dedos)
                    elif gesture in ("FREE", "CLICK"):
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            is_clicking = gesture == "CLICK"
                            mode_label = "CLIQUE MENU (3 Dedos)" if is_clicking else "LIVRE (2 Dedos)"
                            prev_draw_pt = None
                            prev_erase_pt = None

                            overlay_cursor = ("CLICK" if is_clicking else "FREE", (sx, sy), 11)

                            action = menu.update_cursor(sx, sy, is_clicking=is_clicking)
                            if action == "CLEAR_ALL":
                                canvas.fill(0)
                                canvas_mask.fill(0)

                    # Mao fechada / soltou a pinca -> pausa o traco
                    else:
                        prev_erase_pt = None
                        menu.reset_interaction()
                        non_draw_streak += 1
                        if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC and non_draw_streak < 3:
                            overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                        else:
                            mode_label = "LIVRE / PAUSADO"
                            prev_draw_pt = None
                            overlay_cursor = ("FREE", (int(raw_x), int(raw_y)), 8)
                else:
                    prev_erase_pt = None
                    menu.reset_interaction()
                    if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                        overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                    else:
                        mode_label = "AGUARDANDO MAO"
                        prev_draw_pt = None
                        smooth_pt = None
                        overlay_cursor = None

            # Fusao ultra-rapida O(1) in-place (0.03 ms/frame)
            cv2.copyTo(canvas, canvas_mask, frame)

            # Desenha esqueleto da mao, Menu Lateral Direito e cursor ativo
            tracker.draw_landmarks_fast(frame, lms2d)
            menu.draw(frame)

            if overlay_cursor is not None:
                kind, (cx, cy), rad = overlay_cursor
                if kind == "DRAW":
                    cv2.circle(frame, (cx, cy), rad, menu.brush_color, cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), rad + 2, (255, 255, 255), 1, cv2.LINE_AA)
                elif kind == "FREE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 3, menu.brush_color, cv2.FILLED, cv2.LINE_AA)
                elif kind == "CLICK":
                    # Feedback visual nitido de clique pressionado (3 dedos)
                    cv2.circle(frame, (cx, cy), rad + 4, (0, 255, 180), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 180), cv2.FILLED, cv2.LINE_AA)
                elif kind == "ERASE":
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2, cv2.LINE_AA)

            curr_time = time.perf_counter()
            inst_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            fps_smooth = 0.90 * fps_smooth + 0.10 * inst_fps

            cv2.putText(
                frame,
                f"FPS: {int(fps_smooth)} | Modo: {mode_label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
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

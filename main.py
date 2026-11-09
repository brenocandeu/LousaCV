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
HOLD_SNAP_SEC = 0.42    # Tempo segurando o dedo parado no fim do traco para converter igual ao Canva


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
        ind_ext = cls._is_extended_3d(w3d, 5, 6, 8, cos_min=0.35, ratio_min=1.05)
        med_ext = cls._is_extended_3d(w3d, 9, 10, 12, cos_min=0.50, ratio_min=1.08)
        ane_ext = cls._is_extended_3d(w3d, 13, 14, 16, cos_min=0.45, ratio_min=1.07)
        min_ext = cls._is_extended_3d(w3d, 17, 18, 20, cos_min=0.68, ratio_min=1.13)

        palm_2d = max(math.hypot(lms2d[9].x - lms2d[0].x, lms2d[9].y - lms2d[0].y), 1e-6)
        pinch_2d = math.hypot(lms2d[4].x - lms2d[8].x, lms2d[4].y - lms2d[8].y) < 0.35 * palm_2d
        pinch_3d = cls._dist3(w3d, 4, 8) < 0.035
        not_fist = (
            cls._dist3(w3d, 0, 8) > cls._dist3(w3d, 0, 5) * 1.18
            and cls._finger_cos3(w3d, 5, 6, 8) > 0.15
        )
        pen_pinch = pinch_2d and pinch_3d and not_fist and not (med_ext and ane_ext)

        cursor_x = (lms2d[8].x + lms2d[12].x) * 0.5 * width
        cursor_y = (lms2d[8].y + lms2d[12].y) * 0.5 * height

        if ind_ext and med_ext and ane_ext and min_ext and not pen_pinch:
            ex = sum(lms2d[idx].x for idx in (8, 12, 16, 20)) * 0.25 * width
            ey = sum(lms2d[idx].y for idx in (8, 12, 16, 20)) * 0.25 * height
            return "ERASE", (ex, ey)

        if ind_ext and med_ext and ane_ext and not min_ext and not pen_pinch:
            return "CLICK", (cursor_x, cursor_y)

        if ind_ext and med_ext and not ane_ext and not min_ext and not pen_pinch:
            return "FREE", (cursor_x, cursor_y)

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


class ShapeRecognizer:
    """
    Motor Inteligente de Formas estilo Canva:
    Transforma tracos com oscilacoes em Linhas Retas, Setas (1 ou 2 tracos),
    Triangulos, Quadrados, Retangulos, Losangos, Pentagonos, Hexagonos e Circulos.
    """

    # Guarda a ultima reta desenhada (p0, pn, timestamp) para suportar setas feitas em 2 tracos
    _last_line: tuple[tuple[int, int], tuple[int, int], float] | None = None

    @staticmethod
    def draw_freehand(
        canvas: np.ndarray,
        mask: np.ndarray,
        stroke: list[tuple[int, int]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if not stroke:
            return
        if len(stroke) == 1:
            r = max(1, thickness // 2)
            cv2.circle(canvas, stroke[0], r, color, cv2.FILLED)
            cv2.circle(mask, stroke[0], r, 255, cv2.FILLED)
            return
        pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)

    @staticmethod
    def _max_perp_dev(pts: np.ndarray, p0: tuple[int, int], pn: tuple[int, int]) -> float:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(pn[0]), float(pn[1])
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        if norm < 1e-5:
            return 0.0
        devs = np.abs(dy * pts[:, 0] - dx * pts[:, 1] + x1 * y0 - y1 * x0) / norm
        return float(np.max(devs))

    @staticmethod
    def _level_line(p0: tuple[int, int], pn: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Alinha automaticamente linhas quase horizontais ou verticais (< 4.5 graus)."""
        dx = pn[0] - p0[0]
        dy = pn[1] - p0[1]
        if abs(dx) > 35 and abs(dy) / max(abs(dx), 1) < 0.08:
            ym = (p0[1] + pn[1]) // 2
            return (p0[0], ym), (pn[0], ym)
        if abs(dy) > 35 and abs(dx) / max(abs(dy), 1) < 0.08:
            xm = (p0[0] + pn[0]) // 2
            return (xm, p0[1]), (xm, pn[1])
        return p0, pn

    @staticmethod
    def _draw_arrow_head(
        canvas: np.ndarray,
        mask: np.ndarray,
        p_from: tuple[int, int],
        p_tip: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        dx = float(p_tip[0] - p_from[0])
        dy = float(p_tip[1] - p_from[1])
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return
        ux, uy = dx / length, dy / length
        head_len = max(24.0, min(54.0, length * 0.22 + thickness * 1.3))
        angle = math.radians(28.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        bx, by = -ux, -uy

        w1 = (
            int(round(p_tip[0] + head_len * (bx * cos_a - by * sin_a))),
            int(round(p_tip[1] + head_len * (bx * sin_a + by * cos_a))),
        )
        w2 = (
            int(round(p_tip[0] + head_len * (bx * cos_a + by * sin_a))),
            int(round(p_tip[1] + head_len * (-bx * sin_a + by * cos_a))),
        )
        wings = np.array([w1, p_tip, w2], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [wings], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [wings], isClosed=False, color=255, thickness=thickness)

    @classmethod
    def _detect_arrow_one_stroke(cls, pts: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Detecta seta desenhada em 1 unico traco continuo (haste reta + gancho/ponta no final)."""
        p0 = (int(pts[0, 0]), int(pts[0, 1]))
        dists_p0 = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
        tip_idx = int(np.argmax(dists_p0))
        max_dist = float(dists_p0[tip_idx])

        if max_dist < 65.0 or tip_idx < 8 or tip_idx >= len(pts) - 4:
            return None

        shaft_pts = pts[: tip_idx + 1]
        tail_pts = pts[tip_idx:]
        shaft_diffs = np.diff(shaft_pts, axis=0)
        shaft_len = float(np.sum(np.hypot(shaft_diffs[:, 0], shaft_diffs[:, 1])))
        tail_diffs = np.diff(tail_pts, axis=0)
        tail_len = float(np.sum(np.hypot(tail_diffs[:, 0], tail_diffs[:, 1])))

        if shaft_len < 60.0:
            return None

        p_tip = (int(pts[tip_idx, 0]), int(pts[tip_idx, 1]))
        shaft_straight = max_dist / shaft_len
        shaft_dev = cls._max_perp_dev(shaft_pts, p0, p_tip) / max(max_dist, 1.0)
        max_tail_spread = float(np.max(np.hypot(tail_pts[:, 0] - p_tip[0], tail_pts[:, 1] - p_tip[1])))
        end_back_dist = float(math.hypot(pts[-1, 0] - p_tip[0], pts[-1, 1] - p_tip[1]))

        if (
            shaft_straight >= 0.78
            and shaft_dev <= 0.14
            and (0.07 * shaft_len <= tail_len <= 0.80 * shaft_len)
            and max_tail_spread <= 0.45 * max_dist
            and end_back_dist >= 14.0
        ):
            return cls._level_line(p0, p_tip)
        return None

    @classmethod
    def _detect_arrow_two_strokes(
        cls, pts: np.ndarray, path_len: float, direct_dist: float
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """
        Detecta quando o usuario desenhou uma reta antes e agora fez o 'V' da ponta da seta
        perto de uma das extremidades da reta!
        """
        if cls._last_line is None:
            return None
        l_p0, l_pn, l_time = cls._last_line
        if (time.perf_counter() - l_time) > 12.0:
            return None

        if not (28.0 <= path_len <= 380.0 and 0.25 <= (direct_dist / path_len) <= 0.84):
            return None

        # Verifica se o vertice central (dobradura do 'V') esta proximo de l_pn ou l_p0
        approx = cv2.approxPolyDP(pts.astype(np.int32).reshape((-1, 1, 2)), 0.08 * path_len, closed=False)
        if len(approx) < 3 or len(approx) > 5:
            return None

        # Ponto do traco mais proximo das pontas da reta anterior
        d_to_pn = float(np.min(np.hypot(pts[:, 0] - l_pn[0], pts[:, 1] - l_pn[1])))
        d_to_p0 = float(np.min(np.hypot(pts[:, 0] - l_p0[0], pts[:, 1] - l_p0[1])))

        if d_to_pn <= 60.0 and d_to_pn <= d_to_p0:
            return l_p0, l_pn
        if d_to_p0 <= 60.0:
            return l_pn, l_p0
        return None

    @classmethod
    def process_and_draw(
        cls,
        canvas: np.ndarray,
        mask: np.ndarray,
        stroke: list[tuple[int, int]],
        color: tuple[int, int, int],
        thickness: int,
        assist_enabled: bool = True,
    ) -> str:
        if not stroke:
            return "NONE"

        if not assist_enabled or len(stroke) < 10:
            cls.draw_freehand(canvas, mask, stroke, color, thickness)
            return "LIVRE"

        pts = np.array(stroke, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        path_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

        if path_len < 32.0:
            cls.draw_freehand(canvas, mask, stroke, color, thickness)
            return "LIVRE"

        p0 = stroke[0]
        pn = stroke[-1]
        direct_dist = math.hypot(pn[0] - p0[0], pn[1] - p0[1])

        # 1) SETA EM 1 TRACO CONTINUO (haste + ponta)
        arrow_1 = cls._detect_arrow_one_stroke(pts)
        if arrow_1 is not None:
            a_start, a_tip = arrow_1
            cv2.line(canvas, a_start, a_tip, color, thickness, cv2.LINE_AA)
            cv2.line(mask, a_start, a_tip, 255, thickness)
            cls._draw_arrow_head(canvas, mask, a_start, a_tip, color, thickness)
            cls._last_line = None
            return "SETA"

        # 2) SETA EM 2 TRACOS (ponta 'V' adicionada na extremidade da ultima reta)
        arrow_2 = cls._detect_arrow_two_strokes(pts, path_len, direct_dist)
        if arrow_2 is not None:
            a_from, a_tip = arrow_2
            cls._draw_arrow_head(canvas, mask, a_from, a_tip, color, thickness)
            cls._last_line = None
            return "SETA"

        # 3) LINHA RETA (corrige linhas com oscilacoes e tortas igual ao Canva)
        max_dev = cls._max_perp_dev(pts, p0, pn)
        if direct_dist >= 35.0 and (direct_dist / path_len) >= 0.74 and (max_dev / direct_dist) <= 0.14:
            lp0, lpn = cls._level_line(p0, pn)
            cv2.line(canvas, lp0, lpn, color, thickness, cv2.LINE_AA)
            cv2.line(mask, lp0, lpn, 255, thickness)
            cls._last_line = (lp0, lpn, time.perf_counter())
            return "LINHA RETA"

        # 4) FORMAS GEOMETRICAS FECHADAS (Triangulo, Quadrado, Retangulo, Losango, Pentagono, Circulo)
        k = max(3, len(pts) // 5)
        end_dists = np.hypot(pts[:k, None, 0] - pts[-k:, 0], pts[:k, None, 1] - pts[-k:, 1])
        min_end_gap = float(np.min(end_dists))
        is_closed = ((direct_dist / path_len) <= 0.32) or ((min_end_gap / path_len) <= 0.15)

        if is_closed and path_len >= 85.0:
            poly_pts = pts.astype(np.int32).reshape((-1, 1, 2))
            # Usa o Convex Hull para eliminar oscilacoes internas das paredes desenhadas no ar
            hull = cv2.convexHull(poly_pts)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)

            if hull_area >= 500.0 and hull_peri > 0:
                circularity = (4.0 * math.pi * hull_area) / (hull_peri * hull_peri)
                approx = cv2.approxPolyDP(hull, 0.038 * hull_peri, True)
                num_verts = len(approx)

                # TRIANGULO (3 vertices)
                if num_verts == 3:
                    tri = approx.reshape(3, 2)
                    # Nivela a base caso dois vertices estejam quase na mesma altura
                    for i in range(3):
                        j = (i + 1) % 3
                        dx_t = abs(int(tri[i, 0]) - int(tri[j, 0]))
                        dy_t = abs(int(tri[i, 1]) - int(tri[j, 1]))
                        if dx_t > 30 and dy_t / max(dx_t, 1) < 0.09:
                            ym = (int(tri[i, 1]) + int(tri[j, 1])) // 2
                            tri[i, 1] = ym
                            tri[j, 1] = ym
                    tri_pts = tri.reshape((-1, 1, 2))
                    cv2.polylines(canvas, [tri_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                    cv2.polylines(mask, [tri_pts], isClosed=True, color=255, thickness=thickness)
                    cls._last_line = None
                    return "TRIANGULO"

                # QUADRADO, RETANGULO OU LOSANGO (4 vertices e nao-circular)
                if num_verts == 4 and circularity < 0.89:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    approx_area = cv2.contourArea(approx)
                    fill_ratio = approx_area / max(float(bw * bh), 1.0)
                    cx, cy = x + bw // 2, y + bh // 2

                    # Losango (orientado nas diagonais: ocupa ~50% da bounding box)
                    if fill_ratio < 0.66:
                        diamond = np.array(
                            [[cx, y], [x + bw, cy], [cx, y + bh], [x, cy]], dtype=np.int32
                        ).reshape((-1, 1, 2))
                        cv2.polylines(canvas, [diamond], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                        cv2.polylines(mask, [diamond], isClosed=True, color=255, thickness=thickness)
                        cls._last_line = None
                        return "LOSANGO"

                    # Quadrado perfeito (proporcao largura/altura proxima de 1:1)
                    aspect = bw / max(float(bh), 1.0)
                    if 0.82 <= aspect <= 1.22:
                        half = (bw + bh) // 4
                        cv2.rectangle(
                            canvas, (cx - half, cy - half), (cx + half, cy + half), color, thickness, cv2.LINE_AA
                        )
                        cv2.rectangle(mask, (cx - half, cy - half), (cx + half, cy + half), 255, thickness)
                        cls._last_line = None
                        return "QUADRADO"

                    # Retangulo alinhado
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, thickness, cv2.LINE_AA)
                    cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, thickness)
                    cls._last_line = None
                    return "RETANGULO"

                # CIRCULO PERFEITO (alta circularidade ou muitos vertices suaves)
                if circularity >= 0.84 or num_verts > 6:
                    (ccx, ccy), enc_r = cv2.minEnclosingCircle(hull)
                    area_r = math.sqrt(hull_area / math.pi)
                    radius = max(12, int(round((enc_r + area_r) * 0.5)))
                    center = (int(round(ccx)), int(round(ccy)))
                    cv2.circle(canvas, center, radius, color, thickness, cv2.LINE_AA)
                    cv2.circle(mask, center, radius, 255, thickness)
                    cls._last_line = None
                    return "CIRCULO"

                # PENTAGONO / HEXAGONO (5 ou 6 vertices com quinas definidas)
                if num_verts in (5, 6):
                    cv2.polylines(canvas, [approx], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                    cv2.polylines(mask, [approx], isClosed=True, color=255, thickness=thickness)
                    cls._last_line = None
                    return "PENTAGONO" if num_verts == 5 else "HEXAGONO"

        # Fallback: mantem o traco original (escrita livre, letras, numeros)
        cls.draw_freehand(canvas, mask, stroke, color, thickness)
        return "LIVRE"


class SideMenu:
    """
    Menu Lateral Direito com Design Moderno (Glassmorphism + Cantos Arredondados):
    - 2 dedos (FREE)  -> Move livremente entre as opcoes (apenas efeito visual de hover).
    - 3 dedos (CLICK) -> Clica nos botoes e arrasta a cor ou espessura.
    """

    PICKER_SIZE = 184
    HUE_BAR_W = 26

    def __init__(self) -> None:
        self.hue = 0
        self.sat = 255
        self.val = 255
        self.brush_color: tuple[int, int, int] = (0, 0, 255)
        self.brush_thickness: int = 12
        self.shape_assist_active: bool = True

        # Dock flutuante no LADO DIREITO da tela (centralizada verticalmente)
        self.bar_rect = (1172, 114, 1264, 580)
        self.btn_color_rect = (1182, 128, 1254, 224)
        self.btn_size_rect = (1182, 238, 1254, 334)
        self.btn_snap_rect = (1182, 348, 1254, 444)
        self.btn_clear_rect = (1182, 458, 1254, 554)

        # Popups abrem a ESQUERDA da dock direita
        self.active_panel: str | None = None  # None | 'COLOR' | 'SIZE'

        self.color_panel_rect = (890, 128, 1158, 388)
        self.sv_origin = (908, 185)
        self.hue_origin = (1108, 185)

        self.size_panel_rect = (880, 210, 1158, 390)
        self.slider_x1 = 906
        self.slider_x2 = 1132
        self.slider_y = 285
        self.size_presets = [6, 12, 20, 30]

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
        now = time.perf_counter()
        just_clicked = is_clicking and not self._was_clicking
        self._was_clicking = is_clicking

        self.hovered_item = None
        if self._in_rect(cx, cy, self.btn_color_rect, pad=4):
            self.hovered_item = "BTN_COLOR"
        elif self._in_rect(cx, cy, self.btn_size_rect, pad=4):
            self.hovered_item = "BTN_SIZE"
        elif self._in_rect(cx, cy, self.btn_snap_rect, pad=4):
            self.hovered_item = "BTN_SNAP"
        elif self._in_rect(cx, cy, self.btn_clear_rect, pad=4):
            self.hovered_item = "BTN_CLEAR"
        elif self.active_panel == "SIZE":
            for idx, preset in enumerate(self.size_presets):
                px = 918 + idx * 62
                py = 345
                if math.hypot(cx - px, cy - py) <= 22:
                    self.hovered_item = f"PRESET_{preset}"
                    break

        if not is_clicking:
            return None

        # 1) Clique / Arraste continuo dentro do Color Picker estilo Photoshop (3 dedos)
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

        # 2) Clique / Arraste dentro do Painel de Grossura do Traco (3 dedos)
        if self.active_panel == "SIZE" and self._in_rect(cx, cy, self.size_panel_rect, pad=10):
            if (self.slider_x1 - 12) <= cx <= (self.slider_x2 + 12) and (self.slider_y - 26) <= cy <= (self.slider_y + 26):
                ratio = (cx - self.slider_x1) / float(self.slider_x2 - self.slider_x1)
                ratio = max(0.0, min(1.0, ratio))
                self.brush_thickness = int(round(4 + ratio * 32))
                return "SIZE_CHANGED"

            if just_clicked:
                for idx, preset in enumerate(self.size_presets):
                    px = 918 + idx * 62
                    py = 345
                    if math.hypot(cx - px, cy - py) <= 24:
                        self.brush_thickness = preset
                        return "SIZE_CHANGED"
            return None

        # 3) Clique unico (3 dedos) nos cards da Dock Lateral Direita
        if just_clicked and (now - self._last_btn_click_t) >= 0.30:
            if self.hovered_item == "BTN_COLOR":
                self._last_btn_click_t = now
                self.active_panel = None if self.active_panel == "COLOR" else "COLOR"
                return "TOGGLE_COLOR"

            if self.hovered_item == "BTN_SIZE":
                self._last_btn_click_t = now
                self.active_panel = None if self.active_panel == "SIZE" else "SIZE"
                return "TOGGLE_SIZE"

            if self.hovered_item == "BTN_SNAP":
                self._last_btn_click_t = now
                self.shape_assist_active = not self.shape_assist_active
                self.active_panel = None
                return "TOGGLE_SNAP"

            if self.hovered_item == "BTN_CLEAR":
                self._last_btn_click_t = now
                self._clear_flash_until = now + 0.32
                self.active_panel = None
                return "CLEAR_ALL"

        return None

    def draw(self, frame: np.ndarray) -> None:
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

        # --- CARD 3: FORMAS INTELIGENTES (ESTILO CANVA ON/OFF) ---
        g_hover = self.hovered_item == "BTN_SNAP"
        g_on = self.shape_assist_active
        g_tint = (42, 115, 62) if g_on else (46, 50, 66) if g_hover else (34, 36, 46)
        self._draw_rounded_fill(frame, self.btn_snap_rect, radius=16, color=g_tint, alpha=0.92)
        g_border = (170, 255, 190) if (g_on and g_hover) else (120, 220, 145) if g_on else (185, 195, 220) if g_hover else (68, 72, 90)
        self._draw_rounded_border(frame, self.btn_snap_rect, radius=16, color=g_border, thickness=2 if (g_on or g_hover) else 1)

        gx1, gy1, gx2, gy2 = self.btn_snap_rect
        gcx, gcy = (gx1 + gx2) // 2, gy1 + 34
        icon_g_col = (240, 255, 245) if g_on else (160, 168, 185)
        tri_pts = np.array([[gcx - 14, gcy + 11], [gcx - 2, gcy - 12], [gcx + 10, gcy + 11]], dtype=np.int32)
        cv2.polylines(frame, [tri_pts], isClosed=True, color=icon_g_col, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (gcx + 8, gcy - 3), 8, icon_g_col, 2, cv2.LINE_AA)
        snap_lbl = "FORMAS" if g_on else "LIVRE"
        (tw_g, _), _ = cv2.getTextSize(snap_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        cv2.putText(frame, snap_lbl, (gcx - tw_g // 2, gy2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, icon_g_col, 1, cv2.LINE_AA)

        # --- CARD 4: LIMPAR TUDO (ICONE VETORIAL DE LIXEIRA) ---
        cl_hover = self.hovered_item == "BTN_CLEAR"
        flashing = now < self._clear_flash_until
        cl_tint = (90, 95, 235) if flashing else (52, 44, 92) if cl_hover else (38, 34, 54)
        self._draw_rounded_fill(frame, self.btn_clear_rect, radius=16, color=cl_tint, alpha=0.92)
        cl_border = (255, 255, 255) if flashing else (160, 150, 255) if cl_hover else (80, 68, 110)
        self._draw_rounded_border(frame, self.btn_clear_rect, radius=16, color=cl_border, thickness=2 if (cl_hover or flashing) else 1)

        cx1, cy1, cx2, cy2 = self.btn_clear_rect
        icx, icy = (cx1 + cx2) // 2, cy1 + 35
        icon_col = (255, 240, 240) if cl_hover else (190, 185, 255)
        cv2.line(frame, (icx - 13, icy - 10), (icx + 13, icy - 10), icon_col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (icx - 5, icy - 14), (icx + 5, icy - 10), icon_col, 1, cv2.LINE_AA)
        trash_body = np.array(
            [[icx - 10, icy - 8], [icx + 10, icy - 8], [icx + 8, icy + 13], [icx - 8, icy + 13]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [trash_body], isClosed=True, color=icon_col, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (icx - 3, icy - 3), (icx - 3, icy + 8), icon_col, 1, cv2.LINE_AA)
        cv2.line(frame, (icx + 3, icy - 3), (icx + 3, icy + 8), icon_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "LIMPAR", (icx - 27, cy2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 245), 1, cv2.LINE_AA)

        # --- POPUP 1: SELETOR DE CORES HSV (ESTILO PHOTOSHOP) ---
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

            pill_rect = (px2 - 64, py1 + 14, px2 - 18, py1 + 42)
            self._draw_rounded_fill(frame, pill_rect, radius=10, color=self.brush_color, alpha=1.0)
            self._draw_rounded_border(frame, pill_rect, radius=10, color=(235, 235, 245), thickness=1)

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

            for idx, preset in enumerate(self.size_presets):
                cx_p = 918 + idx * 62
                cy_p = 345
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

    # Camada Definitiva (formas corrigidas e desenhos consolidados) + Camada Temporaria (rascunho atual)
    permanent_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    permanent_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    temp_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    temp_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    current_stroke: list[tuple[int, int]] = []
    has_temp_drawing = False
    snapped_hold_lock = False

    prev_draw_pt: tuple[int, int] | None = None
    prev_erase_pt: tuple[int, int] | None = None
    smooth_pt: tuple[float, float] | None = None
    hold_anchor_pt: tuple[int, int] | None = None
    hold_since_t: float = 0.0

    last_submitted_frame_id = -1
    last_processed_seq = -1
    last_draw_seen_t = 0.0
    non_draw_streak = 0

    mode_label = "AGUARDANDO MAO"
    shape_badge_text = ""
    shape_badge_until = 0.0
    overlay_cursor: tuple[str, tuple[int, int], int] | None = None

    fps_smooth = 30.0
    prev_time = time.perf_counter()

    def finalize_current_stroke() -> None:
        nonlocal has_temp_drawing, prev_draw_pt, hold_anchor_pt, shape_badge_text, shape_badge_until
        if current_stroke:
            detected = ShapeRecognizer.process_and_draw(
                permanent_canvas,
                permanent_mask,
                current_stroke,
                menu.brush_color,
                menu.brush_thickness,
                menu.shape_assist_active,
            )
            if detected not in ("NONE", "LIVRE"):
                shape_badge_text = f"Forma: {detected}"
                shape_badge_until = time.perf_counter() + 1.4
            current_stroke.clear()
        if has_temp_drawing:
            temp_canvas.fill(0)
            temp_mask.fill(0)
            has_temp_drawing = False
        prev_draw_pt = None
        hold_anchor_pt = None

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
                            finalize_current_stroke()
                            snapped_hold_lock = False
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
                            thick = menu.brush_thickness

                            # Se acabou de converter a forma segurando o dedo parado (estilo Canva),
                            # aguarda o usuario mover o dedo (> 28px) para iniciar um novo traco
                            if snapped_hold_lock:
                                if hold_anchor_pt is None or math.hypot(sx - hold_anchor_pt[0], sy - hold_anchor_pt[1]) > 28:
                                    snapped_hold_lock = False
                                    hold_anchor_pt = (sx, sy)
                                    hold_since_t = now
                                overlay_cursor = ("DRAW", (sx, sy), max(6, thick // 2))
                            else:
                                if prev_draw_pt is not None and math.hypot(sx - prev_draw_pt[0], sy - prev_draw_pt[1]) > 180:
                                    finalize_current_stroke()

                                color = menu.brush_color
                                current_stroke.append((sx, sy))
                                has_temp_drawing = True

                                if prev_draw_pt is None:
                                    cv2.circle(temp_canvas, (sx, sy), max(1, thick // 2), color, cv2.FILLED)
                                    cv2.circle(temp_mask, (sx, sy), max(1, thick // 2), 255, cv2.FILLED)
                                    hold_anchor_pt = (sx, sy)
                                    hold_since_t = now
                                else:
                                    cv2.line(temp_canvas, prev_draw_pt, (sx, sy), color, thick, cv2.LINE_AA)
                                    cv2.line(temp_mask, prev_draw_pt, (sx, sy), 255, thick)

                                    # Gatilho Estilo Canva: se segurar o dedo parado no final do traco por 0.42s,
                                    # converte a linha/forma/seta imediatamente na tela!
                                    if hold_anchor_pt is None or math.hypot(sx - hold_anchor_pt[0], sy - hold_anchor_pt[1]) > 11:
                                        hold_anchor_pt = (sx, sy)
                                        hold_since_t = now
                                    elif (
                                        menu.shape_assist_active
                                        and len(current_stroke) >= 14
                                        and (now - hold_since_t) >= HOLD_SNAP_SEC
                                    ):
                                        last_pt = (sx, sy)
                                        finalize_current_stroke()
                                        snapped_hold_lock = True
                                        hold_anchor_pt = last_pt

                                if not snapped_hold_lock:
                                    prev_draw_pt = (sx, sy)
                                overlay_cursor = ("DRAW", (sx, sy), max(6, thick // 2))

                    # 2) MODO BORRACHA: 4 dedos estendidos (fora do menu)
                    elif gesture == "ERASE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            finalize_current_stroke()
                            snapped_hold_lock = False
                            menu.reset_interaction()
                            mode_label = "BORRACHA (4 Dedos)"
                            ex, ey = int(raw_x), int(raw_y)

                            cv2.circle(permanent_canvas, (ex, ey), ERASER_RADIUS, (0, 0, 0), cv2.FILLED)
                            cv2.circle(permanent_mask, (ex, ey), ERASER_RADIUS, 0, cv2.FILLED)

                            if prev_erase_pt is not None:
                                cv2.line(permanent_canvas, prev_erase_pt, (ex, ey), (0, 0, 0), ERASER_RADIUS * 2)
                                cv2.line(permanent_mask, prev_erase_pt, (ex, ey), 0, ERASER_RADIUS * 2)

                            prev_erase_pt = (ex, ey)
                            overlay_cursor = ("ERASE", (ex, ey), ERASER_RADIUS)

                    # 3) MODO LIVRE (2 dedos) E CLIQUE NO MENU (3 dedos)
                    elif gesture in ("FREE", "CLICK"):
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            finalize_current_stroke()
                            snapped_hold_lock = False
                            is_clicking = gesture == "CLICK"
                            mode_label = "CLIQUE MENU (3 Dedos)" if is_clicking else "LIVRE (2 Dedos)"
                            prev_erase_pt = None

                            overlay_cursor = ("CLICK" if is_clicking else "FREE", (sx, sy), 11)

                            action = menu.update_cursor(sx, sy, is_clicking=is_clicking)
                            if action == "CLEAR_ALL":
                                permanent_canvas.fill(0)
                                permanent_mask.fill(0)
                                temp_canvas.fill(0)
                                temp_mask.fill(0)
                                current_stroke.clear()
                                has_temp_drawing = False
                                ShapeRecognizer._last_line = None

                    # Mao fechada / soltou a pinca -> pausa e consolida/corrige o traco
                    else:
                        prev_erase_pt = None
                        menu.reset_interaction()
                        non_draw_streak += 1
                        if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC and non_draw_streak < 3:
                            overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                        else:
                            finalize_current_stroke()
                            snapped_hold_lock = False
                            mode_label = "LIVRE / PAUSADO"
                            overlay_cursor = ("FREE", (int(raw_x), int(raw_y)), 8)
                else:
                    prev_erase_pt = None
                    menu.reset_interaction()
                    if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                        overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                    else:
                        finalize_current_stroke()
                        snapped_hold_lock = False
                        mode_label = "AGUARDANDO MAO"
                        smooth_pt = None
                        overlay_cursor = None

            # Fusao ultra-rapida O(1) do permanent_canvas + rascunho temp_canvas (0.05 ms/frame)
            cv2.copyTo(permanent_canvas, permanent_mask, frame)
            if has_temp_drawing:
                cv2.copyTo(temp_canvas, temp_mask, frame)

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

            if now < shape_badge_until:
                cv2.putText(
                    frame,
                    shape_badge_text,
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 255, 200),
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

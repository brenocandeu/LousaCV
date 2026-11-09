import math
import threading
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from config import MODEL_PATH, MODEL_URL


class HandTracker:
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    )

    def __init__(self, max_hands=1):
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
        self._gamma_lut = np.array(
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
    def _ensure_model():
        if not MODEL_PATH.exists():
            print("Baixando modelo hand_landmarker.task...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

    def _on_result(self, result, _output_image, _timestamp_ms):
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

    def submit_frame(self, frame_bgr):
        with self._lock:
            if self._busy:
                return
            self._busy = True

        small = cv2.resize(frame_bgr, self._infer_size, interpolation=cv2.INTER_LINEAR)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y_chan = ycrcb[:, :, 0]
        if float(np.mean(y_chan)) < 90.0:
            y_chan = cv2.LUT(y_chan, self._gamma_lut)
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
    def _dist3(w, a, b):
        return math.sqrt(
            (w[a].x - w[b].x) ** 2 + (w[a].y - w[b].y) ** 2 + (w[a].z - w[b].z) ** 2
        )

    @staticmethod
    def _finger_cos3(w, mcp, pip, tip):
        v1x, v1y, v1z = w[pip].x - w[mcp].x, w[pip].y - w[mcp].y, w[pip].z - w[mcp].z
        v2x, v2y, v2z = w[tip].x - w[pip].x, w[tip].y - w[pip].y, w[tip].z - w[pip].z
        n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
        if n1 < 1e-6 or n2 < 1e-6:
            return -1.0
        return (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)

    @classmethod
    def _is_extended_3d(cls, w, mcp, pip, tip, cos_min=0.50, ratio_min=1.10):
        return (
            cls._finger_cos3(w, mcp, pip, tip) > cos_min
            and cls._dist3(w, 0, tip) > cls._dist3(w, 0, pip) * ratio_min
        )

    @classmethod
    def classify_gesture(cls, lms2d, w3d, width, height):
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
            return "DRAW_INDEX", (lms2d[8].x * width, lms2d[8].y * height)

        return "PAUSED", (lms2d[8].x * width, lms2d[8].y * height)

    @classmethod
    def draw_landmarks(cls, frame, lms):
        if lms is None:
            return
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        for a, b in cls.HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (220, 220, 220), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 120), cv2.FILLED)

    def close(self):
        self.detector.close()


from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

ERASER_RADIUS = 45
SMOOTHING_ALPHA = 0.55
DRAW_GRACE_SEC = 0.20
HOLD_SNAP_SEC = 0.42


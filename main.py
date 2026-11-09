import math
import time
import cv2
import numpy as np

from camera import CameraStream
from config import (
    DRAW_GRACE_SEC,
    ERASER_RADIUS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HOLD_SNAP_SEC,
    SMOOTHING_ALPHA,
)
from menu import SideMenu
from shapes import ShapeRecognizer
from tracker import HandTracker


def main():
    window_name = "Lousa Magica"

    cam = CameraStream(src=0, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    if not cam.opened:
        print("Erro: Nao foi possivel acessar a webcam.")
        return

    tracker = HandTracker(max_hands=1)
    menu = SideMenu()

    permanent_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    permanent_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    temp_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    temp_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    current_stroke = []
    has_temp_drawing = False
    snapped_hold_lock = False

    prev_draw_pt = None
    prev_erase_pt = None
    smooth_pt = None
    hold_anchor_pt = None
    hold_since_t = 0.0

    last_submitted_frame_id = -1
    last_processed_seq = -1
    last_draw_seen_t = 0.0
    non_draw_streak = 0

    mode_label = "AGUARDANDO MAO"
    shape_badge_text = ""
    shape_badge_until = 0.0
    overlay_cursor = None

    fps_smooth = 30.0
    prev_time = time.perf_counter()

    def finalize_stroke():
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
                        sx = int((lms2d[8].x + lms2d[12].x) * 0.5 * FRAME_WIDTH)
                        sy = int((lms2d[8].y + lms2d[12].y) * 0.5 * FRAME_HEIGHT)
                        gesture = "CLICK"

                    if gesture in ("DRAW_INDEX", "DRAW_PEN"):
                        menu.reset_interaction()
                        non_draw_streak = 0
                        last_draw_seen_t = now
                        prev_erase_pt = None

                        if menu.is_over_ui(sx, sy):
                            finalize_stroke()
                            snapped_hold_lock = False
                            mode_label = "MENU (2 dedos mover | 3 dedos clicar)"
                            overlay_cursor = ("FREE", (sx, sy), 10)
                        else:
                            if menu.active_panel is not None:
                                menu.close_panels()

                            mode_label = "DESENHO (Pinca)" if gesture == "DRAW_PEN" else "DESENHO (Indicador)"
                            thick = menu.brush_thickness

                            if snapped_hold_lock:
                                if hold_anchor_pt is None or math.hypot(sx - hold_anchor_pt[0], sy - hold_anchor_pt[1]) > 28:
                                    snapped_hold_lock = False
                                    hold_anchor_pt = (sx, sy)
                                    hold_since_t = now
                                overlay_cursor = ("DRAW", (sx, sy), max(6, thick // 2))
                            else:
                                if prev_draw_pt is not None and math.hypot(sx - prev_draw_pt[0], sy - prev_draw_pt[1]) > 180:
                                    finalize_stroke()

                                color = menu.brush_color
                                current_stroke.append((sx, sy))
                                has_temp_drawing = True

                                if prev_draw_pt is None:
                                    r = max(1, thick // 2)
                                    cv2.circle(temp_canvas, (sx, sy), r, color, cv2.FILLED)
                                    cv2.circle(temp_mask, (sx, sy), r, 255, cv2.FILLED)
                                    hold_anchor_pt = (sx, sy)
                                    hold_since_t = now
                                else:
                                    cv2.line(temp_canvas, prev_draw_pt, (sx, sy), color, thick, cv2.LINE_AA)
                                    cv2.line(temp_mask, prev_draw_pt, (sx, sy), 255, thick)

                                    if hold_anchor_pt is None or math.hypot(sx - hold_anchor_pt[0], sy - hold_anchor_pt[1]) > 11:
                                        hold_anchor_pt = (sx, sy)
                                        hold_since_t = now
                                    elif (
                                        menu.shape_assist_active
                                        and len(current_stroke) >= 14
                                        and (now - hold_since_t) >= HOLD_SNAP_SEC
                                    ):
                                        last_pt = (sx, sy)
                                        finalize_stroke()
                                        snapped_hold_lock = True
                                        hold_anchor_pt = last_pt

                                if not snapped_hold_lock:
                                    prev_draw_pt = (sx, sy)
                                overlay_cursor = ("DRAW", (sx, sy), max(6, thick // 2))

                    elif gesture == "ERASE":
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            finalize_stroke()
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

                    elif gesture in ("FREE", "CLICK"):
                        non_draw_streak += 1
                        if non_draw_streak >= 2 or prev_draw_pt is None:
                            finalize_stroke()
                            snapped_hold_lock = False
                            is_clicking = gesture == "CLICK"
                            mode_label = "CLIQUE MENU (3 Dedos)" if is_clicking else "LIVRE (2 Dedos)"
                            prev_erase_pt = None
                            overlay_cursor = ("CLICK" if is_clicking else "FREE", (sx, sy), 11)

                            if menu.update_cursor(sx, sy, is_clicking=is_clicking):
                                permanent_canvas.fill(0)
                                permanent_mask.fill(0)
                                temp_canvas.fill(0)
                                temp_mask.fill(0)
                                current_stroke.clear()
                                has_temp_drawing = False
                                ShapeRecognizer._last_line = None

                    else:
                        prev_erase_pt = None
                        menu.reset_interaction()
                        non_draw_streak += 1
                        if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC and non_draw_streak < 3:
                            overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                        else:
                            finalize_stroke()
                            snapped_hold_lock = False
                            mode_label = "LIVRE / PAUSADO"
                            overlay_cursor = ("FREE", (int(raw_x), int(raw_y)), 8)
                else:
                    prev_erase_pt = None
                    menu.reset_interaction()
                    if prev_draw_pt is not None and (now - last_draw_seen_t) <= DRAW_GRACE_SEC:
                        overlay_cursor = ("DRAW", prev_draw_pt, max(6, menu.brush_thickness // 2))
                    else:
                        finalize_stroke()
                        snapped_hold_lock = False
                        mode_label = "AGUARDANDO MAO"
                        smooth_pt = None
                        overlay_cursor = None

            cv2.copyTo(permanent_canvas, permanent_mask, frame)
            if has_temp_drawing:
                cv2.copyTo(temp_canvas, temp_mask, frame)

            tracker.draw_landmarks(frame, lms2d)
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

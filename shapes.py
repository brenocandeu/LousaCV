import math
import time
import cv2
import numpy as np


class ShapeRecognizer:
    _last_line = None

    @staticmethod
    def _line(canvas, mask, p0, pn, color, thickness):
        cv2.line(canvas, p0, pn, color, thickness, cv2.LINE_AA)
        cv2.line(mask, p0, pn, 255, thickness)

    @staticmethod
    def _poly(canvas, mask, pts, closed, color, thickness):
        cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [pts], isClosed=closed, color=255, thickness=thickness)

    @classmethod
    def draw_freehand(cls, canvas, mask, stroke, color, thickness):
        if not stroke:
            return
        if len(stroke) == 1:
            r = max(1, thickness // 2)
            cv2.circle(canvas, stroke[0], r, color, cv2.FILLED)
            cv2.circle(mask, stroke[0], r, 255, cv2.FILLED)
            return
        pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
        cls._poly(canvas, mask, pts, False, color, thickness)

    @staticmethod
    def _max_perp_dev(pts, p0, pn):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(pn[0]), float(pn[1])
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        if norm < 1e-5:
            return 0.0
        devs = np.abs(dy * pts[:, 0] - dx * pts[:, 1] + x1 * y0 - y1 * x0) / norm
        return float(np.max(devs))

    @staticmethod
    def _level_line(p0, pn):
        dx = pn[0] - p0[0]
        dy = pn[1] - p0[1]
        if abs(dx) > 35 and abs(dy) / max(abs(dx), 1) < 0.08:
            ym = (p0[1] + pn[1]) // 2
            return (p0[0], ym), (pn[0], ym)
        if abs(dy) > 35 and abs(dx) / max(abs(dy), 1) < 0.08:
            xm = (p0[0] + pn[0]) // 2
            return (xm, p0[1]), (xm, pn[1])
        return p0, pn

    @classmethod
    def _draw_arrow_head(cls, canvas, mask, p_from, p_tip, color, thickness):
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
        cls._poly(canvas, mask, wings, False, color, thickness)

    @classmethod
    def _detect_arrow_one_stroke(cls, pts):
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
    def _detect_arrow_two_strokes(cls, pts, path_len, direct_dist):
        if cls._last_line is None:
            return None
        l_p0, l_pn, l_time = cls._last_line
        if (time.perf_counter() - l_time) > 12.0:
            return None

        if not (28.0 <= path_len <= 380.0 and 0.25 <= (direct_dist / path_len) <= 0.84):
            return None

        approx = cv2.approxPolyDP(pts.astype(np.int32).reshape((-1, 1, 2)), 0.08 * path_len, closed=False)
        if len(approx) < 3 or len(approx) > 5:
            return None

        d_to_pn = float(np.min(np.hypot(pts[:, 0] - l_pn[0], pts[:, 1] - l_pn[1])))
        d_to_p0 = float(np.min(np.hypot(pts[:, 0] - l_p0[0], pts[:, 1] - l_p0[1])))

        if d_to_pn <= 60.0 and d_to_pn <= d_to_p0:
            return l_p0, l_pn
        if d_to_p0 <= 60.0:
            return l_pn, l_p0
        return None

    @classmethod
    def process_and_draw(cls, canvas, mask, stroke, color, thickness, assist_enabled=True):
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

        arrow_1 = cls._detect_arrow_one_stroke(pts)
        if arrow_1 is not None:
            a_start, a_tip = arrow_1
            cls._line(canvas, mask, a_start, a_tip, color, thickness)
            cls._draw_arrow_head(canvas, mask, a_start, a_tip, color, thickness)
            cls._last_line = None
            return "SETA"

        arrow_2 = cls._detect_arrow_two_strokes(pts, path_len, direct_dist)
        if arrow_2 is not None:
            a_from, a_tip = arrow_2
            cls._draw_arrow_head(canvas, mask, a_from, a_tip, color, thickness)
            cls._last_line = None
            return "SETA"

        max_dev = cls._max_perp_dev(pts, p0, pn)
        if direct_dist >= 35.0 and (direct_dist / path_len) >= 0.74 and (max_dev / direct_dist) <= 0.14:
            lp0, lpn = cls._level_line(p0, pn)
            cls._line(canvas, mask, lp0, lpn, color, thickness)
            cls._last_line = (lp0, lpn, time.perf_counter())
            return "LINHA RETA"

        k = max(3, len(pts) // 5)
        end_dists = np.hypot(pts[:k, None, 0] - pts[-k:, 0], pts[:k, None, 1] - pts[-k:, 1])
        min_end_gap = float(np.min(end_dists))
        is_closed = ((direct_dist / path_len) <= 0.32) or ((min_end_gap / path_len) <= 0.15)

        if is_closed and path_len >= 85.0:
            poly_pts = pts.astype(np.int32).reshape((-1, 1, 2))
            hull = cv2.convexHull(poly_pts)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)

            if hull_area >= 500.0 and hull_peri > 0:
                circularity = (4.0 * math.pi * hull_area) / (hull_peri * hull_peri)
                approx = cv2.approxPolyDP(hull, 0.038 * hull_peri, True)
                num_verts = len(approx)

                if num_verts == 3:
                    tri = approx.reshape(3, 2)
                    for i in range(3):
                        j = (i + 1) % 3
                        dx_t = abs(int(tri[i, 0]) - int(tri[j, 0]))
                        dy_t = abs(int(tri[i, 1]) - int(tri[j, 1]))
                        if dx_t > 30 and dy_t / max(dx_t, 1) < 0.09:
                            ym = (int(tri[i, 1]) + int(tri[j, 1])) // 2
                            tri[i, 1] = ym
                            tri[j, 1] = ym
                    cls._poly(canvas, mask, tri.reshape((-1, 1, 2)), True, color, thickness)
                    cls._last_line = None
                    return "TRIANGULO"

                if num_verts == 4 and circularity < 0.89:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    approx_area = cv2.contourArea(approx)
                    fill_ratio = approx_area / max(float(bw * bh), 1.0)
                    cx, cy = x + bw // 2, y + bh // 2

                    if fill_ratio < 0.66:
                        diamond = np.array(
                            [[cx, y], [x + bw, cy], [cx, y + bh], [x, cy]], dtype=np.int32
                        ).reshape((-1, 1, 2))
                        cls._poly(canvas, mask, diamond, True, color, thickness)
                        cls._last_line = None
                        return "LOSANGO"

                    aspect = bw / max(float(bh), 1.0)
                    if 0.82 <= aspect <= 1.22:
                        half = (bw + bh) // 4
                        p1, p2 = (cx - half, cy - half), (cx + half, cy + half)
                        cv2.rectangle(canvas, p1, p2, color, thickness, cv2.LINE_AA)
                        cv2.rectangle(mask, p1, p2, 255, thickness)
                        cls._last_line = None
                        return "QUADRADO"

                    p1, p2 = (x, y), (x + bw, y + bh)
                    cv2.rectangle(canvas, p1, p2, color, thickness, cv2.LINE_AA)
                    cv2.rectangle(mask, p1, p2, 255, thickness)
                    cls._last_line = None
                    return "RETANGULO"

                if circularity >= 0.84 or num_verts > 6:
                    (ccx, ccy), enc_r = cv2.minEnclosingCircle(hull)
                    area_r = math.sqrt(hull_area / math.pi)
                    radius = max(12, int(round((enc_r + area_r) * 0.5)))
                    center = (int(round(ccx)), int(round(ccy)))
                    cv2.circle(canvas, center, radius, color, thickness, cv2.LINE_AA)
                    cv2.circle(mask, center, radius, 255, thickness)
                    cls._last_line = None
                    return "CIRCULO"

                if num_verts in (5, 6):
                    cls._poly(canvas, mask, approx, True, color, thickness)
                    cls._last_line = None
                    return "PENTAGONO" if num_verts == 5 else "HEXAGONO"

        cls.draw_freehand(canvas, mask, stroke, color, thickness)
        return "LIVRE"


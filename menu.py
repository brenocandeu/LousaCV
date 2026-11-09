import math
import time
import cv2
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT


class SideMenu:
    PICKER_SIZE = 184
    HUE_BAR_W = 26

    def __init__(self):
        self.hue = 0
        self.sat = 255
        self.val = 255
        self.brush_color = (0, 0, 255)
        self.brush_thickness = 12
        self.shape_assist_active = True

        self.bar_rect = (1172, 114, 1264, 580)
        self.btn_color_rect = (1182, 128, 1254, 224)
        self.btn_size_rect = (1182, 238, 1254, 334)
        self.btn_snap_rect = (1182, 348, 1254, 444)
        self.btn_clear_rect = (1182, 458, 1254, 554)

        self.active_panel = None
        self.color_panel_rect = (890, 128, 1158, 388)
        self.sv_origin = (908, 185)
        self.hue_origin = (1108, 185)

        self.size_panel_rect = (880, 210, 1158, 390)
        self.slider_x1 = 906
        self.slider_x2 = 1132
        self.slider_y = 285
        self.size_presets = [6, 12, 20, 30]

        self._mask_cache = {}

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

        self.hovered_item = None
        self._was_clicking = False
        self._last_btn_click_t = 0.0
        self._clear_flash_until = 0.0

    def _get_rounded_mask(self, w, h, r):
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
    def _draw_rounded_border(frame, rect, radius, color, thickness=1):
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

    def _draw_rounded_fill(self, frame, rect, radius, color, alpha=1.0, frosted=False, shadow=False):
        x1, y1, x2, y2 = rect
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return

        if shadow:
            sx1, sy1 = max(0, x1 - 4), max(0, y1 + 5)
            sx2, sy2 = min(FRAME_WIDTH, x2 + 4), min(FRAME_HEIGHT, y2 + 9)
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

    def _draw_card(self, frame, rect, label, tint, border, thick_border=False, text_color=(235, 240, 250)):
        self._draw_rounded_fill(frame, rect, radius=16, color=tint, alpha=0.92)
        self._draw_rounded_border(frame, rect, radius=16, color=border, thickness=2 if thick_border else 1)
        x1, _, x2, y2 = rect
        cx = (x1 + x2) // 2
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, label, (cx - tw // 2, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    def _rebuild_sv_square(self):
        self._sv_hsv_buf[:, :, 0] = self.hue
        self._sv_bgr_cache = cv2.cvtColor(self._sv_hsv_buf, cv2.COLOR_HSV2BGR)

    def _update_bgr_from_hsv(self):
        px_hsv = np.uint8([[[self.hue, self.sat, self.val]]])
        bgr = cv2.cvtColor(px_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        if b + g + r < 30:
            b, g, r = 30, 30, 30
        self.brush_color = (b, g, r)

    @property
    def hex_color(self):
        b, g, r = self.brush_color
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _in_rect(x, y, rect, pad=0):
        x1, y1, x2, y2 = rect
        return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)

    def is_over_ui(self, x, y, pad=10):
        if self._in_rect(x, y, self.bar_rect, pad=pad):
            return True
        if self.active_panel == "COLOR" and self._in_rect(x, y, self.color_panel_rect, pad=pad):
            return True
        if self.active_panel == "SIZE" and self._in_rect(x, y, self.size_panel_rect, pad=pad):
            return True
        return False

    def close_panels(self):
        self.active_panel = None

    def reset_interaction(self):
        self.hovered_item = None
        self._was_clicking = False

    def update_cursor(self, cx, cy, is_clicking):
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
                if math.hypot(cx - (918 + idx * 62), cy - 345) <= 22:
                    self.hovered_item = f"PRESET_{preset}"
                    break

        if not is_clicking:
            return False

        if self.active_panel == "COLOR" and self._in_rect(cx, cy, self.color_panel_rect, pad=10):
            sx0, sy0 = self.sv_origin
            hx0, hy0 = self.hue_origin

            if (sx0 - 10) <= cx <= (sx0 + self.PICKER_SIZE + 10) and (sy0 - 10) <= cy <= (sy0 + self.PICKER_SIZE + 10):
                rel_x = max(0, min(self.PICKER_SIZE - 1, cx - sx0))
                rel_y = max(0, min(self.PICKER_SIZE - 1, cy - sy0))
                self.sat = int(round((rel_x / (self.PICKER_SIZE - 1)) * 255))
                self.val = int(round(255 - (rel_y / (self.PICKER_SIZE - 1)) * 230))
                self._update_bgr_from_hsv()
                return False

            if (hx0 - 10) <= cx <= (hx0 + self.HUE_BAR_W + 10) and (hy0 - 10) <= cy <= (hy0 + self.PICKER_SIZE + 10):
                rel_y = max(0, min(self.PICKER_SIZE - 1, cy - hy0))
                new_hue = int(round((rel_y / (self.PICKER_SIZE - 1)) * 179))
                if new_hue != self.hue:
                    self.hue = new_hue
                    self._rebuild_sv_square()
                    self._update_bgr_from_hsv()
            return False

        if self.active_panel == "SIZE" and self._in_rect(cx, cy, self.size_panel_rect, pad=10):
            if (self.slider_x1 - 12) <= cx <= (self.slider_x2 + 12) and (self.slider_y - 26) <= cy <= (self.slider_y + 26):
                ratio = max(0.0, min(1.0, (cx - self.slider_x1) / float(self.slider_x2 - self.slider_x1)))
                self.brush_thickness = int(round(4 + ratio * 32))
                return False

            if just_clicked:
                for idx, preset in enumerate(self.size_presets):
                    if math.hypot(cx - (918 + idx * 62), cy - 345) <= 24:
                        self.brush_thickness = preset
                        return False
            return False

        if just_clicked and (now - self._last_btn_click_t) >= 0.30:
            if self.hovered_item == "BTN_COLOR":
                self._last_btn_click_t = now
                self.active_panel = None if self.active_panel == "COLOR" else "COLOR"
            elif self.hovered_item == "BTN_SIZE":
                self._last_btn_click_t = now
                self.active_panel = None if self.active_panel == "SIZE" else "SIZE"
            elif self.hovered_item == "BTN_SNAP":
                self._last_btn_click_t = now
                self.shape_assist_active = not self.shape_assist_active
                self.active_panel = None
            elif self.hovered_item == "BTN_CLEAR":
                self._last_btn_click_t = now
                self._clear_flash_until = now + 0.32
                self.active_panel = None
                return True

        return False

    def draw(self, frame):
        now = time.perf_counter()

        self._draw_rounded_fill(
            frame, self.bar_rect, radius=22, color=(24, 26, 34), alpha=0.76, frosted=True, shadow=True
        )
        self._draw_rounded_border(frame, self.bar_rect, radius=22, color=(85, 92, 112), thickness=1)

        # Card 1: Cor
        c_act = self.active_panel == "COLOR"
        c_hov = self.hovered_item == "BTN_COLOR"
        self._draw_card(
            frame,
            self.btn_color_rect,
            "COR",
            (62, 68, 88) if c_act else (46, 50, 66) if c_hov else (34, 36, 46),
            (255, 255, 255) if c_act else (185, 195, 220) if c_hov else (68, 72, 90),
            c_act or c_hov,
        )
        ccx = (self.btn_color_rect[0] + self.btn_color_rect[2]) // 2
        ccy = self.btn_color_rect[1] + 36
        cv2.circle(frame, (ccx, ccy), 22, (235, 235, 245), 2, cv2.LINE_AA)
        cv2.circle(frame, (ccx, ccy), 18, self.brush_color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (ccx - 6, ccy - 6), 4, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

        # Card 2: Grossura do Traco
        s_act = self.active_panel == "SIZE"
        s_hov = self.hovered_item == "BTN_SIZE"
        self._draw_card(
            frame,
            self.btn_size_rect,
            f"{self.brush_thickness}px",
            (62, 68, 88) if s_act else (46, 50, 66) if s_hov else (34, 36, 46),
            (255, 255, 255) if s_act else (185, 195, 220) if s_hov else (68, 72, 90),
            s_act or s_hov,
        )
        scx = (self.btn_size_rect[0] + self.btn_size_rect[2]) // 2
        scy = self.btn_size_rect[1] + 36
        cv2.circle(frame, (scx, scy), 21, (90, 96, 115), 1, cv2.LINE_AA)
        pr = max(2, min(18, self.brush_thickness // 2))
        cv2.circle(frame, (scx, scy), pr, self.brush_color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (scx, scy), pr + 1, (240, 240, 245), 1, cv2.LINE_AA)

        # Card 3: Formas Inteligentes
        g_on = self.shape_assist_active
        g_hov = self.hovered_item == "BTN_SNAP"
        icon_g_col = (240, 255, 245) if g_on else (160, 168, 185)
        self._draw_card(
            frame,
            self.btn_snap_rect,
            "FORMAS" if g_on else "LIVRE",
            (42, 115, 62) if g_on else (46, 50, 66) if g_hov else (34, 36, 46),
            (170, 255, 190) if (g_on and g_hov) else (120, 220, 145) if g_on else (185, 195, 220) if g_hov else (68, 72, 90),
            g_on or g_hov,
            icon_g_col,
        )
        gcx = (self.btn_snap_rect[0] + self.btn_snap_rect[2]) // 2
        gcy = self.btn_snap_rect[1] + 34
        tri_pts = np.array([[gcx - 14, gcy + 11], [gcx - 2, gcy - 12], [gcx + 10, gcy + 11]], dtype=np.int32)
        cv2.polylines(frame, [tri_pts], True, icon_g_col, 2, cv2.LINE_AA)
        cv2.circle(frame, (gcx + 8, gcy - 3), 8, icon_g_col, 2, cv2.LINE_AA)

        # Card 4: Limpar Tudo
        cl_hov = self.hovered_item == "BTN_CLEAR"
        flashing = now < self._clear_flash_until
        self._draw_card(
            frame,
            self.btn_clear_rect,
            "LIMPAR",
            (90, 95, 235) if flashing else (52, 44, 92) if cl_hov else (38, 34, 54),
            (255, 255, 255) if flashing else (160, 150, 255) if cl_hov else (80, 68, 110),
            cl_hov or flashing,
        )
        icx = (self.btn_clear_rect[0] + self.btn_clear_rect[2]) // 2
        icy = self.btn_clear_rect[1] + 35
        icon_col = (255, 240, 240) if cl_hov else (190, 185, 255)
        cv2.line(frame, (icx - 13, icy - 10), (icx + 13, icy - 10), icon_col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (icx - 5, icy - 14), (icx + 5, icy - 10), icon_col, 1, cv2.LINE_AA)
        trash_body = np.array(
            [[icx - 10, icy - 8], [icx + 10, icy - 8], [icx + 8, icy + 13], [icx - 8, icy + 13]],
            dtype=np.int32,
        )
        cv2.polylines(frame, [trash_body], True, icon_col, 2, cv2.LINE_AA)
        cv2.line(frame, (icx - 3, icy - 3), (icx - 3, icy + 8), icon_col, 1, cv2.LINE_AA)
        cv2.line(frame, (icx + 3, icy - 3), (icx + 3, icy + 8), icon_col, 1, cv2.LINE_AA)

        # Popup de Cor
        if self.active_panel == "COLOR":
            px1, py1, px2, _ = self.color_panel_rect
            self._draw_rounded_fill(
                frame, self.color_panel_rect, radius=20, color=(22, 24, 32), alpha=0.86, frosted=True, shadow=True
            )
            self._draw_rounded_border(frame, self.color_panel_rect, radius=20, color=(105, 112, 135), thickness=1)

            cv2.putText(frame, "PALETA DE COR", (px1 + 18, py1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 235, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, self.hex_color, (px2 - 135, py1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 178, 195), 1, cv2.LINE_AA)

            pill_rect = (px2 - 64, py1 + 14, px2 - 18, py1 + 42)
            self._draw_rounded_fill(frame, pill_rect, radius=10, color=self.brush_color, alpha=1.0)
            self._draw_rounded_border(frame, pill_rect, radius=10, color=(235, 235, 245), thickness=1)

            sx0, sy0 = self.sv_origin
            frame[sy0 : sy0 + self.PICKER_SIZE, sx0 : sx0 + self.PICKER_SIZE] = self._sv_bgr_cache
            cv2.rectangle(frame, (sx0, sy0), (sx0 + self.PICKER_SIZE, sy0 + self.PICKER_SIZE), (140, 145, 165), 1, cv2.LINE_AA)

            sel_x = sx0 + int((self.sat / 255.0) * (self.PICKER_SIZE - 1))
            sel_y = max(sy0, min(sy0 + self.PICKER_SIZE - 1, sy0 + int(((255 - self.val) / 230.0) * (self.PICKER_SIZE - 1))))
            cv2.circle(frame, (sel_x, sel_y), 8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (sel_x, sel_y), 7, (255, 255, 255), 2, cv2.LINE_AA)

            hx0, hy0 = self.hue_origin
            frame[hy0 : hy0 + self.PICKER_SIZE, hx0 : hx0 + self.HUE_BAR_W] = self._hue_bar_bgr
            cv2.rectangle(frame, (hx0, hy0), (hx0 + self.HUE_BAR_W, hy0 + self.PICKER_SIZE), (140, 145, 165), 1, cv2.LINE_AA)

            hue_y = hy0 + int((self.hue / 179.0) * (self.PICKER_SIZE - 1))
            cv2.rectangle(frame, (hx0 - 3, hue_y - 4), (hx0 + self.HUE_BAR_W + 3, hue_y + 4), (255, 255, 255), 2, cv2.LINE_AA)

        # Popup de Grossura
        elif self.active_panel == "SIZE":
            px1, py1, px2, _ = self.size_panel_rect
            self._draw_rounded_fill(
                frame, self.size_panel_rect, radius=20, color=(22, 24, 32), alpha=0.86, frosted=True, shadow=True
            )
            self._draw_rounded_border(frame, self.size_panel_rect, radius=20, color=(105, 112, 135), thickness=1)

            cv2.putText(frame, "GROSSURA DO TRACO", (px1 + 18, py1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{self.brush_thickness} px", (px2 - 68, py1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.line(frame, (self.slider_x1, self.slider_y), (self.slider_x2, self.slider_y), (70, 76, 94), 8, cv2.LINE_AA)
            knob_x = int(self.slider_x1 + ((self.brush_thickness - 4) / 32.0) * (self.slider_x2 - self.slider_x1))
            cv2.line(frame, (self.slider_x1, self.slider_y), (knob_x, self.slider_y), self.brush_color, 8, cv2.LINE_AA)
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


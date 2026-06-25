import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import time
import math
from hand_tracking_module import HandDetector
from shape_recognition_module import (
    apply_stroke_correction,
    shape_display_name,
)
from letter_recognition_module import recognize_letter

# Camera capture size (lower = faster tracking). Display stays at W x H.
CAM_W, CAM_H = 640, 480
SMOOTHING = 0.82  # Higher = finger follows faster (less lag)
WINDOW_NAME = "AI Air Drawing System"
W, H = 1280, 720
HEADER_H = 158
FOOTER_H = 50
CANVAS_MARGIN = 6
CANVAS_TOP = 160
CANVAS_BOTTOM = 670
P_COLORS = (10, 10, 400, 148)
P_TOOLS = (410, 10, 620, 148)
P_BRUSH = (630, 10, 900, 148)
P_HELP = (910, 10, 1270, 148)


def get_work_area_size():
    """Usable screen size (excludes Windows taskbar)."""
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0)
        return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1) - 48
        except Exception:
            return 1280, 720


def setup_layout(screen_w, screen_h):
    """Build proportional UI layout that fills the display."""
    global W, H, HEADER_H, FOOTER_H, CANVAS_MARGIN, CANVAS_TOP, CANVAS_BOTTOM
    global P_COLORS, P_TOOLS, P_BRUSH, P_HELP

    W = max(1024, screen_w)
    H = max(600, screen_h)

    FOOTER_H = max(40, int(H * 0.062))
    HEADER_H = max(128, int(H * 0.152))
    CANVAS_MARGIN = max(4, int(W * 0.005))
    CANVAS_TOP = HEADER_H + 2
    CANVAS_BOTTOM = H - FOOTER_H - 2

    margin = max(10, int(W * 0.012))
    gap = max(8, int(W * 0.007))
    py1, py2 = 8, HEADER_H - 6
    usable = W - 2 * margin - 3 * gap

    w_colors = int(usable * 0.37)
    w_tools = int(usable * 0.19)
    w_brush = int(usable * 0.27)
    w_help = usable - w_colors - w_tools - w_brush

    x = margin
    P_COLORS = (x, py1, x + w_colors, py2)
    x += w_colors + gap
    P_TOOLS = (x, py1, x + w_tools, py2)
    x += w_tools + gap
    P_BRUSH = (x, py1, x + w_brush, py2)
    x += w_brush + gap
    P_HELP = (x, py1, W - margin, py2)

# ── Theme (BGR) ─────────────────────────────────────────────────────────────
C_BLACK = (0, 0, 0)
C_PANEL = (22, 22, 28)
C_PURPLE = (235, 95, 215)
C_PURPLE_DIM = (160, 60, 140)
C_BORDER = (210, 75, 190)
C_TEXT = (245, 245, 250)
C_TEXT_DIM = (175, 175, 185)
C_GREEN = (90, 230, 90)
C_BLUE_DOT = (255, 180, 80)
C_YELLOW = (0, 230, 255)
C_RED_KEY = (90, 90, 255)
C_YELLOW_KEY = (0, 230, 255)

# 6 top row + 6 bottom row (index 0-5 top, 6-11 bottom)
PALETTE = [
    (0, 0, 255),       # 0  Red
    (0, 140, 255),     # 1  Orange
    (0, 255, 255),     # 2  Yellow
    (0, 255, 0),       # 3  Green
    (255, 255, 0),     # 4  Cyan
    (255, 0, 0),       # 5  Blue
    (255, 0, 200),     # 6  Purple
    (180, 105, 255),   # 7  Pink
    (255, 255, 255),   # 8  White
    (0, 255, 128),     # 9  Highlighter lime (distinct from yellow & green)
    (42, 82, 165),     # 10 Brown
    (255, 128, 0),     # 11 Rainbow accent
]

SWATCH_COLS = 6
SWATCH_R = 14
SWATCH_HIT_EXTRA = 18


def pen_color_from_index(index):
    """Actual BGR color used when drawing."""
    return PALETTE[index]


def swatch_display_color(index):
    """Color shown in the toolbar swatch."""
    return PALETTE[index]


def _finish_stroke(img_canvas, prev_canvas, stroke, color, thickness, shape_enabled):
    """Recognize letter first; shape correction only if no letter found."""
    letter, letter_conf = recognize_letter(stroke, canvas_bgr=img_canvas)

    if letter:
        return img_canvas, f"letter:{letter}", letter_conf

    corrected, shape_type, conf = apply_stroke_correction(
        img_canvas, prev_canvas, stroke, color, thickness, enabled=shape_enabled
    )
    return corrected, shape_type, conf


# ── UI helpers ──────────────────────────────────────────────────────────────
def _rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def _label(img, text, pos, scale=0.45, color=C_TEXT, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _panel_content_band(rect, title_offset=36):
    """Vertical band below panel title for centered controls."""
    x1, y1, x2, y2 = rect
    top = y1 + title_offset
    return top, y2 - 8, x2 - x1


def _swatch_centers():
    """6 colors per row, evenly spaced and aligned."""
    x1, y1, x2, y2 = P_COLORS
    pad = max(14, int((x2 - x1) * 0.05))
    inner_w = x2 - x1 - 2 * pad
    top, bottom, _ = _panel_content_band(P_COLORS)
    mid_y = (top + bottom) // 2
    row1_y = mid_y - 22
    row2_y = mid_y + 22
    step = inner_w / (SWATCH_COLS - 1)
    row1 = [(int(x1 + pad + i * step), row1_y) for i in range(SWATCH_COLS)]
    row2 = [(int(x1 + pad + i * step), row2_y) for i in range(SWATCH_COLS)]
    return row1 + row2


def _tool_button_rects():
    x1, y1, x2, y2 = P_TOOLS
    pad = max(6, int((x2 - x1) * 0.04))
    inner_w = x2 - x1 - 2 * pad
    btn_w = inner_w // 4
    top, bottom, _ = _panel_content_band(P_TOOLS, title_offset=8)
    return [
        (x1 + pad + i * btn_w, top, x1 + pad + i * btn_w + btn_w - 3, bottom)
        for i in range(4)
    ]


def _draw_rainbow_swatch(img, cx, cy, r):
    hues = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    for i, col in enumerate(hues):
        start = int(i * 360 / len(hues))
        end = int((i + 1) * 360 / len(hues))
        cv2.ellipse(img, (cx, cy), (r, r), 0, start, end, col, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, C_BORDER, 1, cv2.LINE_AA)


def _draw_pen_icon(img, cx, cy, color):
    cv2.line(img, (cx - 8, cy + 8), (cx + 10, cy - 10), color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx + 10, cy - 10), 3, color, -1, cv2.LINE_AA)


def _draw_eraser_icon(img, cx, cy, color):
    pts = np.array([[cx - 10, cy + 4], [cx + 10, cy + 4], [cx + 8, cy - 8], [cx - 8, cy - 8]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)


def _draw_clear_icon(img, cx, cy, color):
    cv2.rectangle(img, (cx - 7, cy - 2), (cx + 7, cy + 10), color, 2, cv2.LINE_AA)
    cv2.line(img, (cx - 9, cy - 2), (cx + 9, cy - 2), color, 2, cv2.LINE_AA)
    cv2.line(img, (cx - 3, cy - 8), (cx + 3, cy - 8), color, 2, cv2.LINE_AA)


def _draw_save_icon(img, cx, cy, color):
    cv2.rectangle(img, (cx - 9, cy - 8), (cx + 9, cy + 10), color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (cx - 4, cy - 8), (cx + 4, cy - 2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - 6, cy + 2), (cx + 6, cy + 8), (30, 30, 35), -1, cv2.LINE_AA)


def _draw_panel(img, rect, title=None):
    x1, y1, x2, y2 = rect
    _rounded_rect(img, (x1, y1), (x2, y2), C_PANEL, -1, 6)
    cv2.rectangle(img, (x1, y1), (x2, y2), C_PURPLE_DIM, 1)
    if title:
        _label(img, title, (x1 + 12, y1 + 22), 0.52, C_PURPLE, 1)


def draw_ui(img, pen_color, brush_thickness, tool_mode, color_index, status_text, shapes_on=True, letter_model_ready=False):
    """Draw reference-style UI on solid black chrome regions."""
    # Solid black header & footer (camera must not show through)
    img[:HEADER_H] = C_BLACK
    img[H - FOOTER_H :] = C_BLACK
    cv2.line(img, (0, HEADER_H), (W, HEADER_H), C_BORDER, 1)

    # Canvas purple frame
    cv2.rectangle(
        img,
        (CANVAS_MARGIN, CANVAS_TOP),
        (W - CANVAS_MARGIN, CANVAS_BOTTOM),
        C_BORDER,
        2,
    )

    # ── COLORS ──
    _draw_panel(img, P_COLORS, "COLORS")
    for i, (cx, cy) in enumerate(_swatch_centers()):
        if i == 11:
            _draw_rainbow_swatch(img, cx, cy, SWATCH_R)
        else:
            cv2.circle(img, (cx, cy), SWATCH_R, swatch_display_color(i), -1, cv2.LINE_AA)
            if i == 8:
                cv2.circle(img, (cx, cy), SWATCH_R, C_TEXT_DIM, 1, cv2.LINE_AA)
        ring = (255, 255, 255) if i != 8 else (0, 0, 0)
        if i == color_index:
            cv2.circle(img, (cx, cy), SWATCH_R + 5, ring, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), SWATCH_R + 7, C_PURPLE, 2, cv2.LINE_AA)

    # ── Tools (horizontal row of tall buttons) ──
    _draw_panel(img, P_TOOLS)
    tools = [
        ("PEN", "pen", _draw_pen_icon),
        ("ERASER", "eraser", _draw_eraser_icon),
        ("CLEAR", "clear", _draw_clear_icon),
        ("SAVE", "save", _draw_save_icon),
    ]
    for (name, key, icon_fn), (bx1, by1, bx2, by2) in zip(tools, _tool_button_rects()):
        active = tool_mode == key if key in ("pen", "eraser") else False
        fill = (48, 32, 52) if active else (30, 30, 36)
        _rounded_rect(img, (bx1, by1), (bx2, by2), fill, -1, 5)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), C_PURPLE if active else C_PURPLE_DIM, 2 if active else 1)
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        icon_fn(img, cx, cy - 8, C_TEXT)
        tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        _label(img, name, (cx - tw // 2, by2 - 10), 0.38, C_TEXT)

    # ── BRUSH SIZE ──
    bx1, by1, bx2, by2 = P_BRUSH
    _draw_panel(img, P_BRUSH, "BRUSH SIZE")
    mid_y = _brush_mid_y()
    btn_off = min(52, (bx2 - bx1) // 5)
    minus_cx, plus_cx = bx1 + btn_off, bx2 - btn_off
    center_cx = (bx1 + bx2) // 2
    for cx, sym in [(minus_cx, "-"), (plus_cx, "+")]:
        cv2.circle(img, (cx, mid_y), 20, (38, 34, 44), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, mid_y), 20, C_PURPLE_DIM, 1, cv2.LINE_AA)
        _label(img, sym, (cx - 5, mid_y + 7), 0.75, C_TEXT, 2)
    cv2.circle(img, (center_cx, mid_y), 24, (35, 30, 42), -1, cv2.LINE_AA)
    cv2.circle(img, (center_cx, mid_y), 24, C_PURPLE, 2, cv2.LINE_AA)
    sz_txt = str(brush_thickness)
    tw = cv2.getTextSize(sz_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    _label(img, sz_txt, (center_cx - tw // 2, mid_y + 8), 0.6, C_TEXT, 2)
    track_y = by2 - max(22, int((by2 - by1) * 0.18))
    track_x1, track_x2 = bx1 + 18, bx2 - 18
    cv2.line(img, (track_x1, track_y), (track_x2, track_y), (55, 50, 62), 5, cv2.LINE_AA)
    t = (brush_thickness - 2) / 98.0
    thumb_x = int(track_x1 + t * (track_x2 - track_x1))
    cv2.circle(img, (thumb_x, track_y), 9, C_PURPLE, -1, cv2.LINE_AA)

    # ── HOW TO USE (compact panel, vertically centered) ──
    hx1, hy1, hx2, hy2 = P_HELP
    _draw_panel(img, P_HELP, "HOW TO USE")
    letter_status = "Letters: ON" if letter_model_ready else "Letters: NO MODEL"
    hints = [
        (C_GREEN, "Index finger: Draw"),
        (C_BLUE_DOT, "2 Fingers: Select UI"),
        (C_YELLOW, "Auto-Shape: ON" if shapes_on else "Auto-Shape: OFF"),
        (C_TEXT if letter_model_ready else (0, 0, 255), letter_status),
    ]
    line_h = 26
    block_h = len(hints) * line_h
    top, bottom, _ = _panel_content_band(P_HELP)
    start_y = (top + bottom - block_h) // 2 + 8
    for i, (dot_col, text) in enumerate(hints):
        y = start_y + i * line_h
        cv2.circle(img, (hx1 + 12, y - 4), 4, dot_col, -1, cv2.LINE_AA)
        _label(img, text, (hx1 + 22, y), 0.4, C_TEXT_DIM)

    # ── Status bar (always visible) ──
    foot_y = H - FOOTER_H
    cv2.line(img, (0, foot_y), (W, foot_y), C_PURPLE_DIM, 1)
    dot_y = foot_y + FOOTER_H // 2
    dot_col = C_GREEN if status_text in ("READY", "TRACKING") else C_YELLOW
    cv2.circle(img, (22, dot_y), 6, dot_col, -1, cv2.LINE_AA)
    cv2.circle(img, (22, dot_y), 8, dot_col, 1, cv2.LINE_AA)
    text_y = foot_y + FOOTER_H - 14
    _label(img, f"STATUS: {status_text}", (38, text_y), 0.5, C_TEXT_DIM)

    parts = [
        ("PRESS ", C_TEXT_DIM),
        ("'S'", C_YELLOW_KEY),
        (" TO SAVE  |  PRESS ", C_TEXT_DIM),
        ("'Q'", C_RED_KEY),
        (" TO QUIT", C_TEXT_DIM),
    ]
    total_w = sum(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for t, _ in parts)
    x = W - total_w - 18
    for text, col in parts:
        cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]


def compose_frame(camera_feed, img_canvas, frame_buf):
    """Merge camera + drawing inside canvas only (fast path)."""
    y1, y2 = CANVAS_TOP, CANVAS_BOTTOM
    x1, x2 = CANVAS_MARGIN, W - CANVAS_MARGIN

    cam_roi = camera_feed[y1:y2, x1:x2]
    canvas_roi = img_canvas[y1:y2, x1:x2]

    gray = cv2.cvtColor(canvas_roi, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    merged = cv2.bitwise_or(cv2.bitwise_and(cam_roi, inv_bgr), canvas_roi)
    frame_buf[y1:y2, x1:x2] = merged
    return frame_buf


def _in_canvas(y):
    return CANVAS_TOP <= y <= CANVAS_BOTTOM


def _hit_color(x, y):
    px1, py1, px2, py2 = P_COLORS
    if y > HEADER_H or not (px1 <= x <= px2 and py1 <= y <= py2):
        return -1
    best_i, best_d = -1, 1e9
    hit_r = SWATCH_R + SWATCH_HIT_EXTRA
    for i, (cx, cy) in enumerate(_swatch_centers()):
        d = math.hypot(x - cx, y - cy)
        if d <= hit_r and d < best_d:
            best_i, best_d = i, d
    return best_i


def _hit_tool(x, y):
    if y > HEADER_H:
        return None
    names = ["pen", "eraser", "clear", "save"]
    for i, (bx1, by1, bx2, by2) in enumerate(_tool_button_rects()):
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return names[i]
    return None


def _brush_mid_y():
    top, bottom, _ = _panel_content_band(P_BRUSH)
    return (top + bottom) // 2 + 2


def _hit_brush_btn(x, y):
    bx1, by1, bx2, by2 = P_BRUSH
    if not (bx1 <= x <= bx2 and by1 <= y <= by2):
        return None
    mid_y = _brush_mid_y()
    btn_off = min(52, (bx2 - bx1) // 5)
    if math.hypot(x - (bx1 + btn_off), y - mid_y) <= 24:
        return "minus"
    if math.hypot(x - (bx2 - btn_off), y - mid_y) <= 24:
        return "plus"
    return None


def handle_ui_click(x, y, state):
    """Process toolbar hit at (x, y). Returns updated state dict or None."""
    color_hit = _hit_color(x, y)
    if color_hit >= 0:
        state["color_index"] = color_hit
        state["pen_color"] = pen_color_from_index(color_hit)
        state["tool_mode"] = "pen"
        state["status"] = "READY"
        return state

    tool = _hit_tool(x, y)
    if tool == "pen":
        state["tool_mode"] = "pen"
        state["status"] = "READY"
    elif tool == "eraser":
        state["tool_mode"] = "eraser"
        state["status"] = "READY"
    elif tool == "clear":
        state["img_canvas"] = np.zeros((H, W, 3), np.uint8)
        state["prev_canvas"] = None
        state["status"] = "CLEARED"
    elif tool == "save":
        if time.time() - state["last_saved"] > 1.5:
            cv2.imwrite("drawing.png", state["img_canvas"])
            state["last_saved"] = time.time()
            state["status"] = "SAVED"
            print("Drawing saved as drawing.png!")
        return state

    brush_btn = _hit_brush_btn(x, y)
    if brush_btn == "minus" and time.time() - state["last_resize"] > 0.12:
        state["brush_thickness"] = max(2, state["brush_thickness"] - 2)
        state["last_resize"] = time.time()
    elif brush_btn == "plus" and time.time() - state["last_resize"] > 0.12:
        state["brush_thickness"] = min(100, state["brush_thickness"] + 2)
        state["last_resize"] = time.time()

    return state


def draw_cursor(img, x, y, drawing):
    """White ring cursor for drawing mode."""
    if not _in_canvas(y):
        return
    col = (255, 255, 255)
    cv2.circle(img, (x, y), 12, col, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, col, -1, cv2.LINE_AA)


def draw_selection_cursor(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1 - 22), (x2, y2 + 22), color, cv2.FILLED)
    cv2.rectangle(img, (x1, y1 - 22), (x2, y2 + 22), (255, 255, 255), 1)


def main():
    screen_w, screen_h = get_work_area_size()
    setup_layout(screen_w, screen_h)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = HandDetector(min_detection_confidence=0.6)
    detector.set_target_size(W, H)

    pen_color = (0, 0, 255)
    color_index = 0
    tool_mode = "pen"
    brush_thickness = 15
    eraser_thickness = 50

    xp, yp = 0, 0
    smooth_x, smooth_y = 0, 0

    img_canvas = np.zeros((H, W, 3), np.uint8)
    current_stroke = []
    is_drawing = False
    prev_canvas = None

    last_saved = 0
    last_resize = 0
    last_status_change = 0
    status_text = "READY"
    hand_detected = False
    shape_auto_correct = False
    shape_feedback = None
    shape_feedback_until = 0
    gesture_hint = ""
    display_frame = np.zeros((H, W, 3), np.uint8)
    letter_model_ready: bool = False
    try:
        from letter_recognition_module import model_ready
        letter_model_ready = model_ready()
    except Exception:
        letter_model_ready = False

    # WINDOW_GUI_NORMAL hides the OpenCV Qt toolbar (zoom/save/pan buttons).
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, W, H)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        # Track on native camera size (fast), then upscale for display.
        detector.find_hands(img, draw=False)
        lm_list = detector.find_position()
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        draw_color = (0, 0, 0) if tool_mode == "eraser" else pen_color
        line_thick = eraser_thickness if tool_mode == "eraser" else brush_thickness
        cursor_draw = None
        cursor_select = None

        hand_detected = len(lm_list) != 0

        if hand_detected:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            drawing_gesture = detector.is_drawing_pose()
            selection_gesture = detector.is_selection_pose()

            if drawing_gesture and _in_canvas(y1):
                gesture_hint = "DRAWING"
                cursor_draw = (x1, y1)

                if not is_drawing:
                    is_drawing = True
                    prev_canvas = img_canvas.copy()
                    current_stroke = []

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    smooth_x, smooth_y = x1, y1

                smooth_x = int(x1 * SMOOTHING + smooth_x * (1 - SMOOTHING))
                smooth_y = int(y1 * SMOOTHING + smooth_y * (1 - SMOOTHING))

                cv2.line(img_canvas, (xp, yp), (smooth_x, smooth_y), draw_color, line_thick)
                if tool_mode == "pen":
                    current_stroke.append((smooth_x, smooth_y))

                xp, yp = smooth_x, smooth_y

            else:
                # Drawing gesture ended (or hand moved out of canvas)
                if is_drawing:
                    is_drawing = False
                    if tool_mode == "pen":
                        img_canvas, detected, _conf = _finish_stroke(
                            img_canvas,
                            prev_canvas,
                            current_stroke,
                            pen_color,
                            brush_thickness,
                            shape_auto_correct,
                        )
                        # Always update feedback
                        if detected:
                            if isinstance(detected, str) and detected.startswith("letter:"):
                                letter = detected.split(":", 1)[1]
                                shape_feedback = f"Letter: {letter} ({int(_conf * 100)}%)"
                            else:
                                shape_feedback = f"Shape: {shape_display_name(detected)} ({int(_conf * 100)}%)"
                            shape_feedback_until = time.time() + 2.0
                        else:
                            shape_feedback = "AI: Not confident"
                            shape_feedback_until = time.time() + 1.5
                    current_stroke = []

                xp, yp = 0, 0
                smooth_x, smooth_y = 0, 0


                if not _in_canvas(y1):
                    gesture_hint = "Move hand to canvas"
                elif selection_gesture:
                    gesture_hint = "SELECT"
                elif hand_detected:
                    gesture_hint = "Raise index only to draw"

                if selection_gesture:
                    ux, uy = (x1 + x2) // 2, (y1 + y2) // 2
                    if uy < HEADER_H:
                        ui_state = {
                            "pen_color": pen_color,
                            "color_index": color_index,
                            "tool_mode": tool_mode,
                            "brush_thickness": brush_thickness,
                            "img_canvas": img_canvas,
                            "prev_canvas": prev_canvas,
                            "last_saved": last_saved,
                            "last_resize": last_resize,
                            "status": status_text,
                        }
                        updated = handle_ui_click(ux, uy, ui_state)
                        if updated:
                            pen_color = updated["pen_color"]
                            color_index = updated["color_index"]
                            tool_mode = updated["tool_mode"]
                            brush_thickness = updated["brush_thickness"]
                            img_canvas = updated["img_canvas"]
                            prev_canvas = updated["prev_canvas"]
                            last_saved = updated["last_saved"]
                            last_resize = updated["last_resize"]
                            status_text = updated["status"]
                            if status_text in ("SAVED", "CLEARED"):
                                last_status_change = time.time()

                    sel_color = pen_color if tool_mode == "pen" else (200, 200, 200)
                    cursor_select = (x1, y1, x2, y2, sel_color)

        else:
            gesture_hint = "Show hand to camera"
            if is_drawing:
                is_drawing = False
                if tool_mode == "pen":
                    img_canvas, detected, _conf = _finish_stroke(
                        img_canvas,
                        prev_canvas,
                        current_stroke,
                        pen_color,
                        brush_thickness,
                        shape_auto_correct,
                    )
                    if detected:
                        if isinstance(detected, str) and detected.startswith("letter:"):
                            letter = detected.split(":", 1)[1]
                            shape_feedback = f"Letter: {letter} ({int(_conf * 100)}%)"
                        else:
                            shape_feedback = f"Shape: {shape_display_name(detected)}"
                        shape_feedback_until = time.time() + 2.0
                current_stroke = []
            xp, yp = 0, 0
            smooth_x, smooth_y = 0, 0

        display_frame.fill(0)
        compose_frame(img, img_canvas, display_frame)

        if hand_detected and not is_drawing:
            detector.draw_hand_overlay(display_frame, CANVAS_TOP, CANVAS_BOTTOM, CANVAS_MARGIN)

        if time.time() - last_saved < 1.2:
            status_text = "SAVED"
        elif status_text in ("SAVED", "CLEARED") and time.time() - last_status_change > 1.5:
            status_text = "TRACKING" if hand_detected else "READY"
        elif hand_detected and status_text == "READY":
            status_text = "TRACKING"
        elif not hand_detected and status_text == "TRACKING":
            status_text = "READY"

        draw_ui(
            display_frame, pen_color, brush_thickness, tool_mode, color_index, status_text, shape_auto_correct, letter_model_ready
        )

        if gesture_hint and hand_detected:
            hint_y = CANVAS_TOP + 12
            _rounded_rect(display_frame, (W // 2 - 160, hint_y - 18), (W // 2 + 160, hint_y + 16), (25, 25, 30), -1, 6)
            _label(display_frame, gesture_hint, (W // 2 - 150, hint_y + 4), 0.55, C_YELLOW, 1)

        if shape_feedback and time.time() < shape_feedback_until:
            cy = CANVAS_TOP + 48
            msg = f"AI: {shape_feedback}"
            tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            _rounded_rect(display_frame, (W // 2 - tw // 2 - 16, cy - 22), (W // 2 + tw // 2 + 16, cy + 14), (30, 30, 35), -1, 8)
            cv2.rectangle(
                display_frame,
                (W // 2 - tw // 2 - 16, cy - 22),
                (W // 2 + tw // 2 + 16, cy + 14),
                C_PURPLE,
                1,
            )
            _label(display_frame, msg, (W // 2 - tw // 2, cy + 6), 0.7, C_GREEN, 2)
        elif shape_feedback and time.time() >= shape_feedback_until:
            shape_feedback = None

        if cursor_draw:
            draw_cursor(display_frame, cursor_draw[0], cursor_draw[1], True)
        if cursor_select:
            draw_selection_cursor(display_frame, *cursor_select)

        if time.time() - last_saved < 1.0:
            cy = (CANVAS_TOP + CANVAS_BOTTOM) // 2
            _rounded_rect(display_frame, (W // 2 - 120, cy - 40), (W // 2 + 120, cy + 40), (30, 30, 35), -1, 12)
            cv2.rectangle(display_frame, (W // 2 - 120, cy - 40), (W // 2 + 120, cy + 40), C_PURPLE, 2)
            _label(display_frame, "SAVED!", (W // 2 - 55, cy + 10), 1.2, C_GREEN, 2)

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("drawing.png", img_canvas)
            last_saved = time.time()
            last_status_change = time.time()
            status_text = "SAVED"
            print("Drawing saved as drawing.png!")
        if key == ord("a"):
            shape_auto_correct = not shape_auto_correct
            shape_feedback = "ON" if shape_auto_correct else "OFF"
            shape_feedback_until = time.time() + 1.0
            print(f"Auto-shape correction: {'ON' if shape_auto_correct else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

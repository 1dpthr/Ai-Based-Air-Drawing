"""
Uppercase letter recognition from air-drawn strokes (EMNIST CNN).
"""

import os

import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ABSL_LOGGING_VERBOSITY"] = "0"
warnings.filterwarnings("ignore")

import cv2
import numpy as np

MODEL_PATH = "emnist_uppercase_model.h5"
CLASS_NAMES = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
MIN_STROKE_POINTS = 5
MIN_CONFIDENCE = 0.15
MIN_STROKE_SIZE = 14

_model = None
_model_available = None


def _load_model():
    global _model, _model_available
    if _model_available is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        _model_available = False
        return None

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH)
        _model_available = True
    except Exception as exc:
        print(f"Letter model load failed: {exc}")
        _model_available = False
        _model = None
    return _model


def model_ready():
    return _load_model() is not None


def to_emnist_style(gray: np.ndarray) -> np.ndarray:
    """
    EMNIST uses white background + dark letter.
    Air canvas uses dark background + bright stroke — invert to match training.
    """
    gray = gray.astype(np.uint8)
    if gray.max() == 0:
        return gray

    # Bright strokes on dark canvas -> invert
    if np.mean(gray) < 127:
        gray = 255 - gray

    # Normalize contrast: letter dark, background white
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = 255 - binary
    return binary


def _fit_to_28x28(gray: np.ndarray, margin: int = 4) -> np.ndarray:
    gray = to_emnist_style(gray)
    h, w = gray.shape[:2]
    if h < 1 or w < 1:
        return np.zeros((28, 28), dtype=np.uint8)

    scale = (28 - 2 * margin) / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((28, 28), 255, dtype=np.uint8)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def stroke_from_canvas(canvas_bgr, points, pad=32):
    if canvas_bgr is None or len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h_img, w_img = canvas_bgr.shape[:2]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w_img, max(xs) + pad)
    y2 = min(h_img, max(ys) + pad)

    if x2 - x1 < MIN_STROKE_SIZE or y2 - y1 < MIN_STROKE_SIZE:
        return None

    roi = canvas_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return _fit_to_28x28(binary)


def stroke_to_emnist_image(points, size=28, margin=4, thickness=3):
    if len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(max_x - min_x, 1)
    h = max(max_y - min_y, 1)

    if w < MIN_STROKE_SIZE and h < MIN_STROKE_SIZE:
        return None

    img = np.zeros((size, size), dtype=np.uint8)
    scale = (size - 2 * margin) / max(w, h)

    def to_px(px, py):
        x = int((px - min_x) * scale + margin)
        y = int((py - min_y) * scale + margin)
        return int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))

    for i in range(1, len(points)):
        p1 = to_px(points[i - 1][0], points[i - 1][1])
        p2 = to_px(points[i][0], points[i][1])
        cv2.line(img, p1, p2, 255, thickness, cv2.LINE_AA)

    return _fit_to_28x28(img)


def _variants(img: np.ndarray) -> list[np.ndarray]:
    """Small shifts/scales to improve robustness (test-time augmentation)."""
    if img is None or img.max() == 0:
        return []

    out = [img]
    for margin in (3, 5):
        h, w = img.shape
        scale = (28 - 2 * margin) / max(h, w)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((28, 28), 255, dtype=np.uint8)
        y0 = (28 - nh) // 2
        x0 = (28 - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        out.append(canvas)

    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.erode(img, kernel, iterations=1)
    out.append(thick)
    return out


def recognize_letter(points, canvas_bgr=None):
    """Predict A–Z from stroke. Returns (letter_or_None, confidence)."""
    if len(points) < MIN_STROKE_POINTS:
        return None, 0.0

    model = _load_model()
    if model is None:
        return None, 0.0

    base_images = []
    canvas_img = stroke_from_canvas(canvas_bgr, points) if canvas_bgr is not None else None
    points_img = stroke_to_emnist_image(points)

    if canvas_img is not None and canvas_img.max() > 0:
        base_images.append(canvas_img)
    if points_img is not None and points_img.max() > 0:
        base_images.append(points_img)

    if not base_images:
        return None, 0.0

    all_variants = []
    for img in base_images:
        all_variants.extend(_variants(img))

    batch = np.stack([(v.astype(np.float32) / 255.0) for v in all_variants], axis=0)
    batch = batch.reshape(-1, 28, 28, 1)

    probs = model.predict(batch, verbose=0)
    mean_probs = np.mean(probs, axis=0)
    idx = int(np.argmax(mean_probs))
    conf = float(mean_probs[idx])

    if conf < MIN_CONFIDENCE:
        return None, conf
    return CLASS_NAMES[idx], conf

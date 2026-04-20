"""Nose‑ring try‑on engine.

The function ``tryon_nosering`` mirrors the logic from the original Colab
notebook but is expressed as a callable that works with file paths.
"""

import cv2
import numpy as np
import math
from .mp_adapter import create_face_mesh

_face_mesh = create_face_mesh()


def _overlay_pin(bg, overlay, x, y):
    oh, ow = overlay.shape[:2]
    x1, y1 = x - ow // 2, y - oh // 2
    x2, y2 = x1 + ow, y1 + oh
    bh, bw = bg.shape[:2]
    tx1, ty1 = max(0, x1), max(0, y1)
    tx2, ty2 = min(bw, x2), min(bh, y2)
    if tx1 >= tx2 or ty1 >= ty2:
        return bg
    overlay_crop = overlay[ty1 - y1 : ty2 - y1, tx1 - x1 : tx2 - x1]
    bg_crop = bg[ty1:ty2, tx1:tx2]
    alpha = overlay_crop[..., 3] / 255.0
    for c in range(3):
        bg_crop[..., c] = alpha * overlay_crop[..., c] + (1 - alpha) * bg_crop[..., c]
    bg[ty1:ty2, tx1:tx2] = bg_crop
    return bg


def tryon_nosering(face_path: str, pin_path: str, output_path: str) -> str:
    """Overlay a nose pin onto ``face_path``.

    ``pin_path`` must be a PNG with an alpha channel.
    """
    image = cv2.imread(face_path)
    if image is None:
        raise FileNotFoundError(face_path)
    pin = cv2.imread(pin_path, cv2.IMREAD_UNCHANGED)
    if pin is None:
        raise FileNotFoundError(pin_path)
    if pin.shape[2] != 4:
        raise ValueError("Pin image must have an alpha channel (PNG).")

    h, w = image.shape[:2]
    results = _face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected in the supplied image.")
    lm = results.multi_face_landmarks[0]

    # Landmark 134 corresponds to the left nostril wing.
    target = lm.landmark[134]
    x = int(target.x * w)
    y = int(target.y * h)

    # Scale based on eye distance for a natural size.
    left_eye = lm.landmark[133]
    right_eye = lm.landmark[362]
    eye_dist = (
        np.linalg.norm(
            np.array([left_eye.x, left_eye.y]) - np.array([right_eye.x, right_eye.y])
        )
        * w
    )
    size = int(eye_dist * 0.15)
    if size < 15:
        size = 15

    # Rotation based on head tilt (using landmarks 454 and 234).
    dx = lm.landmark[454].x - lm.landmark[234].x
    dy = lm.landmark[454].y - lm.landmark[234].y
    angle = math.degrees(math.atan2(dy, dx))

    ph, pw = pin.shape[:2]
    M = cv2.getRotationMatrix2D((pw // 2, ph // 2), angle, 1.0)
    pin_rot = cv2.warpAffine(pin, M, (pw, ph), flags=cv2.INTER_LINEAR)
    pin_resized = cv2.resize(pin_rot, (size, size), interpolation=cv2.INTER_AREA)

    output = _overlay_pin(image.copy(), pin_resized, x, y)
    cv2.imwrite(output_path, output)
    return output_path

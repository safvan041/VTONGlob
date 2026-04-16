"""Necklace try‑on engine.

Converted from the original Colab notebook into a reusable function.
"""

import cv2
import numpy as np
import math
from .mp_adapter import create_face_mesh

_face_mesh = create_face_mesh()


def _overlay_necklace(bg, overlay, x, y):
    """Blend ``overlay`` (PNG with alpha) onto ``bg``.

    ``x`` is the centre point, ``y`` is the top of the necklace.
    """
    oh, ow = overlay.shape[:2]
    x1, y1 = x - ow // 2, y
    x2, y2 = x1 + ow, y1 + oh
    bh, bw = bg.shape[:2]
    tx1, ty1 = max(0, x1), max(0, y1)
    tx2, ty2 = min(bw, x2), min(bh, y2)
    if tx1 >= tx2 or ty1 >= ty2:
        return bg
    overlay_crop = overlay[ty1 - y1 : ty2 - y1, tx1 - x1 : tx2 - x1]
    bg_crop = bg[ty1:ty2, tx1:tx2]
    overlay_rgb = overlay_crop[..., :3]
    alpha = overlay_crop[..., 3] / 255.0
    alpha = np.expand_dims(alpha, axis=-1)
    blended = overlay_rgb * alpha + bg_crop * (1 - alpha)
    bg[ty1:ty2, tx1:tx2] = blended.astype(np.uint8)
    return bg


def tryon_necklace(face_path: str, necklace_path: str, output_path: str) -> str:
    """Overlay a necklace onto a face image.

    ``necklace_path`` should be a PNG with transparency.
    """
    # Load images
    raw = cv2.imread(face_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(face_path)
    # Convert possible RGBA background to white RGB (as original notebook did)
    if raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        rgb = raw[:, :, :3]
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        factor = alpha[:, :, np.newaxis] / 255.0
        image = (rgb * factor + white_bg * (1 - factor)).astype(np.uint8)
    else:
        image = raw

    necklace = cv2.imread(necklace_path, cv2.IMREAD_UNCHANGED)
    if necklace is None:
        raise FileNotFoundError(necklace_path)

    h, w = image.shape[:2]
    # Detect facial landmarks
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected.")
    lm = results.multi_face_landmarks[0]

    # Anchor: chin (landmark 152)
    chin = lm.landmark[152]
    cx, cy = int(chin.x * w), int(chin.y * h)

    # Determine vertical offset based on distance nose‑to‑chin
    nose_tip = lm.landmark[4]
    nose_to_chin = abs(chin.y - nose_tip.y) * h
    neck_y = cy + int(nose_to_chin * 0.20)  # 20% below chin
    neck_x = cx + int(w * 0.01)

    # Scale necklace width to jaw width (landmarks 234 & 454)
    left_jaw = lm.landmark[234]
    right_jaw = lm.landmark[454]
    jaw_width_px = math.hypot(
        (right_jaw.x - left_jaw.x) * w, (right_jaw.y - left_jaw.y) * h
    )
    size_w = int(jaw_width_px * 1.5)
    aspect = necklace.shape[0] / necklace.shape[1]
    size_h = int(size_w * aspect)

    # Rotation to follow jaw tilt
    angle = math.degrees(math.atan2(right_jaw.y - left_jaw.y, right_jaw.x - left_jaw.x))

    nh, nw = necklace.shape[:2]
    M = cv2.getRotationMatrix2D((nw // 2, nh // 2), angle, 1.0)
    necklace_rot = cv2.warpAffine(necklace, M, (nw, nh), flags=cv2.INTER_LINEAR)
    necklace_resized = cv2.resize(
        necklace_rot, (size_w, size_h), interpolation=cv2.INTER_AREA
    )

    output = _overlay_necklace(image.copy(), necklace_resized, neck_x, neck_y)
    cv2.imwrite(output_path, output)
    return output_path

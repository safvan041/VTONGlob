"""Earring try‑on engine.

The original Colab script was rewritten as a pure function that can be called from
Django. It expects three file system paths:
    * ``face_path`` – path to the user's face image (any format readable by OpenCV)
    * ``earring_path`` – path to the earring PNG with an alpha channel
    * ``output_path`` – where the blended result will be saved (PNG)

The function returns the ``output_path`` for convenience.
"""

import cv2
import numpy as np
import math
from .mp_adapter import create_face_mesh

# Initialise MediaPipe FaceMesh once – this is cheap and thread‑safe for our use case.
_face_mesh = create_face_mesh()


def _overlay_earring_dual(bg, overlay, x, y, side):
    """Blend ``overlay`` onto ``bg`` at (x, y).

    ``side`` can be ``"left"`` or ``"right"`` – the right side is horizontally
    flipped so the earring mirrors correctly.
    """
    oh, ow = overlay.shape[:2]
    if side == "right":
        overlay = cv2.flip(overlay, 1)

    x1 = x - ow // 2
    y1 = y - int(oh * 0.05)
    x2, y2 = x1 + ow, y1 + oh
    bg_h, bg_w = bg.shape[:2]

    # Clip to background boundaries
    tx1, ty1 = max(0, x1), max(0, y1)
    tx2, ty2 = min(bg_w, x2), min(bg_h, y2)
    if tx1 >= tx2 or ty1 >= ty2:
        return bg

    overlay_crop = overlay[ty1 - y1 : ty2 - y1, tx1 - x1 : tx2 - x1]
    bg_crop = bg[ty1:ty2, tx1:tx2]

    alpha = overlay_crop[..., 3] / 255.0
    for c in range(3):
        bg_crop[..., c] = alpha * overlay_crop[..., c] + (1 - alpha) * bg_crop[..., c]
    bg[ty1:ty2, tx1:tx2] = bg_crop
    return bg


def tryon_earring(face_path: str, earring_path: str, output_path: str) -> str:
    """Apply an earring PNG onto both ears of ``face_path``.

    The implementation follows the original Colab notebook but removes any
    interactive ``files.upload`` calls and ``cv2_imshow`` statements.
    """
    # Load images – the earring is read with alpha channel.
    image = cv2.imread(face_path)
    if image is None:
        raise FileNotFoundError(f"Face image not found: {face_path}")
    earring = cv2.imread(earring_path, cv2.IMREAD_UNCHANGED)
    if earring is None:
        raise FileNotFoundError(f"Earring image not found: {earring_path}")

    h, w = image.shape[:2]
    results = _face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Determine ear anchor points – fall back to centre if detection fails.
    ear_positions = []
    if not results.multi_face_landmarks:
        ear_positions.append({"x": w // 2, "y": h // 2, "side": "left"})
    else:
        lm = results.multi_face_landmarks[0]
        # Left ear (camera left) – landmark 137
        l = lm.landmark[137]
        lx = int(l.x * w) - int(w * 0.025)
        ly = int(l.y * h) + int(h * 0.015)
        ear_positions.append({"x": lx, "y": ly, "side": "left"})
        # Right ear – landmark 366
        r = lm.landmark[366]
        rx = int(r.x * w) + int(w * 0.025)
        ry = int(r.y * h) + int(h * 0.015)
        ear_positions.append({"x": rx, "y": ry, "side": "right"})

    # Scale earring based on image height – factor can be tuned.
    size_w = int(h * 0.07)
    aspect = earring.shape[0] / earring.shape[1]
    size_h = int(size_w * aspect)
    if size_w < 15:
        size_w = 15
        size_h = int(size_w * aspect)

    # Optional rotation to follow head tilt.
    angle = 0.0
    if results.multi_face_landmarks:
        l_eye = lm.landmark[33]
        r_eye = lm.landmark[263]
        angle = math.degrees(math.atan2(r_eye.y - l_eye.y, r_eye.x - l_eye.x))

    # Rotate and resize once – same overlay used for both ears.
    he, we = earring.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((we // 2, he // 2), angle, 1.0)
    earring_rot = cv2.warpAffine(earring, rot_mat, (we, he), flags=cv2.INTER_LINEAR)
    earring_resized = cv2.resize(
        earring_rot, (size_w, size_h), interpolation=cv2.INTER_AREA
    )

    output = image.copy()
    for pos in ear_positions:
        output = _overlay_earring_dual(
            output, earring_resized, pos["x"], pos["y"], pos["side"]
        )

    cv2.imwrite(output_path, output)
    return output_path

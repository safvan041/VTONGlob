"""MediaPipe face landmark adapter used by the try-on engines."""

import os
from pathlib import Path
from types import SimpleNamespace


def create_face_mesh():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"),
    )

    try:
        from mediapipe.solutions import face_mesh as mp_face_mesh

        return mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        )
    except ImportError:
        return TaskFaceMesh()


class TaskFaceMesh:
    """Expose MediaPipe Tasks FaceLandmarker through the legacy FaceMesh shape."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
        if not model_path.exists():
            raise RuntimeError(
                "Missing MediaPipe model file. Download it to "
                f"{model_path} before running the try-on service."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, image_rgb):
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return SimpleNamespace(multi_face_landmarks=None)

        return SimpleNamespace(
            multi_face_landmarks=[
                SimpleNamespace(landmark=result.face_landmarks[0])
            ]
        )

    def close(self):
        self._landmarker.close()

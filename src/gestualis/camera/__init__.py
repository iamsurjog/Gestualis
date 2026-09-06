import os
from typing import Any
import numpy as np
import numpy.typing as npt
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_detector: HandLandmarker | None = None


def _resolve_model_path() -> str:
    """Find the hand_landmarker.task model file path reliably."""
    # 1. Check relative to this package root
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    candidate1 = os.path.join(package_root, "models", "hand_landmarker.task")
    if os.path.exists(candidate1):
        return candidate1

    # 2. Check current working directory
    candidate2 = os.path.abspath(os.path.join("models", "hand_landmarker.task"))
    if os.path.exists(candidate2):
        return candidate2

    # 3. Fallback to default user path
    return r"C:\Users\Sukruta Nadkarni\OneDrive - vit.ac.in\Desktop\Project - I\Gestualis\Gestualis-v1-dev\models\hand_landmarker.task"


def get_detector() -> HandLandmarker:
    """Returns a singleton HandLandmarker instance to prevent re-instantiation every frame."""
    global _detector
    if _detector is None:
        model_path = _resolve_model_path()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
        )
        _detector = HandLandmarker.create_from_options(options)
    return _detector


def close_detector() -> None:
    """Closes and resets the singleton detector instance."""
    global _detector
    if _detector is not None:
        _detector.close()
        _detector = None


def get_landmarks(
    img: npt.NDArray[np.uint8],
) -> tuple[dict[int, npt.NDArray[np.float64]], Any]:
    """
    Detects hand landmarks in an RGB image. Reuses the persistent detector
    instance for high performance, smooth tracking, and minimal memory usage.

    Returns:
        (formatted_landmarks, hand_landmarker_result)
        formatted_landmarks is {0: np.array([x, y, z]), ...} or {} if no hands detected.
    """
    detector = get_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    hand_landmarker_result = detector.detect(mp_image)

    if not hand_landmarker_result.hand_landmarks:
        return {}, hand_landmarker_result

    first_hand_landmarks = hand_landmarker_result.hand_landmarks[0]
    formatted_landmarks = {
        i: np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)
        for i, landmark in enumerate(first_hand_landmarks)
    }

    return formatted_landmarks, hand_landmarker_result





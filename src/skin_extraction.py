"""
Face landmark based skin-tone sampling.

Method (as specified): find the face, drop a vertical line straight down
from the pupil and a horizontal line straight across from the nose tip;
where they cross, sample a small circular patch of skin color there.
Only one side of the face is used (whichever eye appears further left in
the photo, i.e. camera-frame left -- not "the subject's own left/right",
to avoid mirroring confusion).

NOTE: the exact mediapipe landmark index used for "nose tip" (NOSE_TIP_IDX
below) is the commonly documented one, but should be double-checked
against a real sample selfie once we have one -- flagged for the
real-environment testing pass.
"""

from dataclasses import dataclass
import numpy as np
import cv2
import mediapipe as mp

LEFT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_CENTER_IDX = 473
NOSE_TIP_IDX = 4  # TODO: verify against a real selfie

_face_mesh = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # needed for iris landmarks 468-477
            min_detection_confidence=0.5,
        )
    return _face_mesh


@dataclass
class SkinSampleResult:
    success: bool
    message: str = ""
    raw_rgb: np.ndarray = None
    sample_point: tuple = None
    sample_radius: int = None
    used_eye: str = None  # "left_in_frame" or "right_in_frame"


def _landmark_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def extract_skin_sample(bgr_img: np.ndarray, radius_factor: float = 0.16) -> SkinSampleResult:
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    face_mesh = _get_face_mesh()
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return SkinSampleResult(False, "사진에서 얼굴을 찾지 못했어요")

    landmarks = results.multi_face_landmarks[0].landmark

    iris_a = _landmark_px(landmarks, LEFT_IRIS_CENTER_IDX, w, h)
    iris_b = _landmark_px(landmarks, RIGHT_IRIS_CENTER_IDX, w, h)
    nose_tip = _landmark_px(landmarks, NOSE_TIP_IDX, w, h)

    # Pick whichever iris is further left *in the image* (camera-frame
    # left), not by mediapipe's subject-relative left/right labels.
    if iris_a[0] <= iris_b[0]:
        chosen_iris, used_eye = iris_a, "left_in_frame"
    else:
        chosen_iris, used_eye = iris_b, "right_in_frame"

    interpupillary_dist = float(np.linalg.norm(iris_a - iris_b))
    radius = max(6, int(interpupillary_dist * radius_factor))

    sample_x = chosen_iris[0]
    sample_y = nose_tip[1]
    cx, cy = int(sample_x), int(sample_y)

    if not (0 <= cx < w and 0 <= cy < h):
        return SkinSampleResult(False, "샘플링 위치가 사진 범위를 벗어났어요 (얼굴이 더 잘 보이게 다시 촬영해주세요)")

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    pixels = rgb[mask == 255]
    if pixels.size == 0:
        return SkinSampleResult(False, "피부색 샘플을 추출하지 못했어요")

    # Robust to occasional specular highlight / shadow pixels within the circle.
    median_rgb = np.median(pixels.reshape(-1, 3), axis=0)

    return SkinSampleResult(
        True,
        "피부색 샘플 추출 완료",
        raw_rgb=median_rgb,
        sample_point=(cx, cy),
        sample_radius=radius,
        used_eye=used_eye,
    )

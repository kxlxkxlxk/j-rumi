"""
End-to-end pipeline: a single photo containing BOTH the ColorChecker card
and the subject's face -> camera/lighting-corrected skin Lab value ->
best-matching foundation shade(s) from the DB.
"""
from dataclasses import dataclass, field
import numpy as np

from .calibration import calibrate_from_image
from .skin_extraction import extract_skin_sample
from .color_match import rgb_to_lab, find_best_matches


@dataclass
class RecommendationResult:
    success: bool
    message: str = ""
    corrected_lab: np.ndarray = None
    matches: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def recommend_foundation(bgr_img: np.ndarray, shades: list, top_n: int = 3) -> RecommendationResult:
    calib = calibrate_from_image(bgr_img)
    if not calib.success:
        return RecommendationResult(False, f"색상카드 인식 실패: {calib.message}")

    skin = extract_skin_sample(bgr_img)
    if not skin.success:
        return RecommendationResult(False, f"얼굴 인식 실패: {skin.message}")

    corrected_rgb = np.clip(calib.correction_matrix @ np.append(skin.raw_rgb, 1.0), 0, 255)
    lab = rgb_to_lab(corrected_rgb)
    matches = find_best_matches(lab, shades, top_n=top_n)

    return RecommendationResult(
        True,
        "추천 완료",
        corrected_lab=lab,
        matches=matches,
        debug={
            "raw_skin_rgb": skin.raw_rgb,
            "corrected_skin_rgb": corrected_rgb,
            "calibration_mean_error": calib.mean_delta_e,
            "sample_point": skin.sample_point,
            "sample_radius": skin.sample_radius,
        },
    )


def build_shade_from_photos(bgr_imgs: list, swatch_box_picker=None) -> dict:
    """For the admin DB-builder: given up to 5 photos of (card + foundation
    swatch), calibrate each and sample the swatch region, then average.

    swatch_box_picker(calib, bgr_img) -> (x, y, w, h) in original image
    coordinates for where the swatch is; if not provided, the caller is
    expected to have already cropped each image to just the swatch area
    next to the card (simplest for the admin UI: user marks/crops it).
    Returns {"L":.., "a":.., "b":.., "n_photos_used":..}
    """
    labs = []
    for img in bgr_imgs:
        calib = calibrate_from_image(img)
        if not calib.success:
            continue
        if swatch_box_picker is not None:
            box = swatch_box_picker(calib, img)
            x, y, w, h = box
            crop = img[y : y + h, x : x + w]
        else:
            crop = img
        median_bgr = np.median(crop.reshape(-1, 3), axis=0)
        raw_rgb = median_bgr[::-1]
        corrected_rgb = np.clip(calib.correction_matrix @ np.append(raw_rgb, 1.0), 0, 255)
        labs.append(rgb_to_lab(corrected_rgb))

    if not labs:
        return None
    mean_lab = np.mean(labs, axis=0)
    return {"L": round(float(mean_lab[0]), 3), "a": round(float(mean_lab[1]), 3), "b": round(float(mean_lab[2]), 3), "n_photos_used": len(labs)}

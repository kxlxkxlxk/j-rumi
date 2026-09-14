"""
End-to-end pipeline: a single photo containing BOTH the ColorChecker card
and the subject's face -> camera/lighting-corrected skin Lab value ->
best-matching foundation shade(s) from the DB.

Skin color step (matches the report's documented approach): sample several
ROIs on the face (both lower cheeks, under the mouth, chin) rather than a
single point, correct each with the card-based calibration, convert to
Lab, then pick the most representative one via a CIEDE2000 "medoid" --
the ROI closest (in perceptual color distance) to all the others -- after
dropping any ROI that looks like an outlier (blush, stray shadow, a
slightly clipped highlight).
"""
from dataclasses import dataclass, field
import numpy as np
from skimage.color import deltaE_ciede2000

from .calibration import calibrate_from_image
from .skin_extraction import extract_skin_regions
from .color_match import rgb_to_lab, find_best_matches


@dataclass
class RecommendationResult:
    success: bool
    message: str = ""
    corrected_lab: np.ndarray = None
    matches: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def _pairwise_delta_e(lab_list) -> np.ndarray:
    n = len(lab_list)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lab_i = np.array(lab_list[i], dtype=np.float64).reshape(1, 1, 3)
            lab_j = np.array(lab_list[j], dtype=np.float64).reshape(1, 1, 3)
            de = float(deltaE_ciede2000(lab_i, lab_j)[0, 0])
            d[i, j] = d[j, i] = de
    return d


def _select_medoid(region_labs, outlier_factor: float = 1.8):
    """1) Compute pairwise CIEDE2000 distance between every ROI's Lab.
    2) The medoid is the ROI whose total distance to all others is
       smallest (the one the rest most agree with).
    3) Any ROI much farther from the medoid than the typical pairwise
       distance is dropped as an outlier, then the medoid is recomputed
       among the rest.
    Returns (final_lab, medoid_index, kept_indices, dist_matrix)."""
    n = len(region_labs)
    if n == 1:
        return region_labs[0], 0, [0], np.zeros((1, 1))

    dist = _pairwise_delta_e(region_labs)
    totals = dist.sum(axis=1)
    medoid_idx = int(np.argmin(totals))

    off_diag = dist[dist > 0]
    median_pairwise = float(np.median(off_diag)) if off_diag.size else 0.0
    # ΔE < ~3 is close to "barely perceptible" -- don't prune on noise alone
    threshold = max(median_pairwise * outlier_factor, 3.0)
    kept = [i for i in range(n) if dist[i, medoid_idx] <= threshold]

    if 2 <= len(kept) < n:
        sub_dist = dist[np.ix_(kept, kept)]
        sub_totals = sub_dist.sum(axis=1)
        medoid_idx = kept[int(np.argmin(sub_totals))]
    else:
        kept = list(range(n))

    return region_labs[medoid_idx], medoid_idx, kept, dist


def recommend_foundation(bgr_img: np.ndarray, shades: list, top_n: int = 3) -> RecommendationResult:
    calib = calibrate_from_image(bgr_img)
    if not calib.success:
        return RecommendationResult(False, f"색상카드 인식 실패: {calib.message}")

    skin = extract_skin_regions(bgr_img)
    if not skin.success:
        return RecommendationResult(False, f"얼굴 인식 실패: {skin.message}")

    region_labs = []
    region_debug = []
    for region in skin.regions:
        corrected_rgb = np.clip(calib.correction_matrix @ np.append(region.raw_rgb, 1.0), 0, 255)
        lab = rgb_to_lab(corrected_rgb)
        region_labs.append(lab)
        region_debug.append(
            {
                "name": region.name,
                "raw_rgb": [round(float(x), 1) for x in region.raw_rgb],
                "corrected_rgb": [round(float(x), 1) for x in corrected_rgb],
                "lab": [round(float(x), 2) for x in lab],
                "center": region.center,
                "radius": region.radius,
            }
        )

    final_lab, medoid_idx, kept_idx, _dist = _select_medoid(region_labs)
    for i, rd in enumerate(region_debug):
        rd["used_as_final"] = i == medoid_idx
        rd["excluded_as_outlier"] = i not in kept_idx

    matches = find_best_matches(final_lab, shades, top_n=top_n)

    return RecommendationResult(
        True,
        "추천 완료",
        corrected_lab=final_lab,
        matches=matches,
        debug={
            "regions": region_debug,
            "final_region": skin.regions[medoid_idx].name,
            "calibration_mean_error": calib.mean_delta_e,
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

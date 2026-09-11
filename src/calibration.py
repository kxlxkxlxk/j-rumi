"""
ColorChecker Classic (24-patch) detection + color-correction calibration.

Pipeline:
  1. Find the black card in the photo (largest dark quadrilateral contour).
  2. Perspective-warp it to a flat canonical rectangle.
  3. Detect the 24 individual color patches inside the warped card.
  4. Sample each patch's color (robust median of a central crop).
  5. Match the 24 sampled colors to the 24 known reference colors
     (Hungarian assignment on color distance -- works regardless of the
     card's rotation/orientation in the photo).
  6. Solve a linear color-correction transform (observed -> reference).

The same `find_card_and_correction()` function is used both when building
the foundation DB (photo of card + foundation swatch) and when a user
submits a face photo (photo of card + face) -- in both cases we first
figure out "what this camera/lighting did to a known color" and correct
the *other* thing in the same photo by the same transform.
"""

from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .reference_colors import REFERENCE_RGB_LIST, REFERENCE_NAMES

CANONICAL_W, CANONICAL_H = 1200, 800  # landscape canonical warp size (~3:2 card)


@dataclass
class CalibrationResult:
    success: bool
    message: str = ""
    correction_matrix: np.ndarray = None  # 3x4 affine (maps [R,G,B,1] observed -> corrected RGB)
    mean_delta_e: float = None
    card_corners: np.ndarray = None  # 4x2 in original image coords
    patch_centers_canonical: np.ndarray = None
    observed_patch_rgb: np.ndarray = None
    matched_reference_rgb: np.ndarray = None


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _largest_quad_from_mask(mask: np.ndarray, img_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.01 or area > img_area * 0.95:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        quad = _order_quad_points(np.array(box))
        side_top = np.linalg.norm(quad[1] - quad[0])
        side_bottom = np.linalg.norm(quad[2] - quad[3])
        side_left = np.linalg.norm(quad[3] - quad[0])
        side_right = np.linalg.norm(quad[2] - quad[1])
        long_side = max(side_top, side_bottom, side_left, side_right)
        short_side = min(side_top, side_bottom, side_left, side_right)
        if short_side < 1:
            continue
        aspect = long_side / short_side
        # ColorChecker Classic is roughly 1.4-1.6 : 1
        if 1.2 < aspect < 2.0:
            # how well the contour fills its own rotated bounding box
            # (rejects L-shaped / partial blobs)
            rect_area = short_side * long_side
            fill = area / rect_area if rect_area > 0 else 0
            if fill > 0.75:
                candidates.append((area, quad))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def find_card_quad(bgr_img: np.ndarray):
    """Locate the ColorChecker's outer black border as a quadrilateral.

    The card frame is near-black against a much brighter, fairly uniform
    background, so a single global (Otsu) threshold on brightness cleanly
    separates the two -- adaptive thresholding was tried first but broke
    the frame into disconnected pieces under shading gradients.
    """
    h, w = bgr_img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    quad = _largest_quad_from_mask(otsu, img_area)
    if quad is not None:
        return quad

    # Fallback: fixed low-brightness threshold in case Otsu's split point
    # was skewed by a large dark background.
    for cutoff in (40, 60, 80):
        _, mask = cv2.threshold(blur, cutoff, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        quad = _largest_quad_from_mask(mask, img_area)
        if quad is not None:
            return quad
    return None


def warp_card(bgr_img: np.ndarray, quad: np.ndarray):
    side_w = max(
        np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
    )
    side_h = max(
        np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
    )
    landscape = side_w >= side_h
    out_w, out_h = (CANONICAL_W, CANONICAL_H) if landscape else (CANONICAL_H, CANONICAL_W)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr_img, M, (out_w, out_h))
    return warped


def _find_bright_patch_boxes(warped_bgr: np.ndarray):
    """Contour-detect the lighter/more saturated patches (dark ones such as
    black, dark gray, dark purple blend into the card's black grid and are
    reliably missed here -- that's fixed by _complete_grid below)."""
    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = h * w
    expected_patch_area = total_area / 24
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < expected_patch_area * 0.25 or area > expected_patch_area * 2.5:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh)
        if 0.5 < ar < 2.0:
            boxes.append((x, y, bw, bh))
    return boxes


def _cluster_1d(values, min_gap):
    """Group nearby scalar values into clusters; return sorted cluster means."""
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _kmeans_1d(values, k, n_iter=50):
    """Small fixed-k 1D k-means (grid column/row positions are well
    separated, so this converges trivially -- avoids depending on an
    extra clustering library)."""
    values = np.array(sorted(values), dtype=np.float64)
    # even-quantile init keeps it stable regardless of uneven point counts per cluster
    centers = np.quantile(values, np.linspace(0.05, 0.95, k))
    for _ in range(n_iter):
        dists = np.abs(values[:, None] - centers[None, :])
        assign = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            pts = values[assign == i]
            if len(pts) > 0:
                new_centers[i] = pts.mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return sorted(centers.tolist())


def detect_patches(warped_bgr: np.ndarray):
    """Find all 24 color-patch cells inside a warped card image.

    Bright/saturated patches are found directly by contour. Their centers
    are then clustered into grid rows and columns (the card's patches are
    laid out on a perfectly regular grid), which reveals the row/column
    positions even for dark patches no contour was found for -- those
    cells are filled in at the expected grid intersection.
    """
    boxes = _find_bright_patch_boxes(warped_bgr)
    if len(boxes) < 6:
        return boxes  # too little signal to infer a grid

    h, w = warped_bgr.shape[:2]
    # The ColorChecker Classic is always 6x4 patches; we warp to a canonical
    # rectangle above so the orientation (landscape vs portrait) tells us
    # which axis has 6 and which has 4.
    n_cols, n_rows = (6, 4) if w >= h else (4, 6)

    centers = [(x + bw / 2, y + bh / 2) for (x, y, bw, bh) in boxes]
    median_w = float(np.median([b[2] for b in boxes]))
    median_h = float(np.median([b[3] for b in boxes]))

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    col_xs = _kmeans_1d(xs, n_cols)
    row_ys = _kmeans_1d(ys, n_rows)

    full_boxes = []
    for cy in row_ys:
        for cx in col_xs:
            full_boxes.append(
                (int(cx - median_w / 2), int(cy - median_h / 2), int(median_w), int(median_h))
            )
    return full_boxes


def _sample_patch_color(warped_bgr: np.ndarray, box):
    x, y, bw, bh = box
    # Sample the central 50% of the patch to avoid grid-line/edge bleed.
    cx0 = x + int(bw * 0.25)
    cx1 = x + int(bw * 0.75)
    cy0 = y + int(bh * 0.25)
    cy1 = y + int(bh * 0.75)
    crop = warped_bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        crop = warped_bgr[y : y + bh, x : x + bw]
    median_bgr = np.median(crop.reshape(-1, 3), axis=0)
    return median_bgr[::-1]  # -> RGB


def match_patches_to_reference(observed_rgb: np.ndarray):
    """Hungarian-match observed patch colors to the 24 known reference colors."""
    ref = np.array(REFERENCE_RGB_LIST, dtype=np.float64)
    obs = np.array(observed_rgb, dtype=np.float64)
    n = min(len(ref), len(obs))
    cost = np.zeros((len(obs), len(ref)))
    for i, o in enumerate(obs):
        cost[i] = np.linalg.norm(ref - o, axis=1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost


def solve_correction_matrix(observed_rgb: np.ndarray, reference_rgb: np.ndarray):
    """Least-squares affine map: reference ≈ M @ [observed; 1]."""
    obs = np.array(observed_rgb, dtype=np.float64)
    ref = np.array(reference_rgb, dtype=np.float64)
    ones = np.ones((obs.shape[0], 1))
    A = np.hstack([obs, ones])  # N x 4
    M, *_ = np.linalg.lstsq(A, ref, rcond=None)  # 4 x 3
    return M.T  # 3 x 4


def apply_correction(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    rgb = np.array(rgb, dtype=np.float64)
    vec = np.append(rgb, 1.0)
    corrected = M @ vec
    return np.clip(corrected, 0, 255)


def calibrate_from_image(bgr_img: np.ndarray) -> CalibrationResult:
    quad = find_card_quad(bgr_img)
    if quad is None:
        return CalibrationResult(False, "카드를 사진에서 찾지 못했어요 (색상카드가 잘 보이게 다시 촬영해주세요)")

    warped = warp_card(bgr_img, quad)
    boxes = detect_patches(warped)
    if len(boxes) < 20:
        return CalibrationResult(
            False, f"카드 안 색상 패치를 충분히 찾지 못했어요 ({len(boxes)}/24개 인식)"
        )

    observed_rgb = np.array([_sample_patch_color(warped, b) for b in boxes])
    row_ind, col_ind, cost = match_patches_to_reference(observed_rgb)

    matched_observed = observed_rgb[row_ind]
    matched_reference = np.array(REFERENCE_RGB_LIST)[col_ind]

    M = solve_correction_matrix(matched_observed, matched_reference)

    corrected = np.array([apply_correction(o, M) for o in matched_observed])
    mean_err = float(np.mean(np.linalg.norm(corrected - matched_reference, axis=1)))

    centers = np.array([[x + bw / 2, y + bh / 2] for (x, y, bw, bh) in boxes])

    return CalibrationResult(
        True,
        f"카드 {len(boxes)}개 패치 인식 완료, 평균 보정 오차 {mean_err:.1f}",
        correction_matrix=M,
        mean_delta_e=mean_err,
        card_corners=quad,
        patch_centers_canonical=centers[row_ind],
        observed_patch_rgb=matched_observed,
        matched_reference_rgb=matched_reference,
    )

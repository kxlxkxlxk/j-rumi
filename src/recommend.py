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
from .color_match import rgb_to_lab, find_best_matches_meta


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


def _agreement_clusters(dist: np.ndarray, threshold: float):
    """Union-find over region indices: connect i,j whenever their
    perceptual distance is within `threshold` (i.e. they plausibly read
    the same real skin tone). Returns a list of clusters (each a list of
    indices)."""
    n = dist.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _select_medoid(region_labs, agreement_threshold: float = 8.0, region_names=None):
    """1) Compute pairwise CIEDE2000 distance between every ROI's Lab.
    2) Group ROIs into clusters of mutual agreement (perceptually close
       to each other, within agreement_threshold) via union-find.
    3) Take the LARGEST cluster -- the set of regions that most agree
       with each other on a skin tone. Ties are broken by which cluster
       is internally tightest (lowest average pairwise distance): if two
       clusters are the same size, the one where members agree more
       closely with each other is more likely to be measuring real,
       evenly-lit skin rather than a mix contaminated by shadow/highlight.
    4) Within that cluster, the medoid is the member closest (by total
       distance) to the rest of the cluster.

    This replaces a plain "closest to everyone" medoid, which can pick
    the wrong side on an even split -- e.g. 2 well-lit cheek ROIs that
    closely agree with each other vs. 2 shadowed under-mouth/chin ROIs
    that also loosely agree with each other: summing distance to ALL
    other regions lets one shadowed region "win" by a hair even though
    the other side is the tighter, more mutually-consistent group.

    Anatomical cheek priority (step 0, before clustering): chin and
    under_mouth sit right next to the same shadow-casting features (the
    lower lip, the jawline) and get darkened by the SAME light source at
    the SAME time. That makes them look artificially "tight"/mutually
    consistent to a pure statistics-based tightness tiebreak -- tighter
    than two independently-lit cheeks -- even though the cheeks are the
    standard, more reliable skin-sampling site. So: if both cheek_a and
    cheek_b were sampled and they agree with each other (within
    agreement_threshold), trust them immediately and short-circuit
    before the general largest-cluster/tightness comparison ever runs.
    Only when the cheeks themselves disagree (occlusion, stray highlight
    on one side, etc. -- meaning we can't trust them) does this fall
    through to the general logic below.

    Returns (final_lab, medoid_index, kept_indices, dist_matrix)."""
    n = len(region_labs)
    if n == 1:
        return region_labs[0], 0, [0], np.zeros((1, 1))

    dist = _pairwise_delta_e(region_labs)

    if region_names is not None and "cheek_a" in region_names and "cheek_b" in region_names:
        ia = region_names.index("cheek_a")
        ib = region_names.index("cheek_b")
        if dist[ia, ib] <= agreement_threshold:
            kept = [ia, ib]
            sub_dist = dist[np.ix_(kept, kept)]
            sub_totals = sub_dist.sum(axis=1)
            medoid_idx = kept[int(np.argmin(sub_totals))]
            return region_labs[medoid_idx], medoid_idx, kept, dist

    clusters = _agreement_clusters(dist, agreement_threshold)
    max_size = max(len(c) for c in clusters)
    largest = [c for c in clusters if len(c) == max_size]

    def _tightness(cluster):
        if len(cluster) == 1:
            return 0.0
        pairs = [dist[i, j] for a, i in enumerate(cluster) for j in cluster[a + 1 :]]
        return float(np.mean(pairs))

    largest.sort(key=_tightness)
    kept = largest[0]

    if len(kept) == 1:
        medoid_idx = kept[0]
    else:
        sub_dist = dist[np.ix_(kept, kept)]
        sub_totals = sub_dist.sum(axis=1)
        medoid_idx = kept[int(np.argmin(sub_totals))]

    return region_labs[medoid_idx], medoid_idx, kept, dist


def recommend_foundation(
    bgr_img: np.ndarray, shades: list, top_n: int = 3, card_bgr: np.ndarray = None
) -> RecommendationResult:
    """bgr_img: the full photo (card + face), used for face/skin detection.
    card_bgr: optional -- a crop containing ONLY the color card, used for
    calibration instead of bgr_img. Real photos with hair/clothing/jewelry
    next to the card can fool automatic card-quad detection, so the card
    region is user-cropped in the UI; pass that crop here. Falls back to
    bgr_img itself (old automatic-detection-on-the-whole-photo behavior)
    when not given.
    """
    calib = calibrate_from_image(card_bgr if card_bgr is not None else bgr_img)
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

    final_lab, medoid_idx, kept_idx, _dist = _select_medoid(
        region_labs, region_names=[r.name for r in skin.regions]
    )
    for i, rd in enumerate(region_debug):
        rd["used_as_final"] = i == medoid_idx
        rd["excluded_as_outlier"] = i not in kept_idx

    matches, match_meta = find_best_matches_meta(final_lab, shades, top_n=top_n, require_brighter=True)

    return RecommendationResult(
        True,
        "추천 완료",
        corrected_lab=final_lab,
        matches=matches,
        debug={
            "regions": region_debug,
            "final_region": skin.regions[medoid_idx].name,
            "calibration_mean_error": calib.mean_delta_e,
            "correction_matrix": calib.correction_matrix.tolist(),
            "brighter_pool_used": match_meta["brighter_pool_used"],
            "n_candidates_considered": match_meta["n_candidates"],
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

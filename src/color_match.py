"""
Lab conversion + perceptual color distance against the foundation DB.
"""
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def rgb_to_lab(rgb_0_255: np.ndarray) -> np.ndarray:
    """rgb_0_255: array-like [R,G,B] each 0-255 -> Lab [L,a,b]."""
    rgb = np.array(rgb_0_255, dtype=np.float64) / 255.0
    rgb = np.clip(rgb, 0, 1).reshape(1, 1, 3)
    lab = rgb2lab(rgb)
    return lab[0, 0]


def find_best_matches(lab_value: np.ndarray, shades: list, top_n: int = 3, require_brighter: bool = True):
    """shades: list of dicts with at least 'L','a','b' (+ any metadata).
    Returns shades sorted by CIEDE2000 distance, each with a 'delta_e' key.

    The measured skin color itself is NOT touched -- extraction/calibration
    stays a straight, defensible color measurement. require_brighter
    controls the comparison POOL used for ranking, not the measurement:
    when True (default), only DB shades at least as light as the measured
    skin tone (shade L >= measured L) are considered at all, and among
    only those, the one closest (smallest ΔE) to the measured tone wins.
    Shades darker than the measured skin are never picked. If no shade in
    the DB is light enough (the person's skin already reads lighter than
    every shade on file), this falls back to the full DB so the app still
    returns something rather than nothing -- 'brighter_pool_used' in the
    return metadata says which happened (see find_best_matches_meta).
    """
    matches, _meta = find_best_matches_meta(lab_value, shades, top_n, require_brighter)
    return matches


def find_best_matches_meta(lab_value: np.ndarray, shades: list, top_n: int = 3, require_brighter: bool = True):
    """Same as find_best_matches, but also returns a small metadata dict
    -- {"brighter_pool_used": bool, "n_candidates": int} -- so callers can
    show/debug whether the brighter-only filter actually applied or fell
    back to the full DB."""
    lab1 = np.array(lab_value, dtype=np.float64).reshape(1, 1, 3)
    measured_L = float(lab_value[0])

    candidates = shades
    brighter_pool_used = False
    if require_brighter:
        brighter = [s for s in shades if s["L"] >= measured_L]
        if brighter:
            candidates = brighter
            brighter_pool_used = True

    scored = []
    for shade in candidates:
        lab2 = np.array([shade["L"], shade["a"], shade["b"]], dtype=np.float64).reshape(1, 1, 3)
        de = float(deltaE_ciede2000(lab1, lab2)[0, 0])
        scored.append({**shade, "delta_e": de})
    scored.sort(key=lambda s: s["delta_e"])

    meta = {"brighter_pool_used": brighter_pool_used, "n_candidates": len(candidates)}
    return scored[:top_n], meta

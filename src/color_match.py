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


def find_best_matches(lab_value: np.ndarray, shades: list, top_n: int = 3):
    """shades: list of dicts with at least 'L','a','b' (+ any metadata).
    Returns shades sorted by CIEDE2000 distance, each with a 'delta_e' key.
    """
    lab1 = np.array(lab_value, dtype=np.float64).reshape(1, 1, 3)
    scored = []
    for shade in shades:
        lab2 = np.array([shade["L"], shade["a"], shade["b"]], dtype=np.float64).reshape(1, 1, 3)
        de = float(deltaE_ciede2000(lab1, lab2)[0, 0])
        scored.append({**shade, "delta_e": de})
    scored.sort(key=lambda s: s["delta_e"])
    return scored[:top_n]

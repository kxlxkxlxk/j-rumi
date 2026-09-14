"""
Calibrite/X-Rite ColorChecker Classic (24-patch) reference sRGB values.

These are the standard published average sRGB values for the 24 patches
(D65 viewing / sRGB color space). Used as calibration anchors: whatever
order the 24 patches are detected in a photo, they get matched to this
set of 24 known colors (not by position, but by nearest color, since a
photo could be rotated/flipped) and a correction transform is solved.

Reference: standard ColorChecker Classic sRGB spec, widely published
(e.g. https://en.wikipedia.org/wiki/ColorChecker).
"""

# name -> (R, G, B) in 0-255 sRGB
REFERENCE_PATCHES = {
    "dark_skin":      (115, 82, 68),
    "light_skin":     (194, 150, 130),
    "blue_sky":       (98, 122, 157),
    "foliage":        (87, 108, 67),
    "blue_flower":    (133, 128, 177),
    "bluish_green":   (103, 189, 170),
    "orange":         (214, 126, 44),
    "purplish_blue":  (80, 91, 166),
    "moderate_red":   (193, 90, 99),
    "purple":         (94, 60, 108),
    "yellow_green":   (157, 188, 64),
    "orange_yellow":  (224, 163, 46),
    "blue":           (56, 61, 150),
    "green":          (70, 148, 73),
    "red":            (175, 54, 60),
    "yellow":         (231, 199, 31),
    "magenta":        (187, 86, 149),
    "cyan":           (8, 133, 161),
    "white_95":       (243, 243, 242),
    "neutral_80":     (200, 200, 200),
    "neutral_65":     (160, 160, 160),
    "neutral_50":     (122, 122, 121),
    "neutral_35":     (85, 85, 85),
    "black_20":       (52, 52, 52),
}

REFERENCE_RGB_LIST = list(REFERENCE_PATCHES.values())
REFERENCE_NAMES = list(REFERENCE_PATCHES.keys())

assert len(REFERENCE_RGB_LIST) == 24


# ---- optional custom reference (captured from the user's own physical
# card under clean/ideal lighting, via the admin page) -----------------
#
# The official values above are what most calibration workflows use, but
# a specific printed card can drift from spec (printing batch, age,
# lighting used when it was measured), and correcting every photo toward
# a target that's slightly off from THIS card can look worse than
# correcting toward what the card actually, verifiably reads under good
# light. If the admin has captured a clean reference photo, use that
# instead -- merged over the official values so a partial capture (not
# all 24 patches) still falls back sensibly for the rest.
import time

_custom_cache = {"patches": None, "loaded_at": 0.0}
_CACHE_TTL_SECONDS = 60


def get_active_reference_patches():
    """The 24-patch {name: (R, G, B)} reference actually used for color
    correction targets. Cached briefly (calibrate_from_image tries
    several card-location candidates per photo, so without caching this
    would mean a GitHub API round-trip per candidate)."""
    now = time.time()
    if now - _custom_cache["loaded_at"] > _CACHE_TTL_SECONDS:
        _custom_cache["loaded_at"] = now
        try:
            from . import github_storage

            data, _sha = github_storage.read_card_reference()
        except Exception:
            data = None
        _custom_cache["patches"] = data

    data = _custom_cache["patches"]
    if data and data.get("patches"):
        merged = dict(REFERENCE_PATCHES)
        for name, rgb in data["patches"].items():
            if name in merged:
                merged[name] = tuple(rgb)
        return merged
    return REFERENCE_PATCHES


def get_active_reference_rgb_list():
    patches = get_active_reference_patches()
    return [patches[name] for name in REFERENCE_NAMES]


def clear_reference_cache():
    """Call right after the admin page saves or resets the custom
    reference so THIS session picks up the change immediately instead of
    waiting out the cache TTL."""
    _custom_cache["patches"] = None
    _custom_cache["loaded_at"] = 0.0

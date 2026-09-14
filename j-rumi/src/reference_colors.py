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

"""Tissue / pallor saliency helpers.

These purely image-based signals are used to:
1. Bias Grad-CAM heatmaps toward flesh-coloured regions (conjunctiva / nail bed)
   so the heat doesn't land on the iris, background, or shadows.
2. Provide a complementary anemia signal from the nail bed, since the 6-class
   nail CNN does NOT directly predict anemia (only 3 of its 6 classes are
   anemia-positive).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _rgb_to_hsv_numpy(arr01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB->HSV. arr01 in [0,1], returns h,s,v each in [0,1]."""
    r, g, b = arr01[..., 0], arr01[..., 1], arr01[..., 2]
    maxc = np.max(arr01, axis=-1)
    minc = np.min(arr01, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-8), 0.0)
    # Hue
    rc = np.where(delta > 0, (maxc - r) / np.maximum(delta, 1e-8), 0)
    gc = np.where(delta > 0, (maxc - g) / np.maximum(delta, 1e-8), 0)
    bc = np.where(delta > 0, (maxc - b) / np.maximum(delta, 1e-8), 0)
    h = np.where(r == maxc, bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return h, s, v


def tissue_mask(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    """Soft mask in [0,1] highlighting pink/red flesh-toned pixels.

    Used as a multiplicative gate for Grad-CAM heatmaps.
    """
    img = image
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv_numpy(arr)

    # "Pinkness": hue close to 0 (or 1) — red/pink/orange/yellow-pink band
    hue_dist = np.minimum(h, 1.0 - h)              # 0 = pure red, 0.5 = cyan
    hue_score = np.clip(1.0 - hue_dist / 0.22, 0.0, 1.0)  # full weight within ~80° of red

    # Reject very dark / very desaturated regions (eye pupil, deep shadows)
    val_score = np.clip((v - 0.18) / 0.5, 0.0, 1.0)
    sat_score = np.clip((s - 0.05) / 0.4, 0.0, 1.0)

    mask = hue_score * (0.4 + 0.6 * (val_score * 0.6 + sat_score * 0.4))
    # Smooth slightly so the mask doesn't pixelate the heatmap
    mask = _box_blur(mask, k=5)
    return np.clip(mask, 0.0, 1.0)


def _box_blur(arr: np.ndarray, k: int = 5) -> np.ndarray:
    """Simple separable mean blur — no scipy dependency."""
    if k <= 1:
        return arr
    pad = k // 2
    a = np.pad(arr, pad, mode="edge")
    # vertical
    out = np.zeros_like(arr)
    for i in range(k):
        out += a[i:i + arr.shape[0], pad:pad + arr.shape[1]]
    out /= k
    # horizontal
    a = np.pad(out, pad, mode="edge")
    out2 = np.zeros_like(arr)
    for j in range(k):
        out2 += a[pad:pad + arr.shape[0], j:j + arr.shape[1]]
    out2 /= k
    return out2


def pallor_score(image: Image.Image) -> dict:
    """Estimate how pale the dominant tissue region looks.

    Returns dict with `pallor` in [0,1] where higher = more anemia-like
    (pale / desaturated tissue), plus the mean tissue saturation/lightness
    used to derive it.
    """
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv_numpy(arr)

    # Identify the tissue area (pink/red, not background)
    mask = tissue_mask(img, size=img.size)
    weight = float(mask.sum()) + 1e-6
    if weight < 200:    # tissue too small / not detected
        return {"pallor": 0.0, "tissue_coverage": 0.0, "mean_sat": 0.0, "mean_light": 0.0}

    mean_sat = float((s * mask).sum() / weight)
    # Lightness from HSV's V (close enough to L for our purpose)
    mean_light = float((v * mask).sum() / weight)
    coverage = float(weight / mask.size)

    # Anemic = pale = LOW saturation in the tissue area (primary signal),
    # mildly boosted when the tissue is also bright. We deliberately avoid
    # letting brightness alone drive the score — well-lit healthy tissue is
    # bright too.
    sat_score = float(np.clip(1.0 - mean_sat * 2.2, 0.0, 1.0))
    light_score = float(np.clip((mean_light - 0.55) / 0.4, 0.0, 1.0))
    pallor = float(np.clip(sat_score * (0.7 + 0.3 * light_score), 0.0, 1.0))

    return {
        "pallor": pallor,
        "tissue_coverage": coverage,
        "mean_sat": mean_sat,
        "mean_light": mean_light,
    }

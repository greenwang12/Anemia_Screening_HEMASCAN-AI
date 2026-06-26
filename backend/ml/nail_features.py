"""Clinical-feature nail analyzer.

Detects the canonical anemia-related nail signs from the image itself,
complementing the 6-class CNN. Every detector is implemented with simple,
inspectable numpy operations so a teacher / reviewer can audit each step.

Outputs (each in [0, 1] where higher = more anemic-looking):
  pallor       – nail-bed desaturation (delegated to ml.pallor)
  koilonychia  – spoon-shape: dark centre vs. bright edges (vertical profile)
  platonychia  – flatness:    uniform brightness, low specular convexity
  ridging      – vertical ridges / striations on the nail plate
  brittleness  – jagged / irregular distal edge of the nail
  yellowing    – yellow hue shift in tissue (auxiliary; weakly anemia-related)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .pallor import pallor_score, tissue_mask, _rgb_to_hsv_numpy, _box_blur


# -----------------------------------------------------------------------------
# Individual feature detectors
# -----------------------------------------------------------------------------

def _nail_roi(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray, mask) where mask~1 over the nail tissue."""
    img_small = img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.asarray(img_small, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    mask = tissue_mask(img_small, size=(224, 224))
    return gray, mask, arr


def koilonychia_score(img: Image.Image) -> float:
    """Spoon-shaped nails have a darker centre than the curved edges
    (light reflects toward the edges of the concavity). We compute the
    nail mask centroid and compare *central* vs *peripheral* brightness.
    """
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 800:
        return 0.0
    # Mask of solidly-in-tissue pixels
    solid = mask > 0.55
    ys, xs = np.where(solid)
    if len(xs) < 200:
        return 0.0
    cx, cy = float(xs.mean()), float(ys.mean())
    # Distance map from centroid, normalized by tissue radius
    yy, xx = np.indices(gray.shape)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = float(dist[solid].max())
    if radius < 10:
        return 0.0
    norm = dist / radius
    central = solid & (norm < 0.30)
    peripheral = solid & (norm > 0.55) & (norm <= 0.95)
    if central.sum() < 60 or peripheral.sum() < 60:
        return 0.0
    central_b = float(gray[central].mean())
    peripheral_b = float(gray[peripheral].mean())
    delta = peripheral_b - central_b
    # Healthy nail has slight convexity (central brighter): delta < 0 → score 0
    # Strong spoon: delta ≈ 0.10–0.20 → score ~1.0
    return float(np.clip(delta * 6.0, 0.0, 1.0))


def platonychia_score(img: Image.Image) -> float:
    """Flat nail = low variance in the central brightness profile and very
    little specular highlight. Detects 'no curvature' look."""
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 500:
        return 0.0
    tissue_pixels = gray[mask > 0.3]
    if tissue_pixels.size < 200:
        return 0.0
    std = float(tissue_pixels.std())
    # Specular highlights: top-2% brightest pixels
    p98 = float(np.percentile(tissue_pixels, 98))
    p50 = float(np.percentile(tissue_pixels, 50))
    specular_excess = max(0.0, p98 - p50)
    # Flat nail: low std AND small specular excess
    flatness = (1.0 - np.clip(std * 4.0, 0.0, 1.0)) * (1.0 - np.clip(specular_excess * 3.0, 0.0, 1.0))
    return float(np.clip(flatness, 0.0, 1.0))


def ridging_score(img: Image.Image) -> float:
    """Vertical striations on the nail plate produce strong HORIZONTAL gradients
    that repeat across the nail width. We measure the mean magnitude of the
    horizontal Sobel-like response in the tissue region."""
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 500:
        return 0.0
    # Horizontal gradient (vertical lines -> strong horizontal d/dx)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    # Tissue-weighted texture
    weight = mask.sum() + 1e-6
    ridge = float((gx * mask).sum() / weight)
    # Normalise: smooth nail (~0.012-0.018), heavy ridges (~0.040+).
    # The 0.018 baseline accounts for natural skin/nail noise.
    return float(np.clip((ridge - 0.018) * 32.0, 0.0, 1.0))


def brittleness_score(img: Image.Image) -> float:
    """Brittle / chipped nail tips have a jagged distal edge. We locate the
    tissue->non-tissue boundary along the top-of-image edge and measure how
    irregular it is."""
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 500:
        return 0.0
    # For each column, find the FIRST row (top-down) where tissue starts
    bw = mask > 0.35
    starts = []
    for c in range(bw.shape[1]):
        col = bw[:, c]
        idx = np.argmax(col)  # first True (0 if all False)
        if col[idx]:
            starts.append(idx)
    if len(starts) < 30:
        return 0.0
    starts = np.array(starts, dtype=np.float32)
    # Smooth baseline and measure deviation
    baseline = _smooth(starts, k=9)
    irregularity = float(np.abs(starts - baseline).mean())
    # Normalise: smooth edge ~0.5px, jagged ~6px
    return float(np.clip((irregularity - 0.6) / 5.0, 0.0, 1.0))


def yellowing_score(img: Image.Image) -> float:
    """Mean hue shift toward yellow inside the tissue region."""
    img_small = img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.asarray(img_small, dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv_numpy(arr)
    mask = tissue_mask(img_small, size=(224, 224)) * (s > 0.10)
    weight = float(mask.sum()) + 1e-6
    if weight < 200:
        return 0.0
    mean_h = float((h * mask).sum() / weight)
    # Yellow hue ≈ 0.13 - 0.18 (47-65°)
    dist = abs(mean_h - 0.155)
    yellow = max(0.0, 1.0 - dist / 0.06)
    return float(yellow)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

# Weights of the OpenCV-only sub-score. Sum to 1.0.
CLINICAL_FEATURE_WEIGHTS = {
    "pallor":      0.40,
    "koilonychia": 0.22,
    "ridging":     0.14,
    "brittleness": 0.10,
    "platonychia": 0.07,
    "yellowing":   0.07,
}

# Blend: 60% binary AI model + 40% OpenCV clinical features
MODEL_WEIGHT = 0.60
FEATURE_WEIGHT = 0.40


def analyze_nail_features(img: Image.Image) -> dict:
    """Run every detector and return a dict of normalized scores."""
    pallor_d = pallor_score(img)
    return {
        "pallor":      pallor_d["pallor"],
        "koilonychia": koilonychia_score(img),
        "platonychia": platonychia_score(img),
        "ridging":     ridging_score(img),
        "brittleness": brittleness_score(img),
        "yellowing":   yellowing_score(img),
        "_pallor_info": pallor_d,
    }


def combine_nail_signals(features: dict, p_model_anemia: float) -> tuple[float, float, float]:
    """Combine OpenCV clinical features with the binary AI model's P(anemia).

    Returns:
        (p_final, confidence, p_features) all in [0, 1].
    """
    p_features = float(np.clip(
        sum(CLINICAL_FEATURE_WEIGHTS[k] * features[k] for k in CLINICAL_FEATURE_WEIGHTS),
        0.0, 1.0,
    ))
    p_final = float(np.clip(MODEL_WEIGHT * p_model_anemia + FEATURE_WEIGHT * p_features, 0.0, 1.0))

    # Confidence: high when (a) model is decisive and (b) features and model agree.
    model_decisiveness = abs(p_model_anemia - 0.5) * 2.0            # 0..1
    agreement = 1.0 - abs(p_model_anemia - p_features)               # 0..1
    active = sum(1 for k in CLINICAL_FEATURE_WEIGHTS if features[k] > 0.20)
    base = 0.45 + 0.08 * active                                      # 0.45..0.93
    confidence = float(np.clip(
        0.55 * base + 0.25 * model_decisiveness + 0.20 * agreement,
        0.35, 0.95,
    ))
    return p_final, confidence, p_features


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _smooth(arr: np.ndarray, k: int = 7) -> np.ndarray:
    if k <= 1:
        return arr
    pad = k // 2
    a = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(k):
        out += a[i:i + arr.shape[0]]
    return out / k

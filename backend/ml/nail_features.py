"""Clinical-feature nail analyzer."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .pallor import pallor_score, tissue_mask, _rgb_to_hsv_numpy, _box_blur
from .config import NAIL_CONFIDENT_ANEMIA_THRESHOLD, NAIL_CONFIDENT_NON_ANEMIA_THRESHOLD


# -----------------------------------------------------------------------------
# Individual feature detectors
# -----------------------------------------------------------------------------

def _nail_roi(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gray, mask, arr) where mask~1 over the nail tissue."""
    img_small = img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.asarray(img_small, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    mask = tissue_mask(img_small, size=(224, 224))
    return gray, mask, arr


def koilonychia_score(img: Image.Image) -> float:
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 800:
        return 0.0
    solid = mask > 0.55
    ys, xs = np.where(solid)
    if len(xs) < 200:
        return 0.0
    cx, cy = float(xs.mean()), float(ys.mean())
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
    return float(np.clip(delta * 6.0, 0.0, 1.0))


def platonychia_score(img: Image.Image) -> float:
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 500:
        return 0.0
    tissue_pixels = gray[mask > 0.3]
    if tissue_pixels.size < 200:
        return 0.0
    std = float(tissue_pixels.std())
    p98 = float(np.percentile(tissue_pixels, 98))
    p50 = float(np.percentile(tissue_pixels, 50))
    specular_excess = max(0.0, p98 - p50)
    flatness = (1.0 - np.clip(std * 4.0, 0.0, 1.0)) * (1.0 - np.clip(specular_excess * 3.0, 0.0, 1.0))
    return float(np.clip(flatness, 0.0, 1.0))


def ridging_score(img: Image.Image) -> float:
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 500:
        return 0.0
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    weight = mask.sum() + 1e-6
    ridge = float((gx * mask).sum() / weight)
    return float(np.clip((ridge - 0.018) * 32.0, 0.0, 1.0))


def brittleness_score(img: Image.Image) -> float:
    gray, mask, _ = _nail_roi(img)
    if mask.sum() < 500:
        return 0.0
    bw = mask > 0.35
    starts = []
    for c in range(bw.shape[1]):
        col = bw[:, c]
        idx = np.argmax(col)
        if col[idx]:
            starts.append(idx)
    if len(starts) < 30:
        return 0.0
    starts = np.array(starts, dtype=np.float32)
    baseline = _smooth(starts, k=9)
    irregularity = float(np.abs(starts - baseline).mean())
    return float(np.clip((irregularity - 0.6) / 5.0, 0.0, 1.0))


def yellowing_score(img: Image.Image) -> float:
    img_small = img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.asarray(img_small, dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv_numpy(arr)
    mask = tissue_mask(img_small, size=(224, 224)) * (s > 0.10)
    weight = float(mask.sum()) + 1e-6
    if weight < 200:
        return 0.0
    mean_h = float((h * mask).sum() / weight)
    dist = abs(mean_h - 0.155)
    yellow = max(0.0, 1.0 - dist / 0.06)
    return float(yellow)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

CLINICAL_FEATURE_WEIGHTS = {
    "pallor":      0.45,
    "koilonychia": 0.20,
    "ridging":     0.13,
    "brittleness": 0.08,
    "platonychia": 0.07,
    "yellowing":   0.07,
}

MODEL_WEIGHT   = 0.60
FEATURE_WEIGHT = 0.40


def analyze_nail_features(img: Image.Image) -> dict:
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
    """
    Hybrid AI + OpenCV clinical analyzer.

    Workflow:
    1. Compute OpenCV clinical score.
    2. If nail appears clinically normal (low pallor), reduce CNN confidence.
    3. Blend CNN + OpenCV.
    4. Return probability and confidence.
    """

    # -------------------------------------------------------
    # Clinical Feature Score
    # -------------------------------------------------------
    p_features = float(np.clip(
        sum(
            CLINICAL_FEATURE_WEIGHTS[k] * features[k]
            for k in CLINICAL_FEATURE_WEIGHTS
        ),
        0.0,
        1.0,
    ))

    # -------------------------------------------------------
    # Step 1: OpenCV examines pallor FIRST
    # -------------------------------------------------------
    # -------------------------------------------------------
# Step 1: Calculate Clinical Normality Score
# -------------------------------------------------------

    normal_score = (
    0.45 * (1 - features["pallor"]) +
    0.20 * (1 - features["koilonychia"]) +
    0.13 * (1 - features["ridging"]) +
    0.08 * (1 - features["brittleness"]) +
    0.07 * (1 - features["platonychia"]) +
    0.07 * (1 - features["yellowing"])
    )

    if normal_score > 0.90:
        p_model_anemia *= 0.40

    elif normal_score > 0.80:
        p_model_anemia *= 0.60

    elif normal_score > 0.70:
        p_model_anemia *= 0.80
    # -------------------------------------------------------
    # Step 2: Blend CNN + Clinical Features
    # -------------------------------------------------------
    p_final = float(np.clip(
        MODEL_WEIGHT * p_model_anemia +
        FEATURE_WEIGHT * p_features,
        0.0,
        1.0,
    ))
     
     # Prevent obviously healthy nails from receiving a high anemia score
    if normal_score > 0.90:
        p_final = min(p_final, 0.40)

    elif normal_score > 0.80:
        p_final = min(p_final, 0.55)
    # -------------------------------------------------------
    # Confidence
    # -------------------------------------------------------
    agreement = 1.0 - abs(p_model_anemia - p_features)

    model_decisiveness = abs(p_model_anemia - 0.5) * 2

    active = sum(
        1
        for k in CLINICAL_FEATURE_WEIGHTS
        if features[k] > 0.25
    )

    confidence = float(np.clip(
    0.40
    + 0.25 * model_decisiveness
    + 0.20 * agreement
    + 0.05 * active
    + 0.10 * abs(normal_score - 0.5),
    0.35,
    0.95,
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
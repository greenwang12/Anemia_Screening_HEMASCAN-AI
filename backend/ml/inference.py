"""Inference with Test-Time Augmentation + temperature calibration."""
from __future__ import annotations

import logging
import numpy as np
from PIL import Image

from .config import (
    EYE_ANEMIA_INDEX, EYE_SIGMOID,
    NAIL_ANEMIA_INDEX, NAIL_SIGMOID,
    TEMP_EYE, TEMP_NAIL,
)
from .preprocessing import tta_batch
from .nail_features import (
    analyze_nail_features, combine_nail_signals, CLINICAL_FEATURE_WEIGHTS,
    MODEL_WEIGHT, FEATURE_WEIGHT,
)

logger = logging.getLogger("ml.inference")


def _apply_temperature(p: float, T: float) -> float:
    if T == 1.0:
        return p
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - p))
    return float(1 / (1 + np.exp(-logit / T)))


def _sigmoid_to_p_anemia(preds: np.ndarray, anemia_index: int) -> float:
    """Single-sigmoid output → P(anemia).
    If the model's positive class is "anemia" → anemia_index=0 → return as-is.
    If the positive class is "non_anemia"     → anemia_index=1 → invert.
    """
    p_pos = float(np.mean(preds.reshape(-1)))
    return p_pos if anemia_index == 0 else (1.0 - p_pos)


def predict_eye(model, img: Image.Image) -> dict:
    """Return P(anemia) for the eye image using TTA averaging."""
    batch = tta_batch(img)
    preds = np.asarray(model.predict(batch, verbose=0))

    if EYE_SIGMOID or preds.shape[-1] == 1:
        p_anemia = _sigmoid_to_p_anemia(preds, EYE_ANEMIA_INDEX)
    else:
        probs = preds.mean(axis=0)
        p_anemia = float(probs[EYE_ANEMIA_INDEX])

    p_anemia = _apply_temperature(p_anemia, TEMP_EYE)

    return {
        "p_anemia": p_anemia,
        "confidence": float(min(1.0, abs(p_anemia - 0.5) * 2 + 0.4)),
        "model": "MobileNetV2-binary",
    }


def predict_nail(model, img: Image.Image) -> dict:
    """Binary nail anemia classifier (EfficientNetB0 / MobileNetV2) blended
    with the OpenCV clinical-sign analyzer.

    Pipeline:
      1. Single-sigmoid model with TTA → P_model(anemia).
         (Polarity controlled by NAIL_ANEMIA_INDEX, default = invert.)
      2. Multi-feature OpenCV analyzer → pallor, koilonychia, ridging,
         brittleness, platonychia, yellowing → P_features(anemia).
      3. Blend  60% model  +  40% features  → final P(anemia).
    """
    batch = tta_batch(img)
    preds = np.asarray(model.predict(batch, verbose=0))

    if NAIL_SIGMOID or preds.shape[-1] == 1:
        p_model_anemia = _sigmoid_to_p_anemia(preds, NAIL_ANEMIA_INDEX)
    else:
        # If a future model is multi-class softmax, take the "anemia" column.
        probs = preds.mean(axis=0)
        p_model_anemia = float(probs[NAIL_ANEMIA_INDEX])

    p_model_anemia = _apply_temperature(p_model_anemia, TEMP_NAIL)

    # OpenCV clinical-sign analyzer (drives the "rich story")
    features = analyze_nail_features(img)
    p_anemia, confidence, p_features = combine_nail_signals(features, p_model_anemia)

    return {
        "p_anemia": p_anemia,
        "p_model_anemia": p_model_anemia,
        "p_features_anemia": p_features,
        "p_pallor": features["pallor"],
        "clinical_features": {
            "pallor":      features["pallor"],
            "koilonychia": features["koilonychia"],
            "platonychia": features["platonychia"],
            "ridging":     features["ridging"],
            "brittleness": features["brittleness"],
            "yellowing":   features["yellowing"],
        },
        "feature_weights": CLINICAL_FEATURE_WEIGHTS,
        "blend_weights": {"model": MODEL_WEIGHT, "features": FEATURE_WEIGHT},
        "confidence": confidence,
        "pallor_info": features["_pallor_info"],
        "model": "EfficientNetB0-binary + OpenCV clinical analyzer",
    }


def to_screening_result(
    image_type: str,
    p_anemia: float,
    confidence: float,
    extras: dict,
    attention_regions: list,
    quality: dict,
) -> dict:
    """Shape the output to match the existing screening contract used by the frontend."""
    from .config import risk_label

    pct = int(round(p_anemia * 100))
    findings = []
    if image_type == "nail":
        cf = extras.get("clinical_features") or {}
        signs_present = [k for k, v in cf.items() if v > 0.30]
        if signs_present:
            pretty = {
                "pallor": "nail-bed pallor",
                "koilonychia": "spoon-shape (koilonychia)",
                "ridging": "vertical ridging",
                "brittleness": "brittleness",
                "platonychia": "flattened nail plate",
                "yellowing": "yellow tint",
            }
            findings.append("Clinical signs detected: " + ", ".join(pretty.get(s, s) for s in signs_present))
        else:
            findings.append("No strong anemia-related nail signs detected")
        findings.append(f"Nail-bed pallor: {round(cf.get('pallor', 0)*100)}%")
        if "p_model_anemia" in extras:
            findings.append(f"AI model confidence in anemia: {round(extras['p_model_anemia']*100)}%")
    if image_type == "eye":
        findings.append(f"Conjunctival pallor probability: {pct}%")
        findings.append(f"Risk level: {risk_label(pct)}")
    if quality and not quality.get("passed", True):
        findings.append("⚠ " + " | ".join(quality.get("issues", [])))
    findings.append(f"Overall anemia likelihood: {pct}%")

    reasoning = (
        "We combined an AI model that learned what anemic nails look like with "
        "a separate vision check that measures real clinical signs (pallor, "
        "ridging, koilonychia, brittleness, flatness, yellow tint). The model "
        "gives the overall feel, and the clinical checks tell us why."
        if image_type == "nail" else
        "We analysed the inner lower eyelid for paleness using a deep-learning "
        "model with multi-view averaging for stability. For best accuracy use a "
        "sharp, well-lit photo of the inner lower eyelid."
    )

    return {
        "image_type": image_type,
        "risk_percent": pct,
        "risk_label": risk_label(pct),
        "confidence": confidence,
        "pallor_score": min(10, max(0, round(p_anemia * 10))),
        "key_findings": findings[:4],
        "attention_regions": attention_regions,
        "reasoning": reasoning,
        "model_extras": extras,
        "quality": quality,
    }

"""Inference with Test-Time Augmentation + temperature calibration."""
from __future__ import annotations

import logging
import numpy as np
from PIL import Image

from .config import (
    EYE_ANEMIA_INDEX, EYE_SIGMOID,
    NAIL_ANEMIA_INDEX, NAIL_SIGMOID,
    TEMP_EYE, TEMP_NAIL,
    NAIL_CONFIDENT_ANEMIA_THRESHOLD,
    NAIL_CONFIDENT_NON_ANEMIA_THRESHOLD,
)
from .preprocessing import tta_batch, tta_batch_efficientnet
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
    batch = tta_batch_efficientnet(img)
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
        "model": "EfficientNetB0-binary",
    }


def predict_nail(model, img: Image.Image) -> dict:
    """Binary nail anemia classifier (EfficientNetB0) blended with the
    OpenCV clinical-sign analyzer.

    Pipeline:
      1. EfficientNetB0 with TTA → P_model(anemia).
         Raw [0, 255] input — model has Rescaling baked in.
         Polarity controlled by NAIL_ANEMIA_INDEX (default = invert).
      2. Multi-feature OpenCV analyzer → pallor, koilonychia, ridging,
         brittleness, platonychia, yellowing → P_features(anemia).
      3. If model is highly confident (above NAIL_CONFIDENT_ANEMIA_THRESHOLD
         or below NAIL_CONFIDENT_NON_ANEMIA_THRESHOLD) → trust CNN directly.
         Otherwise blend: 60% model + 40% features.
    """
    batch = tta_batch_efficientnet(img)
    preds = np.asarray(model.predict(batch, verbose=0))

    if NAIL_SIGMOID or preds.shape[-1] == 1:
        p_model_anemia = _sigmoid_to_p_anemia(preds, NAIL_ANEMIA_INDEX)
    else:
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

    pct = int(round(p_anemia * 100))

    # -----------------------------
    # Separate risk labels
    # -----------------------------
    if image_type == "nail":
        if pct < 65:
            label = "Normal / Non-Anemic"
        elif pct < 85:
            label = "Moderate Anemia Risk"
        else:
            label = "High Anemia Risk"

    else:
        from .config import risk_label
        label = risk_label(pct)

    findings = []

    if image_type == "nail":

        cf = extras.get("clinical_features") or {}

        signs_present = [
            k for k, v in cf.items()
            if v > 0.30
        ]

        if signs_present:

            pretty = {
                "pallor": "nail-bed pallor",
                "koilonychia": "spoon-shape (koilonychia)",
                "ridging": "vertical ridging",
                "brittleness": "brittleness",
                "platonychia": "flattened nail plate",
                "yellowing": "yellow tint",
            }

            findings.append(
                "Clinical signs detected: "
                + ", ".join(pretty[s] for s in signs_present)
            )

        else:
            findings.append(
                "No significant anemia-related nail changes detected."
            )

        findings.append(
            f"Nail-bed pallor: {round(cf.get('pallor',0)*100)}%"
        )

        findings.append(
            f"Overall anemia likelihood: {pct}%"
        )

    else:

        findings.append(
            f"Conjunctival pallor probability: {pct}%"
        )

        findings.append(
            f"Risk level: {label}"
        )

    if quality and not quality.get("passed", True):
        findings.append(
            "⚠ " + " | ".join(quality.get("issues", []))
        )

    if image_type == "nail":

        reasoning = (
            "The nail image was analysed using EfficientNetB0 with "
            "Test-Time Augmentation. OpenCV then evaluated clinical "
            "features including nail-bed pallor, koilonychia, ridging, "
            "brittleness, platonychia and yellow discoloration. "
            "The OpenCV findings adjusted the AI prediction before "
            "producing the final hybrid anemia risk."
        )

    else:

        reasoning = (
            "The inner lower eyelid was analysed using EfficientNetB0 "
            "with Test-Time Augmentation. Grad-CAM highlighted the "
            "conjunctival region contributing most to the prediction."
        )

    return {

        "image_type": image_type,

        "risk_percent": pct,

        "risk_label": label,

        "confidence": confidence,

        "pallor_score": min(
            10,
            max(
                0,
                round(p_anemia * 10)
            )
        ),

        "key_findings": findings[:4],

        "attention_regions": attention_regions,

        "reasoning": reasoning,

        "model_extras": extras,

        "quality": quality,
    }
"""End-to-end pipeline entrypoint used by the FastAPI server."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .quality import check_quality
from .preprocessing import decode_image
from .model_loader import get_eye_model, get_nail_model
from .inference import predict_eye, predict_nail, to_screening_result
from .gradcam import gradcam
from .fusion import fuse
from .config import (
    EYE_SIGMOID, EYE_ANEMIA_INDEX,
    NAIL_SIGMOID, NAIL_ANEMIA_INDEX,
)

logger = logging.getLogger("ml.pipeline")


def _analyze_eye_sync(image_b64: str) -> dict:
    quality = check_quality(image_b64)
    img = decode_image(image_b64)
    model = get_eye_model()
    if model is None:
        raise RuntimeError("EYE_MODEL_MISSING")

    pred = predict_eye(model, img)
    is_sigmoid = (EYE_SIGMOID or model.output_shape[-1] == 1)
    cam = gradcam(
        model, img,
        class_index=EYE_ANEMIA_INDEX if not is_sigmoid else None,
        sigmoid_head=is_sigmoid,
        modality="eye",
    )

    result = to_screening_result(
        image_type="eye",
        p_anemia=pred["p_anemia"],
        confidence=pred["confidence"],
        extras={"model": pred["model"]},
        attention_regions=cam["attention_regions"],
        quality=quality,
    )
    result["gradcam_overlay_base64"] = cam["overlay_base64"]
    result["gradcam_heatmap_base64"] = cam["heatmap_base64"]
    result["gradcam_layer"] = cam["layer"]
    return result


def _analyze_nail_sync(image_b64: str) -> dict:
    quality = check_quality(image_b64)
    img = decode_image(image_b64)
    model = get_nail_model()
    if model is None:
        raise RuntimeError("NAIL_MODEL_MISSING")

    pred = predict_nail(model, img)
    is_sigmoid = (NAIL_SIGMOID or model.output_shape[-1] == 1)
    cam = gradcam(
        model, img,
        class_index=NAIL_ANEMIA_INDEX if not is_sigmoid else None,
        sigmoid_head=is_sigmoid,
        modality="nail",
    )

    result = to_screening_result(
        image_type="nail",
        p_anemia=pred["p_anemia"],
        confidence=pred["confidence"],
        extras={
            "p_model_anemia": pred["p_model_anemia"],
            "p_features_anemia": pred["p_features_anemia"],
            "p_pallor": pred["p_pallor"],
            "pallor_info": pred["pallor_info"],
            "clinical_features": pred["clinical_features"],
            "feature_weights": pred["feature_weights"],
            "blend_weights": pred["blend_weights"],
            "model": pred["model"],
        },
        attention_regions=cam["attention_regions"],
        quality=quality,
    )
    result["gradcam_overlay_base64"] = cam["overlay_base64"]
    result["gradcam_heatmap_base64"] = cam["heatmap_base64"]
    result["gradcam_layer"] = cam["layer"]
    return result


async def analyze_eye(image_b64: str) -> Optional[dict]:
    try:
        return await asyncio.to_thread(_analyze_eye_sync, image_b64)
    except RuntimeError as e:
        if str(e) == "EYE_MODEL_MISSING":
            return None
        raise


async def analyze_nail(image_b64: str) -> Optional[dict]:
    try:
        return await asyncio.to_thread(_analyze_nail_sync, image_b64)
    except RuntimeError as e:
        if str(e) == "NAIL_MODEL_MISSING":
            return None
        raise


def fuse_results(eye: dict | None, nail: dict | None) -> dict:
    eye_for_fusion = {"p_anemia": eye["risk_percent"] / 100, "confidence": eye["confidence"]} if eye else None
    nail_for_fusion = {"p_anemia": nail["risk_percent"] / 100, "confidence": nail["confidence"]} if nail else None
    return fuse(eye_for_fusion, nail_for_fusion)

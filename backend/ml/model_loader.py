"""Lazy model loader for the two MobileNetV2 .h5 files."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from .config import EYE_MODEL_PATH, NAIL_MODEL_PATH

logger = logging.getLogger("ml.loader")

_LOCK = threading.Lock()
_EYE_MODEL = None
_NAIL_MODEL = None


def _load(path: Path):
    """Import TF only on demand so the API stays responsive on cold start.
    Returns None if TensorFlow is not installed (production fallback to Gemini).
    """
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        logger.info("TensorFlow not available — using Gemini vision fallback")
        return None
    if not path.exists():
        return None
    try:
        model = load_model(str(path), compile=False)
        logger.info(f"Loaded model from {path.name} — input {model.input_shape}, output {model.output_shape}")
        return model
    except Exception as e:
        logger.exception(f"Failed to load {path}: {e}")
        return None


def get_eye_model():
    global _EYE_MODEL
    with _LOCK:
        if _EYE_MODEL is None:
            _EYE_MODEL = _load(EYE_MODEL_PATH)
    return _EYE_MODEL


def get_nail_model():
    global _NAIL_MODEL
    with _LOCK:
        if _NAIL_MODEL is None:
            _NAIL_MODEL = _load(NAIL_MODEL_PATH)
    return _NAIL_MODEL


def reload_models():
    """Call after uploading new .h5 files via the admin endpoint."""
    global _EYE_MODEL, _NAIL_MODEL
    with _LOCK:
        _EYE_MODEL = None
        _NAIL_MODEL = None
    return {"eye": get_eye_model() is not None, "nail": get_nail_model() is not None}


def models_status() -> dict:
    return {
        "eye_loaded": EYE_MODEL_PATH.exists(),
        "nail_loaded": NAIL_MODEL_PATH.exists(),
        "eye_path": str(EYE_MODEL_PATH),
        "nail_path": str(NAIL_MODEL_PATH),
    }

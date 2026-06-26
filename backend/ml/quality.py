"""Quick image quality checks (blur, brightness)."""
from __future__ import annotations

import io
import base64
import numpy as np
from PIL import Image

from .config import MIN_BLUR_VARIANCE, MIN_BRIGHTNESS, MAX_BRIGHTNESS


def _decode(image_b64: str) -> Image.Image:
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian — proxy for image sharpness."""
    pad = np.pad(gray.astype(np.float32), 1, mode="edge")
    lap = (
        pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        - 4 * pad[1:-1, 1:-1]
    )
    return float(lap.var())


def check_quality(image_b64: str) -> dict:
    """Return a quality report. `passed=False` means we should warn the user."""
    img = _decode(image_b64)
    arr = np.asarray(img)
    gray = arr.mean(axis=2)

    blur = laplacian_variance(gray)
    brightness = float(gray.mean())

    issues = []
    if blur < MIN_BLUR_VARIANCE:
        issues.append("Image is too blurry — please retake with steadier hands.")
    if brightness < MIN_BRIGHTNESS:
        issues.append("Image is too dark — please use better lighting.")
    if brightness > MAX_BRIGHTNESS:
        issues.append("Image is over-exposed — reduce lighting glare.")

    return {
        "passed": len(issues) == 0,
        "blur_variance": round(blur, 1),
        "brightness": round(brightness, 1),
        "issues": issues,
    }

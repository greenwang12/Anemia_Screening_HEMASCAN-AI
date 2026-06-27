"""Image preprocessing and Test-Time Augmentation (TTA)."""
from __future__ import annotations

import io
import base64
import numpy as np
from PIL import Image

from .config import IMG_SIZE


def decode_image(image_b64: str) -> Image.Image:
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """MobileNetV2 expects RGB scaled to [-1, 1]."""
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr  # (H, W, 3)


def preprocess_efficientnet(img: Image.Image) -> np.ndarray:
    """EfficientNetB0 has a Rescaling layer baked in — pass raw [0, 255]."""
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return arr  # (H, W, 3) — NO scaling


def _tta_variants(img: Image.Image) -> list[Image.Image]:
    """Shared TTA augmentation variants (used by both batch functions)."""
    variants = [img]
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    for angle in (-10, 10):
        variants.append(img.rotate(angle, resample=Image.BILINEAR))
    # Center-crop zoom (90%)
    w, h = img.size
    pad = int(min(w, h) * 0.05)
    variants.append(img.crop((pad, pad, w - pad, h - pad)))
    return variants


def tta_batch(img: Image.Image) -> np.ndarray:
    """TTA batch for MobileNetV2 — scaled to [-1, 1]."""
    batch = np.stack([preprocess(v) for v in _tta_variants(img)], axis=0)
    return batch  # (N, H, W, 3)


def tta_batch_efficientnet(img: Image.Image) -> np.ndarray:
    """TTA batch for EfficientNetB0 — raw [0, 255], model handles rescaling."""
    batch = np.stack([preprocess_efficientnet(v) for v in _tta_variants(img)], axis=0)
    return batch  # (N, H, W, 3)
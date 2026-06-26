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
    """MobileNetV2 expects RGB / scaled to [-1, 1]."""
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr  # (H, W, 3)


def tta_batch(img: Image.Image) -> np.ndarray:
    """Build a small TTA batch: original + h-flip + small rotations + center-crop zoom."""
    variants = [img]
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    for angle in (-10, 10):
        variants.append(img.rotate(angle, resample=Image.BILINEAR))
    # Center-crop zoom (90%)
    w, h = img.size
    pad = int(min(w, h) * 0.05)
    variants.append(img.crop((pad, pad, w - pad, h - pad)))

    batch = np.stack([preprocess(v) for v in variants], axis=0)
    return batch  # (N, H, W, 3)

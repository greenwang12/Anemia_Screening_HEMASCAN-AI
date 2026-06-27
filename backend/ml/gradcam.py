"""Real Grad-CAM implementation using tf.GradientTape."""
from __future__ import annotations

import io
import base64
import logging
import numpy as np
from PIL import Image

from .config import IMG_SIZE, GRADCAM_LAYER
from .pallor import tissue_mask

logger = logging.getLogger("ml.gradcam")


def _iris_anchored_conjunctiva_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Detect iris position and place a conjunctiva ellipse just below it.

    Works at any input resolution — all measurements are relative to the
    detected iris radius, not image pixel coordinates.

    Fallback chain:
      1. HoughCircles finds iris → place ellipse below iris bottom edge
      2. No iris found → use lower-third center band (generic safe guess)
      3. Exception → return uniform ones (no masking)
    """
    H, W = img_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)

    try:
        import cv2

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)

        min_r = max(8,  min(H, W) // 12)
        max_r = max(min_r + 4, min(H, W) // 3)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(20, min(H, W) // 4),
            param1=80, param2=22,
            minRadius=min_r, maxRadius=max_r,
        )

        if circles is not None and len(circles[0]) > 0:
            # Pick the darkest circle — iris is the darkest region
            best, best_score = None, 1e9
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                y0 = max(0, cy - r // 2);  y1 = min(H, cy + r // 2)
                x0 = max(0, cx - r // 2);  x1 = min(W, cx + r // 2)
                if y1 <= y0 or x1 <= x0:
                    continue
                score = float(gray[y0:y1, x0:x1].mean())
                if score < best_score:
                    best, best_score = (cx, cy, r), score

            if best is not None:
                iris_cx, iris_cy, iris_r = best
                # Conjunctiva sits just below the iris bottom
                # Place ellipse lower on the palpebral conjunctiva
                conj_cy = iris_cy + int(iris_r * 1.50)  # below iris
                conj_cx = iris_cx                          # same horizontal centre
                # Slightly wider and thinner ellipse
                ax_x = int(iris_r * 2.35)
                ax_y = int(iris_r * 0.50)              # shallow strip
                # Clamp to image bounds
                conj_cy = min(H - ax_y - 2, max(ax_y + 2, conj_cy))
                ax_x    = min(ax_x, conj_cx, W - conj_cx)
                if ax_x > 4 and ax_y > 2:
                    cv2.ellipse(
                        mask, (conj_cx, conj_cy), (ax_x, ax_y),
                        angle=0, startAngle=0, endAngle=360,
                        color=1.0, thickness=-1,
                    )
                    sigma = max(4.0, iris_r * 0.25)
                    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
                    if mask.max() > 0:
                        mask /= mask.max()
                    return mask.astype(np.float32)

        # Fallback: lower-third center band (no iris found)
        logger.info("Grad-CAM eye: no iris found, using lower-third fallback")
        roi_h = int(H * 0.22)
        roi_w = int(W * 0.70)
        y0 = int(H * 0.58);  y1 = min(H, y0 + roi_h)
        x0 = (W - roi_w) // 2;  x1 = x0 + roi_w
        mask[y0:y1, x0:x1] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(6.0, W / 25.0))
        if mask.max() > 0:
            mask /= mask.max()
        return mask.astype(np.float32)

    except Exception as e:
        logger.warning("conjunctiva mask failed: %s — returning uniform mask", e)
        return np.ones((H, W), dtype=np.float32)


def _nail_plate_mask(img_rgb: np.ndarray) -> np.ndarray | None:
    """Skin/nail HSV mask — suppresses background and non-tissue regions.

    Pure numpy: no OpenCV connected components needed.
    Returns float32 [0,1] mask same size as img_rgb, or None if no tissue found.
    """
    H, W = img_rgb.shape[:2]

    r = img_rgb[..., 0] / 255.0
    g = img_rgb[..., 1] / 255.0
    b = img_rgb[..., 2] / 255.0

    v = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = v - mn

    s = np.where(v > 0, delta / v, 0.0)

    safe_delta = np.where(delta > 0, delta, 1.0)
    h_raw = np.where(
        v == r, (g - b) / safe_delta % 6,
        np.where(v == g, (b - r) / safe_delta + 2,
                          (r - g) / safe_delta + 4)
    )
    h_norm = np.where(delta > 0, h_raw / 6.0, 0.0) % 1.0

    # Skin hue: near red (0°/360°) — distance from the 0/1 boundary
    hue_dist = np.minimum(h_norm, 1.0 - h_norm)

    # Skin pixels: reddish-pink hue, moderate saturation, non-shadow brightness
    skin = (hue_dist < 0.13) & (s > 0.08) & (s < 0.85) & (v > 0.25)

    # Also explicitly exclude very low saturation + bright regions = background/wall
    background = (s < 0.06) & (v > 0.70)
    skin = skin & ~background

    if skin.sum() < 0.02 * H * W:
        return None

    # Nail plate: bright + low-saturation within skin region
    bright     = np.clip((v - 0.35) / 0.50, 0.0, 1.0)
    pale       = np.clip((0.55 - s) / 0.45, 0.0, 1.0)
    nail_weight = skin.astype(np.float32) * (0.55 * bright + 0.45 * pale)

    ksize  = max(3, min(H, W) // 50) | 1
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    try:
        from scipy.ndimage import convolve
        nail_weight = convolve(nail_weight, kernel)
        if nail_weight.max() > 0:
            nail_weight /= nail_weight.max()
        mask = np.clip(0.30 * skin.astype(np.float32) + 0.70 * nail_weight, 0.0, 1.0)
        mask = convolve(mask, kernel)
    except Exception:
        mask = np.clip(0.30 * skin.astype(np.float32) + 0.70 * nail_weight, 0.0, 1.0)

    if mask.max() > 0:
        mask /= mask.max()

    return mask.astype(np.float32)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _to_attention_regions(heatmap: np.ndarray, k: int = 3) -> list:
    h, w = heatmap.shape
    if heatmap.max() <= 0:
        return []
    grid = 12
    cell_h = max(1, h // grid)
    cell_w = max(1, w // grid)
    peaks = []
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * cell_h, min(h, (i + 1) * cell_h)
            x0, x1 = j * cell_w, min(w, (j + 1) * cell_w)
            patch = heatmap[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            v = float(patch.max())
            ly, lx = np.unravel_index(np.argmax(patch), patch.shape)
            peaks.append((v, y0 + ly, x0 + lx))
    peaks.sort(reverse=True)
    return [
        {"cx": round(x / w, 3), "cy": round(y / h, 3),
         "radius": 0.18, "intensity": round(min(1.0, v), 3)}
        for v, y, x in peaks[:k]
    ]


def gradcam(model, image: Image.Image, class_index: int | None = None,
            layer_name: str | None = None, sigmoid_head: bool = False,
            modality: str = "nail", preprocess: str = "efficientnet") -> dict:
    """Compute Grad-CAM for `image` against `model`.

    Args:
        model:        Keras model.
        image:        PIL Image (any size — resized internally to IMG_SIZE).
        class_index:  Output index to take gradient of (None = top class).
        layer_name:   Last conv layer name. Defaults to config.GRADCAM_LAYER.
        sigmoid_head: True if the head is a single sigmoid neuron.
        modality:     "nail" or "eye" — controls spatial masking strategy.
        preprocess:   "efficientnet" → raw [0,255]; "mobilenet" → [-1,1].
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("TensorFlow required for Grad-CAM") from e

    layer_name = layer_name or GRADCAM_LAYER

    def _4d(layer):
        try:
            return len(layer.output.shape) == 4
        except Exception:
            return False

    # Layer resolution: configured name → known fallbacks → nested submodel → last 4D
    target_layer = None
    for name in [layer_name, "top_conv", "top_activation", "out_relu", "Conv_1", "Conv_1_bn"]:
        try:
            cand = model.get_layer(name)
            if _4d(cand):
                target_layer = cand
                logger.info("Grad-CAM: using layer '%s'", name)
                break
        except ValueError:
            continue

    if target_layer is None:
        for layer in reversed(model.layers):
            inner = getattr(layer, "layers", None)
            if not inner or not _4d(layer):
                continue
            aug_only = all(
                type(s).__name__.startswith("Random") or
                type(s).__name__ in {"Rescaling", "Normalization"}
                for s in inner
            )
            if not aug_only:
                target_layer = layer
                logger.info("Grad-CAM: using nested submodel '%s'", layer.name)
                break

    if target_layer is None:
        for layer in reversed(model.layers):
            if _4d(layer):
                target_layer = layer
                logger.info("Grad-CAM: last-resort layer '%s'", layer.name)
                break

    if target_layer is None:
        raise RuntimeError("No 4-D conv layer found for Grad-CAM.")

    # Build grad_model
    is_submodel = bool(getattr(target_layer, "layers", []))
    grad_model  = None

    if not is_submodel:
        try:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )
        except Exception:
            pass

    if grad_model is None:
        sub_input = tf.keras.Input(shape=model.input_shape[1:])
        conv_out_sym = None
        x = sub_input
        if is_submodel:
            x = target_layer(sub_input, training=False)
            conv_out_sym = x
            seen = False
            for layer in model.layers:
                if not seen:
                    seen = (layer is target_layer)
                    continue
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)
        else:
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)
                if layer is target_layer:
                    conv_out_sym = x
        if conv_out_sym is None:
            raise RuntimeError("Grad-CAM: target layer not visited.")
        grad_model = tf.keras.Model(inputs=sub_input, outputs=[conv_out_sym, x])

    # Preprocess input
    img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    if preprocess == "efficientnet":
    # Model already contains a Rescaling layer
        arr = np.asarray(img_resized, dtype=np.float32)
    else:
    # MobileNetV2 preprocessing
        arr = (np.asarray(img_resized, dtype=np.float32) / 127.5) - 1.0

    arr = np.expand_dims(arr, axis=0)

    # GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr, training=False)
        if sigmoid_head or predictions.shape[-1] == 1:
            target = predictions[:, 0]
        else:
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            target = predictions[:, class_index]

    grads  = tape.gradient(target, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    fmap   = conv_outputs[0].numpy()

    cam = np.einsum("hwc,c->hw", fmap, pooled)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= (cam.max() + 1e-8)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    cam_np = np.asarray(cam_img, dtype=np.float32) / 255.0

    # ── Spatial masking ───────────────────────────────────────────────────────
    img_rgb_np = np.asarray(img_resized.convert("RGB"))

    if modality == "eye":
        # Iris-anchored ellipse → scales with any image size/framing
        try:
            conj_mask = _iris_anchored_conjunctiva_mask(img_rgb_np)
            # Soft attention weighting instead of hard clipping
            cam_np = cam_np * (0.30 + 0.70 * conj_mask)

            cam_np = np.clip(cam_np, 0.0, 1.0)

            if cam_np.max() > 0:
                cam_np /= (cam_np.max() + 1e-8)
        except Exception as e:
            logger.warning("eye mask failed: %s", e)

    else:
        # Nail: skin HSV mask suppresses background/wall hotspots
        try:
            nail_mask = _nail_plate_mask(img_rgb_np)
            if nail_mask is None:
                nail_mask = tissue_mask(img_resized, size=(IMG_SIZE, IMG_SIZE))
                gated = cam_np * (0.30 + 0.70 * nail_mask)
            else:
                gated = cam_np * nail_mask
                if gated.max() < 0.05:
                    gated = nail_mask * float(np.clip(cam_np.mean() * 3.0, 0.25, 1.0))
            gated = np.power(np.clip(gated, 0.0, 1.0), 0.6)
            if gated.max() > 0:
                gated /= (gated.max() + 1e-8)
            cam_np = gated
        except Exception as e:
            logger.warning("nail mask failed: %s", e)

    # ── Overlay ───────────────────────────────────────────────────────────────
    alpha_gain = 1.65 if modality == "eye" else 1.35
    rgb   = (_jet_colormap(cam_np) * 255).astype(np.uint8)
    alpha = (np.clip(cam_np * alpha_gain, 0.0, 1.0) * 235).astype(np.uint8)
    rgba  = np.concatenate([rgb, alpha[..., None]], axis=-1)
    overlay  = Image.fromarray(rgba, mode="RGBA")
    composed = Image.alpha_composite(img_resized.convert("RGBA"), overlay).convert("RGB")

    def to_b64(img: Image.Image, fmt="PNG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "heatmap_base64":    to_b64(overlay,  fmt="PNG"),
        "overlay_base64":    to_b64(composed, fmt="JPEG"),
        "attention_regions": _to_attention_regions(cam_np, k=3),
        "layer":             target_layer.name,
    }
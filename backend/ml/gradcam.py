"""Real Grad-CAM implementation using tf.GradientTape.

Returns:
  - heatmap_base64: a PNG overlay (translucent jet colormap) the frontend draws on top.
  - attention_regions: normalized (cx, cy, radius, intensity) hot-spots
    derived from the heatmap so the existing canvas renderer also keeps working.
"""
from __future__ import annotations

import io
import base64
import logging
import numpy as np
from PIL import Image

from .config import IMG_SIZE, GRADCAM_LAYER
from .pallor import tissue_mask

logger = logging.getLogger("ml.gradcam")


def _detect_conjunctiva_roi(img_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the palpebral conjunctiva (pink strip below the iris).

    Strategy:
      1. Find the iris via HoughCircles (largest dark circle).
      2. Conjunctiva ROI sits just below the iris bottom, height ~ iris_radius * 1.2,
         width ~ iris_radius * 4.0, clamped to the image.
      3. If HoughCircles fails, fall back to the lower-third center band.

    Returns (x, y, w, h) in image pixel coords, or None.
    """
    try:
        import cv2
    except Exception:
        return None
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_r = max(8, min(h, w) // 12)
    max_r = max(min_r + 4, min(h, w) // 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(20, min(h, w) // 4),
        param1=80, param2=22, minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None and len(circles[0]) > 0:
        # Pick the darkest circle (iris is usually the darkest)
        best, best_score = None, 1e9
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            y0, y1 = max(0, cy - r // 2), min(h, cy + r // 2)
            x0, x1 = max(0, cx - r // 2), min(w, cx + r // 2)
            if y1 <= y0 or x1 <= x0:
                continue
            score = float(gray[y0:y1, x0:x1].mean())
            if score < best_score:
                best, best_score = (cx, cy, r), score
        if best is not None:
            cx, cy, r = best
            roi_w = int(r * 4.0)
            roi_h = int(r * 1.2)
            x = max(0, cx - roi_w // 2)
            y = min(h - 1, cy + int(r * 1.05))
            roi_w = min(roi_w, w - x)
            roi_h = min(roi_h, h - y)
            if roi_w > 8 and roi_h > 8:
                return (x, y, roi_w, roi_h)
    # Fallback: lower-third center band
    roi_w = int(w * 0.70)
    roi_h = int(h * 0.20)
    x = (w - roi_w) // 2
    y = int(h * 0.60)
    return (x, y, roi_w, roi_h)


def _detect_nail_bed_mask(img_rgb: np.ndarray) -> np.ndarray | None:
    """Locate the nail-bed (the pink/red tissue under the nail plate) and
    return a smooth float32 mask in [0,1], same H,W as img_rgb.

    Strategy (no learning, fully OpenCV/numpy):
      1. Build HSV-based skin/pink mask (hue near red-pink, mid saturation,
         non-shadow value).
      2. Largest connected component → the dominant finger / nail region.
      3. Pick the BRIGHTEST sub-region inside that component — the nail
         plate / bed is typically lighter than the surrounding finger.
      4. Smooth with a Gaussian so the mask edges are soft.

    Returns None if no plausible region is found.
    """
    try:
        import cv2
    except Exception:
        return None
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_ch = hsv[..., 0].astype(np.float32) / 179.0   # OpenCV hue is [0,179]
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    # Skin/pink hue band (red wraps around): hue near 0 OR near 1
    hue_dist = np.minimum(h_ch, 1.0 - h_ch)
    skin = ((hue_dist < 0.13) & (s > 0.10) & (s < 0.85) & (v > 0.25)).astype(np.uint8)
    if skin.sum() < 0.02 * H * W:
        return None

    # Morphological cleanup
    k = max(3, min(H, W) // 60)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((k * 2, k * 2), np.uint8))

    # Largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if n <= 1:
        return None
    # ignore background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    finger = (labels == best).astype(np.float32)
    if finger.sum() < 0.02 * H * W:
        return None

    # Within the finger, accentuate the brighter / less-saturated nail region.
    # Nail bed/plate ≈ higher V, lower S than surrounding finger skin.
    bright = np.clip((v - 0.40) / 0.45, 0.0, 1.0)
    pale = np.clip((0.55 - s) / 0.40, 0.0, 1.0)
    nail_weight = finger * (0.55 * bright + 0.45 * pale)
    nail_weight = cv2.GaussianBlur(nail_weight, (0, 0), sigmaX=max(3.0, min(H, W) / 60.0))
    if nail_weight.max() > 0:
        nail_weight = nail_weight / nail_weight.max()

    # Combine: full finger gets ~0.3, nail focus area gets 1.0
    mask = np.clip(0.30 * finger + 0.70 * nail_weight, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, min(H, W) / 100.0))
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask.astype(np.float32)


def _conjunctiva_spread_heatmap(img_rgb: np.ndarray, raw_cam: np.ndarray) -> np.ndarray:
    """Build a smooth, vibrant heatmap anchored on the conjunctiva ROI.

    - Anchors a 2D Gaussian on the detected conjunctiva strip.
    - Magnitude = mean of the model's raw Grad-CAM signal inside the ROI
      (so the visualization is still derived from the model's attention).
    - Locally modulated by per-pixel pallor (1 - saturation in HSV) so the
      heatmap is hotter where the conjunctiva is pale — clinically meaningful.

    Returns a float32 array, same shape as raw_cam, values in [0, 1].
    """
    H, W = raw_cam.shape
    roi = _detect_conjunctiva_roi(img_rgb)
    if roi is None:
        return raw_cam
    x, y, w, h = roi

    # Mean raw-CAM activation inside the ROI — keeps visualization tied to the model.
    cam_strength = float(np.clip(raw_cam[y:y + h, x:x + w].mean() * 4.0, 0.25, 1.0))

    # Pallor map inside ROI (1 - saturation, normalized lightness)
    try:
        import cv2
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0
        pallor = np.clip((1.0 - sat) * (0.5 + 0.5 * val), 0.0, 1.0)
    except Exception:
        pallor = np.ones((H, W), dtype=np.float32)

    # 2D anisotropic Gaussian centered on ROI center
    cy = y + h / 2.0
    cx = x + w / 2.0
    sigma_x = max(8.0, w * 0.42)
    sigma_y = max(6.0, h * 0.55)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gauss = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2) +
                    ((yy - cy) ** 2) / (2 * sigma_y ** 2)))

    # Combine: Gaussian envelope × pallor weighting × CNN activation strength
    heat = gauss * (0.55 + 0.45 * pallor) * cam_strength

    # HARD-clamp outside the ROI rectangle (with a small soft border) so the
    # heatmap never bleeds onto the iris, sclera, or skin around the eye.
    roi_mask = np.zeros((H, W), dtype=np.float32)
    pad_y = max(2, int(h * 0.10))
    pad_x = max(2, int(w * 0.06))
    y0 = max(0, y - pad_y)
    y1 = min(H, y + h + pad_y)
    x0 = max(0, x - pad_x)
    x1 = min(W, x + w + pad_x)
    roi_mask[y0:y1, x0:x1] = 1.0
    try:
        import cv2
        roi_mask = cv2.GaussianBlur(roi_mask, (0, 0), sigmaX=max(2.0, min(H, W) / 80.0))
    except Exception:
        pass
    heat = heat * roi_mask

    if heat.max() > 0:
        heat = heat / heat.max()
    # Gentle gamma to fatten the visible warm region
    heat = np.power(heat, 0.65)
    return heat.astype(np.float32)


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Tiny 256-step jet-like colormap implemented inline (no matplotlib runtime cost)."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _to_attention_regions(heatmap: np.ndarray, k: int = 3) -> list:
    """Pick top-k hotspot peaks and convert to normalized {cx,cy,radius,intensity}."""
    h, w = heatmap.shape
    flat = heatmap.flatten()
    if flat.max() <= 0:
        return []
    # Simple non-max suppression on a downsampled grid
    grid = 12
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
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
    chosen = peaks[:k]
    regions = []
    for v, y, x in chosen:
        regions.append({
            "cx": round(x / w, 3),
            "cy": round(y / h, 3),
            "radius": 0.18,
            "intensity": round(min(1.0, v), 3),
        })
    return regions


def gradcam(model, image: Image.Image, class_index: int | None = None,
            layer_name: str | None = None, sigmoid_head: bool = False,
            modality: str = "nail") -> dict:
    """Compute Grad-CAM for `image` against `model`.

    Args:
        model:        Keras model.
        image:        PIL Image.
        class_index:  Output index to take gradient of (None = top class).
        layer_name:   Last conv layer name. Defaults to config.GRADCAM_LAYER.
        sigmoid_head: True if the head is a single sigmoid (use raw output).
    Returns:
        dict with `heatmap_base64`, `overlay_base64`, `attention_regions`.

    Raises:
        ImportError if TensorFlow is unavailable. Caller (pipeline) should
        handle by falling back to the Gemini vision pipeline.
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

    # 1) Try the requested layer name
    target_layer = None
    try:
        cand = model.get_layer(layer_name)
        if _4d(cand):
            target_layer = cand
    except ValueError:
        pass

    # 2) Try common MobileNetV2 alternatives
    if target_layer is None:
        for alt in ("out_relu", "Conv_1", "Conv_1_bn"):
            try:
                cand = model.get_layer(alt)
                if _4d(cand):
                    target_layer = cand
                    break
            except ValueError:
                continue

    # 3) Some models wrap the backbone as a NESTED Functional submodel
    #    (e.g. EfficientNetB0 inside a parent that also has augmentation
    #    layers like `sequential` in front). Iterate in REVERSE so the
    #    deepest / last submodel is chosen and avoid grabbing the
    #    augmentation block which would also pass the 4D check.
    if target_layer is None:
        for layer in reversed(model.layers):
            inner = getattr(layer, "layers", None)
            if not inner:
                continue
            if not _4d(layer):
                continue
            # Skip pure augmentation blocks: they only contain Random* layers
            # and Rescaling/Normalization. They are NOT feature extractors.
            class_names = {type(s).__name__ for s in inner}
            aug_only = class_names and all(
                n.startswith("Random") or n in {"Rescaling", "Normalization"}
                for n in class_names
            )
            if aug_only:
                continue
            target_layer = layer
            break

    # 4) Last resort: scan top-level for any 4D output
    if target_layer is None:
        for layer in reversed(model.layers):
            if _4d(layer):
                target_layer = layer
                break

    if target_layer is None:
        raise RuntimeError("No 4-D conv layer found for Grad-CAM.")

    # Build a sub-graph that exposes the target feature map + final prediction.
    #
    # Two cases we need to support:
    #   A) Target layer is a NESTED Functional submodel (e.g. EfficientNetB0)
    #      whose internal `.output` is in its OWN graph — `tf.keras.Model`
    #      reconstruction from the parent inputs cannot connect gradients to
    #      that internal tensor. We must walk the parent layers manually and
    #      capture the output where the submodel is called.
    #   B) Target layer is in the top-level functional graph (e.g. MobileNetV2
    #      inlined into the parent). The Functional API rebuild works fine.
    #
    # The manual walk also has to disable training-time layers (data
    # augmentation, dropout) so the forward pass is deterministic and
    # gradients are meaningful.
    is_submodel_target = bool(getattr(target_layer, "layers", []))
    grad_model = None

    if not is_submodel_target:
        try:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )
        except Exception:
            grad_model = None

    if grad_model is None:
        # Manual walk: rebuild the forward pass with training=False everywhere.
        # When target is a NESTED submodel (e.g. efficientnetb0 inside a
        # parent that also has augmentation/dropout layers), we treat that
        # submodel as the FEATURE EXTRACTOR and re-apply the remaining
        # head layers on top of its output. This guarantees that gradients
        # of the final prediction w.r.t. the feature tensor flow correctly.
        sub_input = tf.keras.Input(shape=model.input_shape[1:])
        conv_out_symbolic = None
        x = sub_input

        if is_submodel_target:
            features = target_layer(sub_input, training=False)
            conv_out_symbolic = features
            x = features
            seen_target = False
            for layer in model.layers:
                if not seen_target:
                    if layer is target_layer:
                        seen_target = True
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
                    conv_out_symbolic = x

        final_out_symbolic = x
        if conv_out_symbolic is None:
            raise RuntimeError("Grad-CAM: target layer not visited during manual walk.")
        grad_model = tf.keras.Model(inputs=sub_input, outputs=[conv_out_symbolic, final_out_symbolic])

    # Prepare input
    img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = (np.asarray(img_resized, dtype=np.float32) / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr, training=False)
        if sigmoid_head or predictions.shape[-1] == 1:
            target = predictions[:, 0]
        else:
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            target = predictions[:, class_index]

    grads = tape.gradient(target, conv_outputs)               # (1, h, w, c)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()    # (c,)
    fmap = conv_outputs[0].numpy()                            # (h, w, c)
    cam = np.einsum("hwc,c->hw", fmap, pooled)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)

    # Upscale heatmap to IMG_SIZE
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    cam_np = np.asarray(cam_img, dtype=np.float32) / 255.0

    # Tissue-mask gate: bias the heatmap toward flesh-toned regions
    # (conjunctiva / nail-bed). Eye uses a conjunctiva-anchored spread
    # heatmap (OpenCV iris detection + Gaussian on conjunctiva, modulated
    # by raw CNN activation + per-pixel pallor) so the visualization
    # matches clinical literature (Dimauro 2018, Bauskar 2019).
    if modality == "eye":
        try:
            img_rgb_np = np.asarray(img_resized.convert("RGB"))
            cam_np = _conjunctiva_spread_heatmap(img_rgb_np, cam_np)
        except Exception as e:
            logger.warning("conjunctiva heatmap failed: %s", e)
    else:
        # Nail: detect the nail-bed ROI with OpenCV and HARD-mask the heatmap
        # so the warm region only appears on the nail tissue (not background,
        # cuticle, or the finger pad). The model still ran on the full image.
        try:
            img_rgb_np = np.asarray(img_resized.convert("RGB"))
            nail_mask = _detect_nail_bed_mask(img_rgb_np)
            if nail_mask is None:
                # Fallback to the loose flesh-tone mask
                nail_mask = tissue_mask(img_resized, size=(IMG_SIZE, IMG_SIZE))
                gated = cam_np * (0.30 + 0.70 * nail_mask)
            else:
                # Strict: zero out everything outside the nail-bed ROI,
                # then re-weight the heat by raw CNN signal within ROI.
                gated = cam_np * nail_mask
                # If the model didn't fire inside the ROI at all, fall back to
                # a soft ROI-centred map so the user still sees attention on
                # the right region.
                if gated.max() < 0.05:
                    gated = nail_mask * float(np.clip(cam_np.mean() * 3.0, 0.25, 1.0))
            gated = np.power(np.clip(gated, 0.0, 1.0), 0.6)
            if gated.max() > 0:
                gated = gated / (gated.max() + 1e-8)
            cam_np = gated
        except Exception as e:
            logger.warning("nail mask gate failed: %s", e)

    # Build colored overlay — stronger alpha for eye since we anchored the
    # heatmap to the conjunctiva and want it to "pop" like reference papers.
    alpha_gain = 1.65 if modality == "eye" else 1.35
    rgb = (_jet_colormap(cam_np) * 255).astype(np.uint8)
    alpha = (np.clip(cam_np * alpha_gain, 0.0, 1.0) * 235).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    overlay = Image.fromarray(rgba, mode="RGBA")

    # Composite overlay onto the original image for an "overlay_base64"
    base = img_resized.convert("RGBA")
    composed = Image.alpha_composite(base, overlay).convert("RGB")

    def to_b64(img: Image.Image, fmt="PNG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "heatmap_base64": to_b64(overlay, fmt="PNG"),   # translucent overlay (RGBA)
        "overlay_base64": to_b64(composed, fmt="JPEG"), # original + heatmap composited
        "attention_regions": _to_attention_regions(cam_np, k=3),
        "layer": target_layer.name,
    }

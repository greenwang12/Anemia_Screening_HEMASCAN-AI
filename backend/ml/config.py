"""Static configuration for the ML pipeline."""
import os
from pathlib import Path

# Paths
ML_DIR = Path(__file__).parent
MODELS_DIR = ML_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_EYE_KERAS = MODELS_DIR / "eye_mobilenetv2.keras"
_EYE_H5 = MODELS_DIR / "eye_mobilenetv2.h5"
EYE_MODEL_PATH = _EYE_KERAS if _EYE_KERAS.exists() else _EYE_H5

# New binary nail model (EfficientNetB0). Falls back to legacy .h5 if missing.
_NAIL_KERAS = MODELS_DIR / "nail_efficientnetb0.keras"
_NAIL_H5 = MODELS_DIR / "nail_mobilenetv2.h5"
NAIL_MODEL_PATH = _NAIL_KERAS if _NAIL_KERAS.exists() else _NAIL_H5

FUSION_META_PATH = MODELS_DIR / "fusion_meta.npz"   # optional sklearn weights

# Image input size (EfficientNetB0 / MobileNetV2 both use 224)
IMG_SIZE = 224

# ---- Eye binary head ----
# Many Keras ImageDataGenerator setups go alphabetical; "anemia" sorts before
# "non_anemia", so the sigmoid head typically outputs P(anemia).
EYE_ANEMIA_INDEX = 0     # 0 = anemia, 1 = non_anemia
EYE_SIGMOID = True       # True if single-sigmoid output

# ---- Nail binary head ----
# IMPORTANT polarity note:
#   With label 0 = "anemic" and label 1 = "non_anemic", a single-sigmoid
#   classifier trained with binary_crossentropy outputs P(class == 1)
#   = P(non_anemic). So P(anemia) = 1 - sigmoid_output.
# NAIL_ANEMIA_INDEX = 1 means "the sigmoid is P(non-anemia), invert it".
NAIL_ANEMIA_INDEX = int(os.environ.get("NAIL_ANEMIA_INDEX", "1"))
NAIL_SIGMOID = True

# Grad-CAM target layer
GRADCAM_LAYER = "Conv_1"   # MobileNetV2; auto-fallback handles EfficientNetB0

# Confidence calibration (temperature scaling). 1.0 = no calibration.
TEMP_EYE = 1.0
TEMP_NAIL = 1.0

# Quality thresholds
MIN_BLUR_VARIANCE = 60.0
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 230


# Risk thresholds (anemia probability % -> label)
def risk_label(pct: int) -> str:
    if pct < 35:
        return "Low"
    if pct <= 65:
        return "Moderate"
    return "High"

"""Static configuration for the ML pipeline."""
import os
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

ML_DIR = Path(__file__).parent
MODELS_DIR = ML_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Model Paths
# ==========================================================

EYE_MODEL_PATH = MODELS_DIR / "best_eye_model.keras"
NAIL_MODEL_PATH = MODELS_DIR / "efficientnetb0_nail_final.keras"
FUSION_META_PATH = MODELS_DIR / "fusion_meta.npz"

# ==========================================================
# Image Configuration
# ==========================================================

IMG_SIZE = 224

# ==========================================================
# Eye Model Configuration
# ==========================================================

# Eye model outputs P(Anemia)
EYE_ANEMIA_INDEX = int(os.environ.get("EYE_ANEMIA_INDEX", "0"))
EYE_SIGMOID = True

# ==========================================================
# Nail Model Configuration
# ==========================================================

# Keras folder names:
#   Anemic/
#   Non_Anemic/
#
# Sigmoid output corresponds to Non_Anemic,
# therefore invert prediction to obtain P(Anemia).

NAIL_ANEMIA_INDEX = int(os.environ.get("NAIL_ANEMIA_INDEX", "1"))
NAIL_SIGMOID = True

# ==========================================================
# CNN Confidence Thresholds
# ==========================================================

# CNN prediction confidence

NAIL_CONFIDENT_ANEMIA_THRESHOLD = 0.75
NAIL_CONFIDENT_NON_ANEMIA_THRESHOLD = 0.25

# ==========================================================
# OpenCV Clinical Thresholds
# ==========================================================

# Nail-bed pallor thresholds

NORMAL_PALLOR_THRESHOLD = 0.25
MILD_PALLOR_THRESHOLD = 0.40
MODERATE_PALLOR_THRESHOLD = 0.55

# ==========================================================
# Final Hybrid Decision Threshold
# ==========================================================

# Final blended probability

ANEMIA_THRESHOLD = 0.65

# ==========================================================
# Grad-CAM
# ==========================================================

# Last convolution layer of EfficientNetB0

GRADCAM_LAYER = "top_conv"

# ==========================================================
# Temperature Calibration
# ==========================================================

TEMP_EYE = 1.0
TEMP_NAIL = 1.0

# ==========================================================
# Image Quality
# ==========================================================

MIN_BLUR_VARIANCE = 60.0
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 230

# ==========================================================
# Risk Labels
# ==========================================================

def risk_label(probability):
    """
    probability can be either:
        float (0-1)
        int (0-100)
    """

    if probability <= 1:
        probability *= 100

    if probability < 65:
        return "Normal / Non-Anemic"

    elif probability < 85:
        return "Moderate Anemia Risk"

    else:
        return "High Anemia Risk"
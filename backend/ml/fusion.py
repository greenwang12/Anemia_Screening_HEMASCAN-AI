"""Fusion: combine eye + nail anemia probabilities.

Strategies (chained):
1. Noisy-OR  (default if no meta-learner)         P = 1 - (1-pe)(1-pn)
2. Confidence-weighted average                     w_e*pe + w_n*pn
3. Logistic-regression meta-learner (optional)     sigmoid(W @ [pe,pn,1])

The meta-learner weights are loaded from `fusion_meta.npz` when present.
To train them later, fit `sklearn.linear_model.LogisticRegression` on
(p_eye, p_nail) -> ground-truth anemia and save coef_ / intercept_ as
`weights` / `bias` numpy arrays in that .npz file.
"""
from __future__ import annotations

import logging
import math
import numpy as np

from .config import FUSION_META_PATH, risk_label

logger = logging.getLogger("ml.fusion")

_META = None
def _load_meta():
    global _META
    if _META is not None or not FUSION_META_PATH.exists():
        return _META
    try:
        data = np.load(FUSION_META_PATH)
        _META = (data["weights"].astype(float), float(data["bias"]))
        logger.info("Loaded fusion meta-learner weights")
    except Exception as e:
        logger.warning(f"Could not load fusion_meta: {e}")
        _META = None
    return _META


def _sigmoid(x): return 1.0 / (1.0 + math.exp(-x))


def fuse(eye: dict | None, nail: dict | None) -> dict:
    """Return fused screening dict with risk_percent / label / confidence."""
    if not eye and not nail:
        return {"risk_percent": 0, "risk_label": "Low", "confidence": 0.0, "modalities_used": [],
                "strategy": "none"}

    if eye and nail:
        pe = float(eye.get("p_anemia", eye.get("risk_percent", 0) / 100))
        pn = float(nail.get("p_anemia", nail.get("risk_percent", 0) / 100))
        ce = float(eye.get("confidence", 0.5))
        cn = float(nail.get("confidence", 0.5))

        # 1. Noisy-OR
        noisy_or = 1.0 - (1.0 - pe) * (1.0 - pn)
        # 2. Confidence-weighted average
        w_e, w_n = max(0.1, ce), max(0.1, cn)
        conf_avg = (pe * w_e + pn * w_n) / (w_e + w_n)
        # 3. Optional meta-learner
        meta = _load_meta()
        if meta is not None:
            w, b = meta
            meta_p = _sigmoid(float(w[0]) * pe + float(w[1]) * pn + b)
            strategy = "meta-logreg"
            fused = 0.5 * meta_p + 0.25 * noisy_or + 0.25 * conf_avg
        else:
            strategy = "noisy-or + conf-avg"
            fused = 0.6 * noisy_or + 0.4 * conf_avg
        # Disagreement penalty on confidence
        disagreement = abs(pe - pn)
        confidence = round(min(1.0, max(0.0, (ce + cn) / 2 + 0.05 - 0.5 * disagreement)), 3)
        modalities = ["eye", "nail"]
    elif eye:
        fused = float(eye.get("p_anemia", eye.get("risk_percent", 0) / 100))
        confidence = float(eye.get("confidence", 0.5))
        strategy = "eye-only"
        modalities = ["eye"]
    else:
        fused = float(nail.get("p_anemia", nail.get("risk_percent", 0) / 100))
        confidence = float(nail.get("confidence", 0.5))
        strategy = "nail-only"
        modalities = ["nail"]

    pct = int(round(max(0.0, min(1.0, fused)) * 100))
    return {
        "risk_percent": pct,
        "risk_label": risk_label(pct),
        "confidence": confidence,
        "modalities_used": modalities,
        "strategy": strategy,
    }

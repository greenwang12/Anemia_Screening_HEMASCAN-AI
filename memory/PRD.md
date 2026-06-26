# HemaScan — Anemia Screening (PRD)

## Original problem
Existing HemaScan app shipped a 6-class MobileNetV2 nail classifier with a
custom OpenCV clinical-feature analyzer. The user replaced the model with a
new **binary anemia / non-anemia EfficientNetB0** nail model and asked us to:

1. Remove the 6-class step entirely (no `NAIL_CLASSES`, no `top_class`, no
   `class_probs` in the API response).
2. Keep the OpenCV clinical-feature analyzer (pallor, koilonychia, ridging,
   brittleness, platonychia, yellowing) driving the rich story.
3. Blend the new model's `P(anemia)` with the OpenCV features (60% / 40%).
4. Keep Grad-CAM heatmaps but localise them strictly to the nail-bed (nail
   modality) and the conjunctiva ROI (eye modality).
5. After register, DO NOT auto-login the user — redirect to `/login` with a
   "Account created successfully" banner.
6. Rewrite the technical pipeline / fusion strategy copy in plain English for
   end users.

## Architecture
- Backend: FastAPI + SQLite + TensorFlow 2.x + OpenCV.
  - `ml/config.py` — model paths, NAIL_ANEMIA_INDEX polarity flag.
  - `ml/inference.py` — TTA inference + sigmoid→P(anemia) inversion + 60/40 blend.
  - `ml/nail_features.py` — 6-detector OpenCV analyzer + `combine_nail_signals`.
  - `ml/gradcam.py` — manual-walk grad model for nested submodels +
    OpenCV nail-bed ROI mask + conjunctiva ROI hard clamp.
  - `ml/pipeline.py` — async wrapper used by `server.py`.
- Frontend: React 18 + Tailwind + Sonner toasts.
- Models on disk:
  - `eye_mobilenetv2.keras`  (binary sigmoid, P(anemia) directly)
  - `nail_efficientnetb0.keras` (binary sigmoid, P(non-anemia) → inverted)

## What's been implemented (Jan 2026, this session)
- ✅ Binary nail head wired: `NAIL_ANEMIA_INDEX=1` inverts sigmoid → P(anemia).
- ✅ 60% model / 40% OpenCV blend (`MODEL_WEIGHT`, `FEATURE_WEIGHT`).
- ✅ All references to `class_probs`, `top_class`, `top_prob`,
   `p_class_anemia`, `NAIL_CLASSES` removed from API and UI.
- ✅ Grad-CAM rebuild for nested EfficientNetB0 submodel (uses submodel as
   feature extractor; manual walk with `training=False`).
- ✅ Nail Grad-CAM: HSV-based nail-bed detection +
   brightness/saturation-weighted mask hard-applied to the heatmap.
- ✅ Eye Grad-CAM: ROI rectangle hard-clamps the spread heatmap to the
   detected conjunctiva region.
- ✅ Register endpoint no longer returns a token / sets a cookie. Frontend
   `register()` no longer persists user. `/register` redirects to `/login`
   with `state.justRegistered=true`. `/login` shows the success banner and
   pre-fills the email.
- ✅ "Pipeline" / "Fusion strategy" / "Attention map" / "CNN" copy reworded:
   - Screen sidebar: "How it works" + plain bullets.
   - Results risk cards: "Eye check / Nail check / Overall estimate".
   - Results bottom: "Two signals, one easy-to-read result".
   - GradCamViewer chip: "Where the AI looked".
   - Findings card: "Paleness level" / "How sure we are".
   - Footer: "AI vision · Visual heatmaps · Combined score · Mobile-ready".
- ✅ `EYE_MODEL_PATH` auto-resolves `.keras` → `.h5` to match the nail loader.

## Test status
- Backend pytest: **11/11 pass** (iteration_4).
- Frontend E2E: **100%** of review-request flows verified end-to-end.

## Backlog / known minor
- gradcam_layer string reads "efficientnetb0" for the eye model too because
  the file's inner submodel is named that way. Cosmetic only.
- Optional: friendlier `gradcam_layer` label in the response.

## Test credentials
See `/app/memory/test_credentials.md`.

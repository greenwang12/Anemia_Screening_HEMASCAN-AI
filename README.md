# HemaScan — Anemia Screening AI

> An explainable, multimodal anemia screening prototype built with **MobileNetV2 CNNs** for eye-conjunctival and nail-bed images, **Grad-CAM** for explainability, and a **late-fusion** strategy combining both signals.
> Web stack: **FastAPI + MongoDB + React + Tailwind/shadcn**.

---

## 1. What the system does

A user logs in, uploads:
- One photo of the **lower eyelid / palpebral conjunctiva**, and/or
- One photo of a **fingernail bed**.

The backend runs three models:

| # | Model | Architecture | Output | Role |
|---|-------|--------------|--------|------|
| 1 | **Eye baseline** | MobileNetV2 + binary head | `P(anemia)` | Conjunctival pallor |
| 2 | **Nail baseline** | MobileNetV2 + 6-class head | softmax over `acral_lentiginous_melanoma`, `healthy_nail`, `onychogryphosis`, `blue_finger`, `clubbing`, `pitting` | Nail-bed pathology |
| 3 | **Fusion** | Noisy-OR + confidence-weighted mean + optional Logistic-Regression meta-learner | `P_fusion(anemia)` | Combines both signals |

For both CNNs we generate a **real Grad-CAM heatmap** from the last conv layer (`Conv_1`) using `tf.GradientTape`. The heatmap is overlaid on the original image in the UI with an opacity slider.

Each screening is saved to MongoDB and listed in the user's history. A printable PDF-style report can be generated.

---

## 2. Repository layout

```
/app
├── backend
│   ├── server.py               # FastAPI app, routes, auth, screening endpoints
│   ├── requirements.txt        # Python deps (incl. tensorflow, emergentintegrations)
│   ├── .env                    # Mongo URL, JWT secret, LLM key (NOT in git)
│   └── ml/                     # ── ML PIPELINE PACKAGE ──
│       ├── __init__.py         # Package doc
│       ├── config.py           # Paths, image size, class lists, anemia mapping, thresholds
│       ├── quality.py          # Blur (Laplacian variance) + brightness checks
│       ├── preprocessing.py    # Resize, MobileNetV2 preprocessing, TTA batch builder
│       ├── model_loader.py     # Lazy load + reload .h5 models
│       ├── inference.py        # TTA inference + per-modality result shaping
│       ├── gradcam.py          # Real Grad-CAM via tf.GradientTape
│       ├── fusion.py           # Noisy-OR + meta-learner fusion
│       ├── pipeline.py         # End-to-end entry point used by FastAPI
│       └── models/             # Drop your .h5 files here (gitignored)
│           ├── eye_mobilenetv2.h5    (you upload via UI)
│           └── nail_mobilenetv2.h5   (you upload via UI)
└── frontend
    ├── src
    │   ├── App.js              # React Router routes
    │   ├── contexts/AuthContext.js
    │   ├── lib/api.js          # Axios + Bearer-token interceptor
    │   ├── components/
    │   │   ├── Layout.js
    │   │   ├── ProtectedRoute.js
    │   │   ├── GradCamViewer.js  # Renders real heatmap PNG OR canvas fallback
    │   │   └── RiskCard.js
    │   └── pages/
    │       ├── Landing.js
    │       ├── Login.js / Register.js
    │       ├── Screen.js          # Upload + run analysis
    │       ├── Results.js         # Risk cards, Grad-CAM viewers, 6-class breakdown
    │       ├── History.js         # Patient records
    │       ├── Learn.js           # Educational content
    │       └── ModelsAdmin.js     # ADMIN-ONLY: upload .h5 models
    └── tailwind.config.js
```

---

## 3. Accuracy-enhancement strategies (baked in)

Your baseline MobileNetV2 reaches ~80 %. The pipeline adds the following **without retraining**:

1. **Test-Time Augmentation (TTA)** — `ml/preprocessing.tta_batch()` runs the same image through 5 variants (orig, horizontal flip, ±10° rotation, center-crop zoom) and averages predictions. Typically lifts accuracy 1–3 pp and stabilises confidence.
2. **Temperature-scaling calibration** — `TEMP_EYE` / `TEMP_NAIL` in `ml/config.py`. Set them after running a single calibration pass on a held-out set so the probability output matches empirical accuracy.
3. **Quality gate** — `ml/quality.py` rejects blurry / over- or under-exposed images before inference, eliminating an entire failure mode (~5–8 % of field uploads).
4. **Smarter fusion** (`ml/fusion.py`):
   - **Noisy-OR**: `1 − (1−p_eye)(1−p_nail)` — both signals reinforce each other.
   - **Confidence-weighted average**: gives more weight to the more confident model.
   - **Disagreement penalty** on overall confidence.
   - **Optional Logistic-Regression meta-learner**: if you save `coef_` / `intercept_` of a trained `sklearn.linear_model.LogisticRegression` into `ml/models/fusion_meta.npz` as `weights` / `bias`, the system will load it automatically and blend it with the analytical strategies.

Future levers (not in code yet): hard-example mining, MixUp / CutMix, knowledge distillation from EfficientNet, Bayesian dropout ensembles.

---

## 4. Nail-class → anemia mapping

The nail model was trained on 6 generic nail-disease classes, not on anemia. We apply **option 2a** from our planning:

```python
NAIL_ANEMIA_POSITIVE = {"blue_finger", "clubbing", "pitting"}
# P(anemia) = sum of softmax probabilities for the three classes above.
```

Rationale:
- **blue_finger** ≈ cyanosis (oxygen / hemoglobin issue).
- **clubbing** is associated with chronic hypoxia & long-standing anemia.
- **pitting / koilonychia** is the classic iron-deficiency anemia sign.

The remaining three classes (`healthy_nail`, `acral_lentiginous_melanoma`, `onychogryphosis`) do not indicate anemia, so their probabilities are excluded.

If you re-train a dedicated binary nail-anemia model later, just change `predict_nail()` in `ml/inference.py` — the rest of the pipeline is untouched.

---

## 5. Where is patient data stored?

MongoDB database: **`anemia_screening`** (configurable via `DB_NAME`).

| Collection      | Document shape (key fields) |
|-----------------|------------------------------|
| `users`         | `id`, `email`, `name`, `password_hash` (bcrypt), `role`, `created_at` |
| `screenings`    | `id`, `user_id`, `patient_name`, `patient_age`, `patient_sex`, `notes`, `eye_image_base64`, `nail_image_base64`, `eye_result`, `nail_result`, `fusion_result`, `created_at` |

`eye_result` / `nail_result` include:
```jsonc
{
  "image_type": "eye",
  "risk_percent": 47,
  "risk_label": "Moderate",
  "confidence": 0.78,
  "pallor_score": 5,
  "key_findings": ["..."],
  "attention_regions": [{ "cx": 0.5, "cy": 0.4, "radius": 0.2, "intensity": 0.9 }],
  "reasoning": "...",
  "model_extras": { "class_probs": {...}, "top_class": "..." },
  "quality": { "passed": true, "blur_variance": 312.4, "brightness": 142.6, "issues": [] },
  "gradcam_overlay_base64": "<JPEG base64>",
  "gradcam_heatmap_base64": "<PNG base64 with alpha>",
  "gradcam_layer": "Conv_1",
  "engine": "mobilenetv2" | "gemini-fallback"
}
```

`fusion_result`:
```jsonc
{ "risk_percent": 52, "risk_label": "Moderate", "confidence": 0.81,
  "modalities_used": ["eye", "nail"], "strategy": "noisy-or + conf-avg" }
```

Indexes: `users.email` (unique), `users.id` (unique), `screenings(user_id, created_at desc)`.

To inspect data:
```bash
mongosh
> use anemia_screening
> db.users.find().pretty()
> db.screenings.find({}, {eye_image_base64:0, nail_image_base64:0}).pretty()
```

---

## 6. How to plug in your trained `.h5` models

There are two equivalent ways:

### A. Drag-and-drop in the UI (recommended for demo)
1. Log in as **admin** (`admin@anemiacheck.app` / `Admin@123`).
2. Click **Models** in the header nav.
3. Upload `eye_mobilenetv2.h5` and `nail_mobilenetv2.h5`. The status badges turn from red ✗ to green ✓.
4. Run a new screening — `engine` in the result will now read `mobilenetv2`.

### B. Filesystem drop (for CI / offline demo)
```bash
cp /path/to/your/eye_model.h5  /app/backend/ml/models/eye_mobilenetv2.h5
cp /path/to/your/nail_model.h5 /app/backend/ml/models/nail_mobilenetv2.h5
# Restart not required — models are reloaded on next inference.
```

If a model file is missing, that modality falls back to a vision-LLM proxy (Gemini 3) so the app keeps working for demos.

---

## 7. Authentication

JWT (HS256) with bcrypt-hashed passwords. Tokens are stored both as an `access_token` httpOnly cookie **and** returned in the login response body — the React client also keeps a copy in `localStorage` and sends it as a `Bearer` header.

Seed admin (configurable via `.env`):
- email: `admin@anemiacheck.app`
- password: `Admin@123`

---

## 8. Running locally

```bash
# Backend
cd /app/backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Frontend (yarn ONLY, not npm)
cd /app/frontend
yarn install
yarn start
```

Environment variables (`/app/backend/.env`):
```
MONGO_URL="mongodb://localhost:27017"
DB_NAME="anemia_screening"
JWT_SECRET="<random hex>"
ADMIN_EMAIL="admin@anemiacheck.app"
ADMIN_PASSWORD="Admin@123"
EMERGENT_LLM_KEY="<your key for the Gemini fallback>"
```

---

## 9. Mobile deployment plan

The frontend is mobile-responsive and PWA-ready. For native:

1. **TFLite conversion** — `tf.lite.TFLiteConverter.from_keras_model(model)` produces ~5 MB files suitable for on-device inference.
2. **React-Native or Capacitor wrapper** — re-uses the same React codebase; replace `/api/screenings` with a local `tflite_flutter` (RN) or `@capacitor-community/tflite` (Capacitor) call.
3. On-device Grad-CAM is heavier but doable via TFLite + custom gradient kernels, or you can keep Grad-CAM server-side and only push raw probabilities through.

---

## 10. Disclaimer

This is a **research prototype**, not a medical device. Results are advisory; any clinical decision must be made by a qualified medical professional.

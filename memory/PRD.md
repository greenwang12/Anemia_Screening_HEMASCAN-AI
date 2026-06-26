# HemaScan — Anemia Screening AI

## Problem Statement
Build an anemia screening web app based on eye pallor and nail-bed analysis using a CNN architecture with Grad-CAM explainable AI. First build single baseline models for eye and nail, then a fusion model that combines both. Plan to deploy to mobile later.

## Architecture
- **Backend**: FastAPI + SQLite (aiosqlite via `backend/store.py`). Routes prefixed with `/api`.
- **Frontend**: React 19 + Tailwind + shadcn/ui (light theme, Outfit / Manrope fonts).
- **Auth**: JWT email+password (bcrypt). Token in httpOnly cookie + localStorage Bearer.
- **AI inference**: `emergentintegrations.LlmChat` with `gemini-3-flash-preview` acting as a CNN proxy.
  Returns structured JSON: `risk_percent`, `risk_label`, `confidence`, `key_findings`, `pallor_score`, `attention_regions`, `reasoning`.
- **Grad-CAM placeholder**: LLM returns normalized `attention_regions` (cx, cy, radius, intensity); frontend `<canvas>` renders a radial-gradient heatmap overlay with an opacity slider.
- **Fusion model**: Confidence-weighted average of the two baseline risk scores (late fusion).

## Personas
- Clinician / researcher running prototype screenings.
- Patient / public users wanting an at-home indicator.

## Core Requirements (locked)
- Register / Login / Logout (JWT)
- Upload eye image and/or nail image
- Run AI analysis → 3 results (eye baseline, nail baseline, fusion)
- Grad-CAM-style heatmap viewer for each image
- Patient history (MongoDB)
- Printable PDF report
- Educational content + FAQ
- Mobile-responsive

## Implemented (2026-02 → first finish)
- Full JWT auth (register, login, logout, /me) with bcrypt
- Admin user seeded: admin@anemiacheck.app / Admin@123
- POST /api/screenings → Gemini 3 Flash analyzes uploaded images in parallel, fusion computed server-side
- GET /api/screenings, GET /api/screenings/{id}, DELETE /api/screenings/{id}
- Frontend pages: Landing, Login, Register, Screen, Results, History, Learn
- Grad-CAM viewer with opacity slider, three RiskCards, findings cards, fusion explanation block
- Printable HTML report via window.print

## Implemented (2026-02 → second iteration: real CNN integration)
- Full `backend/ml/` package: config, quality, preprocessing (with TTA), model_loader, inference, gradcam, fusion, pipeline
- TensorFlow MobileNetV2 .h5 loading + lazy inference
- Real Grad-CAM via tf.GradientTape on last conv layer (`Conv_1`)
- Test-Time Augmentation (orig + h-flip + ±10° rotation + center-crop)
- Image quality gate (Laplacian variance + brightness)
- Nail 6-class softmax → anemia mapping (blue_finger + clubbing + pitting)
- Improved fusion: noisy-OR + confidence-weighted + disagreement penalty + optional sklearn meta-learner (`fusion_meta.npz`)
- Admin endpoints: GET/POST/DELETE /api/admin/models/{eye|nail}
- New /admin/models page with drag-drop .h5 upload
- Frontend GradCamViewer renders real heatmap PNG when present, falls back to canvas
- Results page shows nail 6-class probability breakdown with anemia-positive classes highlighted
- /app/README.md with full architecture documentation for teacher review

## Implemented (2026-02 → third iteration: SQLite migration)
- Replaced MongoDB/motor with `aiosqlite`-backed `backend/store.py` (Mongo-compatible drop-in API: `find_one`, `insert_one`, `delete_one`, `find().sort().limit().to_list()`)
- DB file auto-created at `backend/data/hemascan.db` on startup; tables `users` + `screenings` declared in `_TABLES` schema
- Admin user is seeded on startup if missing
- Verified end-to-end: register, login, /me, create/list/get/delete screening, duplicate-email 400, wrong-password 401, unauth 401
- `aiosqlite==0.22.1` added to `backend/requirements.txt`
- No external DB service needed — perfect for college mini-project / VS Code review

## Backlog / Future
**P1**
- Connect a real PyTorch/TensorFlow CNN backend (replace LLM proxy)
- Real Grad-CAM from the CNN gradients
- Capacitor/Expo wrapper for mobile deployment

**P2**
- Camera capture flow (vs. file upload)
- Rate limiting + brute-force lockout on /api/auth/login
- Doctor dashboard / multi-patient organization
- Aggregate analytics (population risk trends)
- Hemoglobin estimate regression
- Public read-only share links for reports

## Test Credentials
See `/app/memory/test_credentials.md`

## Known Limitations
- Images are stored as base64 in MongoDB (fine for demo, swap to object storage at scale)
- CORS uses wildcard + Bearer token (no cross-origin cookies)
- LLM call latency ~5–10s per image

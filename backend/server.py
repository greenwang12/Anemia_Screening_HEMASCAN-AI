from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import os
import json
import re
import uuid
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import bcrypt
import jwt
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field

# Local ML pipeline (real MobileNetV2 + Grad-CAM). Gracefully optional.
from ml import pipeline as ml_pipeline
from ml import model_loader as ml_loader
from ml.config import EYE_MODEL_PATH, NAIL_MODEL_PATH

# Lightweight SQLite-backed store (replaces MongoDB for this college project).
from store import SqliteDB

# -------------------- Config --------------------
MONGO_URL = os.environ.get("MONGO_URL", "")   # kept for backward compat; ignored
DB_NAME = os.environ.get("DB_NAME", "anemia_screening")
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@anemiacheck.app")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@123")
SQLITE_PATH = os.environ.get("SQLITE_PATH", str(ROOT_DIR / "data" / "hemascan.db"))

# -------------------- DB --------------------
db = SqliteDB(SQLITE_PATH)

# -------------------- App / Router --------------------
app = FastAPI(title="Anemia Screening AI")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("anemia")

# -------------------- Auth helpers --------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        # Fallback for browser-triggered downloads where headers can't be set
        token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"id": payload["sub"]})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user.pop("_id", None)
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------- Models --------------------
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ScreeningIn(BaseModel):
    eye_image_base64: Optional[str] = None
    nail_image_base64: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    notes: Optional[str] = None

# -------------------- Auth endpoints --------------------
def _set_auth_cookie(resp: Response, token: str):
    resp.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * 60 * 12,
        path="/",
    )

@api.post("/auth/register")
async def register(payload: RegisterIn):
    email = payload.email.lower().strip()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": email,
        "name": payload.name.strip(),
        "password_hash": hash_password(payload.password),
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(user_doc)
    # NOTE: we intentionally do NOT log the user in here. Account creation
    # only creates the row; the client should redirect to /login and the
    # user must sign in to obtain a session.
    return {"id": user_id, "email": email, "name": payload.name, "role": "user"}

@api.post("/auth/login")
async def login(payload: LoginIn, response: Response):
    email = payload.email.lower().strip()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user["id"], user["email"])
    _set_auth_cookie(response, token)
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name", ""),
        "role": user.get("role", "user"),
        "token": token,
    }

@api.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return {"ok": True}

@api.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user

# -------------------- Gemini vision analysis --------------------
ANALYSIS_PROMPT = """You are a clinical research assistant evaluating images for SIGNS of anemia.
This is for an educational screening prototype and NOT medical advice.

You will be given ONE image of either:
- The lower eyelid / palpebral conjunctiva (inner pink lining when eye is pulled down), OR
- A fingernail bed

Analyze the visible pallor (paleness) of the tissue. Healthy tissue: vibrant red/pink. Anemic tissue: pale, washed-out, light pink, whitish.

Return ONLY a valid JSON object (no markdown, no commentary) with this exact schema:
{
  "image_type": "eye" | "nail",
  "risk_percent": <integer 0-100 likelihood of anemia>,
  "risk_label": "Low" | "Moderate" | "High",
  "confidence": <float 0-1>,
  "key_findings": [<short string>, ...],   // 2-4 bullet observations
  "pallor_score": <integer 0-10 where 10 = severely pale>,
  "attention_regions": [
      {"cx": <float 0-1>, "cy": <float 0-1>, "radius": <float 0-1>, "intensity": <float 0-1>}
  ],  // 1-3 normalized hotspot circles where the model "looked"
  "reasoning": "<2-3 sentence explanation referencing visible cues>"
}

Rules:
- risk_percent < 35 -> "Low";  35-65 -> "Moderate";  > 65 -> "High".
- attention_regions coordinates are normalized [0,1] over the image (0,0 = top-left).
- If image is unusable (blurry, wrong subject, no tissue visible), set risk_percent=0, confidence=0, risk_label="Low", key_findings=["Image unusable"], reasoning="...", attention_regions=[].
- Output strictly the JSON object. No prose around it.
"""

def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip code fences if present
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])

def _strip_data_url(b64: str) -> str:
    if b64.startswith("data:"):
        return b64.split(",", 1)[1]
    return b64

async def analyze_image_with_gemini(image_base64: str, image_type: str) -> dict:
    """
    Local fallback when no LLM is available.
    Since the project has local ML models, this should normally never be used.
    """
    return {
        "image_type": image_type,
        "risk_percent": 0,
        "risk_label": "Low",
        "confidence": 0.0,
        "key_findings": ["LLM fallback disabled"],
        "pallor_score": 0,
        "attention_regions": [],
        "reasoning": "Local ML model is expected to perform the analysis.",
    }

def compute_fusion(eye: Optional[dict], nail: Optional[dict]) -> dict:
    """Late-fusion: weighted average of risk_percent and confidence."""
    if eye and nail:
        # confidence-weighted average
        w_e = max(0.1, eye["confidence"])
        w_n = max(0.1, nail["confidence"])
        total = w_e + w_n
        risk = round((eye["risk_percent"] * w_e + nail["risk_percent"] * w_n) / total)
        conf = round(min(1.0, (eye["confidence"] + nail["confidence"]) / 2 + 0.05), 3)
    elif eye:
        risk = eye["risk_percent"]
        conf = eye["confidence"]
    elif nail:
        risk = nail["risk_percent"]
        conf = nail["confidence"]
    else:
        risk, conf = 0, 0.0
    if risk < 35:
        label = "Low"
    elif risk <= 65:
        label = "Moderate"
    else:
        label = "High"
    return {
        "risk_percent": int(risk),
        "risk_label": label,
        "confidence": float(conf),
        "modalities_used": [m for m, v in [("eye", eye), ("nail", nail)] if v],
    }

# -------------------- Screening endpoints --------------------
async def _analyze_modality(image_b64: str, image_type: str) -> dict:
    """Try the real MobileNetV2 pipeline; if no .h5 model is present, fall back to Gemini."""
    if not image_b64:
        return None
    try:
        if image_type == "eye":
            res = await ml_pipeline.analyze_eye(image_b64)
        else:
            res = await ml_pipeline.analyze_nail(image_b64)
        if res is not None:
            res["engine"] = "mobilenetv2"
            return res
    except Exception as e:
        logger.exception(f"ML pipeline failed for {image_type}: {e}")
    # Fallback
        raise HTTPException(
        status_code=500,
        detail=f"Local {image_type} model failed. Gemini fallback is disabled."
    )

def _fuse_results(eye: Optional[dict], nail: Optional[dict]) -> dict:
    """Use ml.fusion if available; otherwise compute_fusion."""
    try:
        return ml_pipeline.fuse_results(eye, nail)
    except Exception:
        return compute_fusion(eye, nail)


@api.post("/screenings")
async def create_screening(payload: ScreeningIn, user: dict = Depends(get_current_user)):
    if not payload.eye_image_base64 and not payload.nail_image_base64:
        raise HTTPException(status_code=400, detail="At least one image (eye or nail) is required")

    tasks = []
    if payload.eye_image_base64:
        tasks.append(_analyze_modality(payload.eye_image_base64, "eye"))
    if payload.nail_image_base64:
        tasks.append(_analyze_modality(payload.nail_image_base64, "nail"))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    eye_result = None
    nail_result = None
    idx = 0
    if payload.eye_image_base64:
        r = results[idx]
        idx += 1
        if isinstance(r, Exception):
            raise r
        eye_result = r
    if payload.nail_image_base64:
        r = results[idx]
        if isinstance(r, Exception):
            raise r
        nail_result = r

    fusion = _fuse_results(eye_result, nail_result)

    screening_id = str(uuid.uuid4())
    doc = {
        "id": screening_id,
        "user_id": user["id"],
        "patient_name": payload.patient_name or "Anonymous",
        "patient_age": payload.patient_age,
        "patient_sex": payload.patient_sex,
        "notes": payload.notes,
        "eye_image_base64": payload.eye_image_base64,
        "nail_image_base64": payload.nail_image_base64,
        "eye_result": eye_result,
        "nail_result": nail_result,
        "fusion_result": fusion,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.screenings.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api.get("/screenings")
async def list_screenings(user: dict = Depends(get_current_user)):
    # Lightweight list (no images)
    cursor = db.screenings.find(
        {"user_id": user["id"]},
        {
            "_id": 0,
            "eye_image_base64": 0,
            "nail_image_base64": 0,
        },
    ).sort("created_at", -1).limit(100)
    return await cursor.to_list(length=100)

@api.get("/screenings/{screening_id}")
async def get_screening(screening_id: str, user: dict = Depends(get_current_user)):
    doc = await db.screenings.find_one({"id": screening_id, "user_id": user["id"]}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Screening not found")
    return doc

@api.delete("/screenings/{screening_id}")
async def delete_screening(screening_id: str, user: dict = Depends(get_current_user)):
    res = await db.screenings.delete_one({"id": screening_id, "user_id": user["id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Screening not found")
    return {"ok": True}

@api.get("/health")
async def health():
    return {"ok": True, "service": "anemia-screening"}

# -------------------- Admin model management --------------------
def _require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@api.get("/admin/models")
async def admin_models_status(_: dict = Depends(_require_admin)):
    info = ml_loader.models_status()
    return {
        **info,
        "eye_size_mb": round(EYE_MODEL_PATH.stat().st_size / 1024 / 1024, 2) if EYE_MODEL_PATH.exists() else None,
        "nail_size_mb": round(NAIL_MODEL_PATH.stat().st_size / 1024 / 1024, 2) if NAIL_MODEL_PATH.exists() else None,
    }

@api.post("/admin/models/{kind}")
async def admin_upload_model(kind: str, file: UploadFile = File(...), _: dict = Depends(_require_admin)):
    if kind not in ("eye", "nail"):
        raise HTTPException(status_code=400, detail="kind must be 'eye' or 'nail'")
    if not file.filename.lower().endswith((".h5", ".keras")):
        raise HTTPException(status_code=400, detail="Upload a .h5 or .keras file")
    dest = EYE_MODEL_PATH if kind == "eye" else NAIL_MODEL_PATH
    content = await file.read()
    dest.write_bytes(content)
    reloaded = ml_loader.reload_models()
    return {"ok": True, "kind": kind, "path": str(dest), "size_mb": round(len(content)/1024/1024, 2), "reloaded": reloaded}

@api.delete("/admin/models/{kind}")
async def admin_delete_model(kind: str, _: dict = Depends(_require_admin)):
    if kind not in ("eye", "nail"):
        raise HTTPException(status_code=400, detail="kind must be 'eye' or 'nail'")
    dest = EYE_MODEL_PATH if kind == "eye" else NAIL_MODEL_PATH
    if dest.exists():
        dest.unlink()
    ml_loader.reload_models()
    return {"ok": True, "kind": kind}

# -------------------- Codebase export (zip) --------------------
import io
import zipfile

EXPORT_EXCLUDE_DIRS = {
    "node_modules", ".git", "__pycache__", ".next", "build",
    ".cache", ".pytest_cache", ".venv", "venv", "dist", ".idea", ".vscode",
}
EXPORT_EXCLUDE_FILES = {".DS_Store", "yarn-error.log"}
EXPORT_MAX_FILE_MB = 50  # skip any single file larger than this

def _iter_export_files(root: Path):
    for path in root.rglob("*"):
        if any(part in EXPORT_EXCLUDE_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in EXPORT_EXCLUDE_FILES:
            continue
        try:
            if path.stat().st_size > EXPORT_MAX_FILE_MB * 1024 * 1024:
                continue
        except OSError:
            continue
        yield path

@api.get("/admin/export")
async def admin_export_codebase(_: dict = Depends(_require_admin)):
    """Zip the entire /app codebase (excluding node_modules, caches, .git)
    so the teacher can open it in VS Code. Streams as a download."""
    app_root = Path("/app")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in _iter_export_files(app_root):
            arcname = path.relative_to(app_root)
            try:
                zf.write(path, arcname=str(Path("hemascan") / arcname))
            except (OSError, PermissionError):
                continue
    buf.seek(0)
    filename = f"hemascan-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.zip"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

# -------------------- Startup --------------------
@app.on_event("startup")
async def startup():
    # Create SQLite tables / indices on first boot.
    await db.init_schema()
    await db.users.create_index("email", unique=True)
    await db.users.create_index("id", unique=True)
    await db.screenings.create_index([("user_id", 1), ("created_at", -1)])
    # Seed admin
    existing = await db.users.find_one({"email": ADMIN_EMAIL})
    if not existing:
        await db.users.insert_one({
            "id": str(uuid.uuid4()),
            "email": ADMIN_EMAIL,
            "name": "Admin",
            "password_hash": hash_password(ADMIN_PASSWORD),
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Admin user seeded")

@app.on_event("shutdown")
async def shutdown():
    db.close()

# -------------------- Mount --------------------
app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://localhost(:\d+)?|https?://127\.0\.0\.1(:\d+)?|https://.*\.preview\.emergentagent\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
